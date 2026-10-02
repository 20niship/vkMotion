#include "vk_repair.hpp"

#include <cutil/quaternion.hpp>
#include <stdexcept>
#include <vkmotion/vkmotion_config.hpp>

#include "kinematics.hpp"
#include "robot_config.hpp"
#include "vk_buffer.hpp"
#include "vk_context.hpp"
#include "vk_pipeline.hpp"

namespace vkmotion {

namespace {

constexpr int kMaxDof    = 16;
constexpr int kMaxJoints = 16;

struct GpuJoint {
  float origin_pos[4];
  float origin_quat[4];
  float axis[4];
  int32_t meta[4];
};
struct GpuRobotSphere {
  float center_radius[4];
  int32_t depth_pad[4];
};
struct GpuCuboid {
  float pos[4];
  float quat[4];
  float dims[4];
};
struct GpuSphereObstacle {
  float pos_radius[4];
};
struct GpuCylinder {
  float pos[4];
  float quat[4];
  float radius_height[4];
};
struct GpuCapsule {
  float pos[4];
  float quat[4];
  float base_radius[4];
  float tip[4];
};

struct PushConstants {
  uint32_t num_joints;
  uint32_t num_dof;
  uint32_t num_robot_spheres;
  uint32_t num_cuboids;
  uint32_t num_sphere_obstacles;
  uint32_t num_cylinders;
  uint32_t num_capsules;
  uint32_t max_iterations;
  float repair_step;
  float repair_delta;
};

void pack_pose(const Pose& p, float pos[4], float quat[4]) {
  pos[0]  = p.position[0];
  pos[1]  = p.position[1];
  pos[2]  = p.position[2];
  pos[3]  = 0.0f;
  quat[0] = p.quaternion.x;
  quat[1] = p.quaternion.y;
  quat[2] = p.quaternion.z;
  quat[3] = p.quaternion.w;
}

void ensure_capacity(VkContext& ctx, std::unique_ptr<VkStorageBuffer>& buf, size_t bytes) {
  if(!buf || buf->size() < bytes) buf = std::make_unique<VkStorageBuffer>(ctx, std::max<size_t>(bytes, 1));
}

}  // namespace

VkRepairPipeline::VkRepairPipeline(VkContext& ctx, const RobotConfig& robot) : ctx_(ctx) {
  std::string spv_path = std::string(VKMOTION_SHADER_DIR) + "/repair_waypoints.comp.spv";
  pipeline_             = std::make_unique<VkComputePipeline>(ctx_, spv_path, /*num_storage_bindings=*/10, /*push_constant_bytes=*/sizeof(PushConstants));

  const UrdfModel& model = robot.urdf_model();
  SerialChain chain       = build_serial_chain(model);
  num_joints_             = static_cast<uint32_t>(chain.joint_order.size());
  num_dof_                = static_cast<uint32_t>(robot.dof());
  if(num_joints_ > static_cast<uint32_t>(kMaxJoints) || num_dof_ > static_cast<uint32_t>(kMaxDof))
    throw std::runtime_error("vkmotion: VkRepairPipeline robot exceeds compile-time MAX_JOINTS/MAX_DOF of the GPU repair shader");

  const auto& active_names = robot.joint_names();
  std::vector<GpuJoint> gpu_joints(num_joints_);
  for(uint32_t k = 0; k < num_joints_; k++) {
    const UrdfJoint& j = model.joints[static_cast<size_t>(chain.joint_order[k])];
    cutil::Quatf origin_quat = cutil::Quatf::from_euler(j.origin_rpy);
    GpuJoint gj{};
    gj.origin_pos[0]  = j.origin_xyz[0];
    gj.origin_pos[1]  = j.origin_xyz[1];
    gj.origin_pos[2]  = j.origin_xyz[2];
    gj.origin_quat[0] = origin_quat.x;
    gj.origin_quat[1] = origin_quat.y;
    gj.origin_quat[2] = origin_quat.z;
    gj.origin_quat[3] = origin_quat.w;
    gj.axis[0]        = j.axis[0];
    gj.axis[1]        = j.axis[1];
    gj.axis[2]        = j.axis[2];
    gj.meta[0]        = (j.type == UrdfJointType::kPrismatic) ? 1 : (j.type == UrdfJointType::kFixed ? 2 : 0);
    gj.meta[1]        = -1;
    for(size_t qi = 0; qi < active_names.size(); qi++) {
      if(active_names[qi] == j.name) {
        gj.meta[1] = static_cast<int32_t>(qi);
        break;
      }
    }
    gpu_joints[k] = gj;
  }

  const auto& local_spheres = robot.collision_spheres_local();
  num_robot_spheres_          = static_cast<uint32_t>(local_spheres.size());
  std::vector<GpuRobotSphere> gpu_spheres(std::max<size_t>(1, local_spheres.size()));
  for(size_t i = 0; i < local_spheres.size(); i++) {
    const auto& s = local_spheres[i];
    int link_idx   = model.find_link(s.link_name);
    int depth      = (link_idx >= 0) ? chain.link_depth[static_cast<size_t>(link_idx)] : 0;
    if(depth < 0) depth = 0;
    GpuRobotSphere gs{};
    gs.center_radius[0] = s.center[0];
    gs.center_radius[1] = s.center[1];
    gs.center_radius[2] = s.center[2];
    gs.center_radius[3] = s.radius;
    gs.depth_pad[0]      = depth;
    gpu_spheres[i]       = gs;
  }

  std::vector<float> limits(std::max<size_t>(1, num_dof_) * 2, 0.0f);
  for(uint32_t i = 0; i < num_dof_; i++) {
    limits[i * 2 + 0] = robot.joint_position_limits_lower()[i];
    limits[i * 2 + 1] = robot.joint_position_limits_upper()[i];
  }

  ensure_capacity(ctx_, joints_buf_, gpu_joints.size() * sizeof(GpuJoint));
  joints_buf_->upload(gpu_joints.data(), gpu_joints.size() * sizeof(GpuJoint));
  ensure_capacity(ctx_, robot_spheres_buf_, gpu_spheres.size() * sizeof(GpuRobotSphere));
  robot_spheres_buf_->upload(gpu_spheres.data(), gpu_spheres.size() * sizeof(GpuRobotSphere));
  ensure_capacity(ctx_, dof_limits_buf_, limits.size() * sizeof(float));
  dof_limits_buf_->upload(limits.data(), limits.size() * sizeof(float));
}

VkRepairPipeline::~VkRepairPipeline() = default;

std::vector<bool> VkRepairPipeline::repair(std::vector<std::vector<float>*>& qs, const WorldConfig& world, float repair_step, float repair_delta,
                                            uint32_t max_iterations) {
  std::vector<bool> success(qs.size(), true);
  if(qs.empty()) return success;

  std::vector<GpuCuboid> cuboids(std::max<size_t>(1, world.cuboid().size()));
  for(size_t i = 0; i < world.cuboid().size(); i++) {
    const auto& o = world.cuboid()[i];
    pack_pose(o.pose, cuboids[i].pos, cuboids[i].quat);
    cuboids[i].dims[0] = o.dims[0];
    cuboids[i].dims[1] = o.dims[1];
    cuboids[i].dims[2] = o.dims[2];
  }
  std::vector<GpuSphereObstacle> spheres(std::max<size_t>(1, world.sphere().size()));
  for(size_t i = 0; i < world.sphere().size(); i++) {
    const auto& o            = world.sphere()[i];
    spheres[i].pos_radius[0] = o.pose.position[0];
    spheres[i].pos_radius[1] = o.pose.position[1];
    spheres[i].pos_radius[2] = o.pose.position[2];
    spheres[i].pos_radius[3] = o.radius;
  }
  std::vector<GpuCylinder> cylinders(std::max<size_t>(1, world.cylinder().size()));
  for(size_t i = 0; i < world.cylinder().size(); i++) {
    const auto& o = world.cylinder()[i];
    pack_pose(o.pose, cylinders[i].pos, cylinders[i].quat);
    cylinders[i].radius_height[0] = o.radius;
    cylinders[i].radius_height[1] = o.height;
  }
  std::vector<GpuCapsule> capsules(std::max<size_t>(1, world.capsule().size()));
  for(size_t i = 0; i < world.capsule().size(); i++) {
    const auto& o = world.capsule()[i];
    pack_pose(o.pose, capsules[i].pos, capsules[i].quat);
    capsules[i].base_radius[0] = o.base[0];
    capsules[i].base_radius[1] = o.base[1];
    capsules[i].base_radius[2] = o.base[2];
    capsules[i].base_radius[3] = o.radius;
    capsules[i].tip[0]         = o.tip[0];
    capsules[i].tip[1]         = o.tip[1];
    capsules[i].tip[2]         = o.tip[2];
  }

  std::vector<float> initial_q(qs.size() * num_dof_);
  for(size_t i = 0; i < qs.size(); i++)
    for(uint32_t j = 0; j < num_dof_; j++) initial_q[i * num_dof_ + j] = (*qs[i])[j];

  ensure_capacity(ctx_, cuboids_buf_, cuboids.size() * sizeof(GpuCuboid));
  cuboids_buf_->upload(cuboids.data(), cuboids.size() * sizeof(GpuCuboid));
  ensure_capacity(ctx_, spheres_buf_, spheres.size() * sizeof(GpuSphereObstacle));
  spheres_buf_->upload(spheres.data(), spheres.size() * sizeof(GpuSphereObstacle));
  ensure_capacity(ctx_, cylinders_buf_, cylinders.size() * sizeof(GpuCylinder));
  cylinders_buf_->upload(cylinders.data(), cylinders.size() * sizeof(GpuCylinder));
  ensure_capacity(ctx_, capsules_buf_, capsules.size() * sizeof(GpuCapsule));
  capsules_buf_->upload(capsules.data(), capsules.size() * sizeof(GpuCapsule));
  ensure_capacity(ctx_, initial_q_buf_, initial_q.size() * sizeof(float));
  initial_q_buf_->upload(initial_q.data(), initial_q.size() * sizeof(float));
  ensure_capacity(ctx_, out_q_buf_, initial_q.size() * sizeof(float));
  ensure_capacity(ctx_, out_success_buf_, qs.size() * sizeof(uint32_t));

  pipeline_->update_bindings({
      joints_buf_->descriptor_info(),
      robot_spheres_buf_->descriptor_info(),
      dof_limits_buf_->descriptor_info(),
      cuboids_buf_->descriptor_info(),
      spheres_buf_->descriptor_info(),
      cylinders_buf_->descriptor_info(),
      capsules_buf_->descriptor_info(),
      initial_q_buf_->descriptor_info(),
      out_q_buf_->descriptor_info(),
      out_success_buf_->descriptor_info(),
  });

  PushConstants pc{};
  pc.num_joints            = num_joints_;
  pc.num_dof               = num_dof_;
  pc.num_robot_spheres     = num_robot_spheres_;
  pc.num_cuboids           = static_cast<uint32_t>(world.cuboid().size());
  pc.num_sphere_obstacles = static_cast<uint32_t>(world.sphere().size());
  pc.num_cylinders         = static_cast<uint32_t>(world.cylinder().size());
  pc.num_capsules          = static_cast<uint32_t>(world.capsule().size());
  pc.max_iterations        = max_iterations;
  pc.repair_step            = repair_step;
  pc.repair_delta           = repair_delta;

  pipeline_->dispatch(static_cast<uint32_t>(qs.size()), &pc, sizeof(pc));

  std::vector<float> out_q(initial_q.size());
  out_q_buf_->download(out_q.data(), out_q.size() * sizeof(float));
  std::vector<uint32_t> out_success(qs.size());
  out_success_buf_->download(out_success.data(), out_success.size() * sizeof(uint32_t));

  for(size_t i = 0; i < qs.size(); i++) {
    for(uint32_t j = 0; j < num_dof_; j++) (*qs[i])[j] = out_q[i * num_dof_ + j];
    success[i] = out_success[i] != 0;
  }
  return success;
}

}  // namespace vkmotion
