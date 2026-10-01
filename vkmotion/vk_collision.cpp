#include "vk_collision.hpp"

#include <algorithm>
#include <vkmotion/vkmotion_config.hpp>

#include "vk_buffer.hpp"
#include "vk_context.hpp"
#include "vk_pipeline.hpp"

namespace vkmotion {

namespace {

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
  uint32_t num_robot_spheres;
  uint32_t num_cuboids;
  uint32_t num_spheres;
  uint32_t num_cylinders;
  uint32_t num_capsules;
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

constexpr uint32_t kLocalSizeX = 64;

void ensure_capacity(VkContext& ctx, std::unique_ptr<VkStorageBuffer>& buf, size_t bytes) {
  if(!buf || buf->size() < bytes) buf = std::make_unique<VkStorageBuffer>(ctx, std::max<size_t>(bytes, 1));
}

}  // namespace

VkCollisionPipeline::VkCollisionPipeline(VkContext& ctx) : ctx_(ctx) {
  std::string spv_path = std::string(VKMOTION_SHADER_DIR) + "/robot_collision.comp.spv";
  pipeline_             = std::make_unique<VkComputePipeline>(ctx_, spv_path, /*num_storage_bindings=*/6, /*push_constant_bytes=*/sizeof(PushConstants));
}

VkCollisionPipeline::~VkCollisionPipeline() = default;

std::vector<float> VkCollisionPipeline::compute(const std::vector<CollisionSphere>& world_spheres, const WorldConfig& world) {
  std::vector<float> robot_spheres(std::max<size_t>(1, world_spheres.size()) * 4, 0.0f);
  for(size_t i = 0; i < world_spheres.size(); i++) {
    robot_spheres[i * 4 + 0] = world_spheres[i].center[0];
    robot_spheres[i * 4 + 1] = world_spheres[i].center[1];
    robot_spheres[i * 4 + 2] = world_spheres[i].center[2];
    robot_spheres[i * 4 + 3] = world_spheres[i].radius;
  }

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
    const auto& o              = world.sphere()[i];
    spheres[i].pos_radius[0]   = o.pose.position[0];
    spheres[i].pos_radius[1]   = o.pose.position[1];
    spheres[i].pos_radius[2]   = o.pose.position[2];
    spheres[i].pos_radius[3]   = o.radius;
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

  ensure_capacity(ctx_, robot_spheres_buf_, robot_spheres.size() * sizeof(float));
  robot_spheres_buf_->upload(robot_spheres.data(), robot_spheres.size() * sizeof(float));
  ensure_capacity(ctx_, cuboids_buf_, cuboids.size() * sizeof(GpuCuboid));
  cuboids_buf_->upload(cuboids.data(), cuboids.size() * sizeof(GpuCuboid));
  ensure_capacity(ctx_, spheres_buf_, spheres.size() * sizeof(GpuSphereObstacle));
  spheres_buf_->upload(spheres.data(), spheres.size() * sizeof(GpuSphereObstacle));
  ensure_capacity(ctx_, cylinders_buf_, cylinders.size() * sizeof(GpuCylinder));
  cylinders_buf_->upload(cylinders.data(), cylinders.size() * sizeof(GpuCylinder));
  ensure_capacity(ctx_, capsules_buf_, capsules.size() * sizeof(GpuCapsule));
  capsules_buf_->upload(capsules.data(), capsules.size() * sizeof(GpuCapsule));
  ensure_capacity(ctx_, out_buf_, robot_spheres.size() / 4 * sizeof(float));

  pipeline_->update_bindings({
      robot_spheres_buf_->descriptor_info(),
      cuboids_buf_->descriptor_info(),
      spheres_buf_->descriptor_info(),
      cylinders_buf_->descriptor_info(),
      capsules_buf_->descriptor_info(),
      out_buf_->descriptor_info(),
  });

  PushConstants pc{};
  pc.num_robot_spheres = static_cast<uint32_t>(world_spheres.size());
  pc.num_cuboids       = static_cast<uint32_t>(world.cuboid().size());
  pc.num_spheres       = static_cast<uint32_t>(world.sphere().size());
  pc.num_cylinders     = static_cast<uint32_t>(world.cylinder().size());
  pc.num_capsules      = static_cast<uint32_t>(world.capsule().size());

  uint32_t group_count = (pc.num_robot_spheres + kLocalSizeX - 1) / kLocalSizeX;
  if(group_count == 0) group_count = 1;
  pipeline_->dispatch(group_count, &pc, sizeof(pc));

  std::vector<float> result(world_spheres.size());
  if(!result.empty()) out_buf_->download(result.data(), result.size() * sizeof(float));
  return result;
}

}  // namespace vkmotion
