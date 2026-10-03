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
struct GpuPlane {
  float pos[4];
  float quat[4];
};
struct GpuConvexMeta {
  float pos[4];
  float quat[4];
  int32_t offset_count[4];
};
struct GpuSdfMeta {
  float pos[4];
  float quat[4];
  float dims[4];
  int32_t res_offset[4];
};

struct PushConstants {
  uint32_t num_robot_spheres;
  uint32_t num_cuboids;
  uint32_t num_spheres;
  uint32_t num_cylinders;
  uint32_t num_capsules;
  uint32_t num_planes;
  uint32_t num_convex;
  uint32_t num_sdf;
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
  pipeline_             = std::make_unique<VkComputePipeline>(ctx_, spv_path, /*num_storage_bindings=*/11, /*push_constant_bytes=*/sizeof(PushConstants));
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

  std::vector<GpuPlane> planes(std::max<size_t>(1, world.plane().size()));
  for(size_t i = 0; i < world.plane().size(); i++) pack_pose(world.plane()[i].pose, planes[i].pos, planes[i].quat);

  std::vector<GpuConvexMeta> convex_metas(std::max<size_t>(1, world.convex_mesh().size()));
  std::vector<float> convex_verts;
  for(size_t i = 0; i < world.convex_mesh().size(); i++) {
    const auto& o = world.convex_mesh()[i];
    pack_pose(o.pose, convex_metas[i].pos, convex_metas[i].quat);
    convex_metas[i].offset_count[0] = static_cast<int32_t>(convex_verts.size() / 4);
    convex_metas[i].offset_count[1] = static_cast<int32_t>(o.vertices.size());
    for(const auto& v : o.vertices) {
      convex_verts.push_back(v[0]);
      convex_verts.push_back(v[1]);
      convex_verts.push_back(v[2]);
      convex_verts.push_back(0.0f);
    }
  }
  if(convex_verts.empty()) convex_verts.assign(4, 0.0f);

  std::vector<GpuSdfMeta> sdf_metas(std::max<size_t>(1, world.sdf_grid().size()));
  std::vector<float> sdf_data;
  for(size_t i = 0; i < world.sdf_grid().size(); i++) {
    const auto& o = world.sdf_grid()[i];
    pack_pose(o.pose, sdf_metas[i].pos, sdf_metas[i].quat);
    sdf_metas[i].dims[0]       = o.dims[0];
    sdf_metas[i].dims[1]       = o.dims[1];
    sdf_metas[i].dims[2]       = o.dims[2];
    sdf_metas[i].res_offset[0] = o.resolution[0];
    sdf_metas[i].res_offset[1] = o.resolution[1];
    sdf_metas[i].res_offset[2] = o.resolution[2];
    sdf_metas[i].res_offset[3] = static_cast<int32_t>(sdf_data.size());
    sdf_data.insert(sdf_data.end(), o.distances.begin(), o.distances.end());
  }
  if(sdf_data.empty()) sdf_data.assign(1, 0.0f);

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
  ensure_capacity(ctx_, planes_buf_, planes.size() * sizeof(GpuPlane));
  planes_buf_->upload(planes.data(), planes.size() * sizeof(GpuPlane));
  ensure_capacity(ctx_, convex_meta_buf_, convex_metas.size() * sizeof(GpuConvexMeta));
  convex_meta_buf_->upload(convex_metas.data(), convex_metas.size() * sizeof(GpuConvexMeta));
  ensure_capacity(ctx_, convex_verts_buf_, convex_verts.size() * sizeof(float));
  convex_verts_buf_->upload(convex_verts.data(), convex_verts.size() * sizeof(float));
  ensure_capacity(ctx_, sdf_meta_buf_, sdf_metas.size() * sizeof(GpuSdfMeta));
  sdf_meta_buf_->upload(sdf_metas.data(), sdf_metas.size() * sizeof(GpuSdfMeta));
  ensure_capacity(ctx_, sdf_data_buf_, sdf_data.size() * sizeof(float));
  sdf_data_buf_->upload(sdf_data.data(), sdf_data.size() * sizeof(float));
  ensure_capacity(ctx_, out_buf_, robot_spheres.size() / 4 * sizeof(float));

  pipeline_->update_bindings({
      robot_spheres_buf_->descriptor_info(),
      cuboids_buf_->descriptor_info(),
      spheres_buf_->descriptor_info(),
      cylinders_buf_->descriptor_info(),
      capsules_buf_->descriptor_info(),
      planes_buf_->descriptor_info(),
      convex_meta_buf_->descriptor_info(),
      convex_verts_buf_->descriptor_info(),
      sdf_meta_buf_->descriptor_info(),
      sdf_data_buf_->descriptor_info(),
      out_buf_->descriptor_info(),
  });

  PushConstants pc{};
  pc.num_robot_spheres = static_cast<uint32_t>(world_spheres.size());
  pc.num_cuboids       = static_cast<uint32_t>(world.cuboid().size());
  pc.num_spheres       = static_cast<uint32_t>(world.sphere().size());
  pc.num_cylinders     = static_cast<uint32_t>(world.cylinder().size());
  pc.num_capsules      = static_cast<uint32_t>(world.capsule().size());
  pc.num_planes        = static_cast<uint32_t>(world.plane().size());
  pc.num_convex        = static_cast<uint32_t>(world.convex_mesh().size());
  pc.num_sdf           = static_cast<uint32_t>(world.sdf_grid().size());

  uint32_t group_count = (pc.num_robot_spheres + kLocalSizeX - 1) / kLocalSizeX;
  if(group_count == 0) group_count = 1;
  pipeline_->dispatch(group_count, &pc, sizeof(pc));

  std::vector<float> result(world_spheres.size());
  if(!result.empty()) out_buf_->download(result.data(), result.size() * sizeof(float));
  return result;
}

}  // namespace vkmotion
