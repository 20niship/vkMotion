#pragma once

#include <cstdint>
#include <memory>
#include <vector>

#include "types.hpp"
#include "world_config.hpp"

namespace vkmotion {

class VkContext;
class VkComputePipeline;
class VkStorageBuffer;
class RobotConfig;

// Dispatches shaders/repair_waypoints.comp: FK + gradient-descent loop runs entirely GPU-resident, no per-iteration host round trip. Serial-chain robots only.
class VkRepairPipeline {
 public:
  VkRepairPipeline(VkContext& ctx, const RobotConfig& robot);
  ~VkRepairPipeline();
  VkRepairPipeline(const VkRepairPipeline&)            = delete;
  VkRepairPipeline& operator=(const VkRepairPipeline&) = delete;

  // qsをその場で更新。戻り値はqsと同順の収束成否(CPU版repair_waypoints_batchと同じ契約)。
  std::vector<bool> repair(std::vector<std::vector<float>*>& qs, const WorldConfig& world, float repair_step, float repair_delta, uint32_t max_iterations);

 private:
  VkContext& ctx_;
  std::unique_ptr<VkComputePipeline> pipeline_;
  std::unique_ptr<VkStorageBuffer> joints_buf_, robot_spheres_buf_, dof_limits_buf_;
  std::unique_ptr<VkStorageBuffer> cuboids_buf_, spheres_buf_, cylinders_buf_, capsules_buf_;
  std::unique_ptr<VkStorageBuffer> planes_buf_, convex_meta_buf_, convex_verts_buf_, sdf_meta_buf_, sdf_data_buf_;
  std::unique_ptr<VkStorageBuffer> initial_q_buf_, out_q_buf_, out_success_buf_;
  uint32_t num_joints_        = 0;
  uint32_t num_dof_           = 0;
  uint32_t num_robot_spheres_ = 0;
};

}  // namespace vkmotion
