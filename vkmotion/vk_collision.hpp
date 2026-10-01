#pragma once

#include <memory>
#include <vector>

#include "types.hpp"
#include "world_config.hpp"

namespace vkmotion {

class VkContext;
class VkComputePipeline;
class VkStorageBuffer;

// Dispatches shaders/robot_collision.comp. Caches the pipeline/descriptor set across calls.
class VkCollisionPipeline {
 public:
  explicit VkCollisionPipeline(VkContext& ctx);
  ~VkCollisionPipeline();
  VkCollisionPipeline(const VkCollisionPipeline&)            = delete;
  VkCollisionPipeline& operator=(const VkCollisionPipeline&) = delete;

  // world_spheres may concatenate several JointStates' spheres to batch them into one dispatch.
  std::vector<float> compute(const std::vector<CollisionSphere>& world_spheres, const WorldConfig& world);

 private:
  VkContext& ctx_;
  std::unique_ptr<VkComputePipeline> pipeline_;
  // 呼び出しごとの再確保コストを避けるため、必要サイズがcapacityを超えたときだけ再確保して使い回す。
  std::unique_ptr<VkStorageBuffer> robot_spheres_buf_, cuboids_buf_, spheres_buf_, cylinders_buf_, capsules_buf_, out_buf_;
};

}  // namespace vkmotion
