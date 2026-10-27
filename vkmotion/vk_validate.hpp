#pragma once

#include <cstdint>
#include <memory>
#include <vector>

#include "fast_collision.hpp"

namespace vkmotion {

class VkContext;
class VkComputePipeline;
class VkStorageBuffer;

// shaders/robot_validate.comp のラッパー。コンフィグ(関節角)を渡すだけでFK→障害物衝突→自己衝突まで1 dispatchで判定する。
// ロボット記述は構築時に1回、世界はset_world()のときだけアップロードする(従来のvk_collisionは毎回全バッファを再送していた)。
class VkValidatePipeline {
 public:
  VkValidatePipeline(VkContext& ctx, const GpuRobotData& robot);
  ~VkValidatePipeline();
  VkValidatePipeline(const VkValidatePipeline&)            = delete;
  VkValidatePipeline& operator=(const VkValidatePipeline&) = delete;

  void set_world(const GpuWorldData& world);
  // q: n * dof の行優先。out[i]=1: 衝突なし。
  void validate(const float* q, size_t n, float margin, uint8_t* out);

 private:
  VkContext& ctx_;
  GpuRobotData robot_;
  int num_boxes_ = 0, num_balls_ = 0;
  std::unique_ptr<VkComputePipeline> pipeline_;
  std::unique_ptr<VkStorageBuffer> cfg_, steps_meta_, steps_data_, spheres_, cluster_meta_, cluster_bs_, group_meta_, group_bs_, pairs_, boxes_, balls_, out_;
};

}  // namespace vkmotion
