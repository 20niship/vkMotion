#pragma once

#include <cstdint>
#include <memory>
#include <vector>

#include "fast_collision.hpp"

namespace vkmotion {

class VkContext;
class VkComputeSequence;
class VkStorageBuffer;

struct GpuRrtSettings {
  float margin      = 0.0f;   // 内部点・新規ノードの安全余裕[m]
  float range       = 1.0f;   // 1回のextendで伸ばす最大L2距離[rad]
  float motion_step = 0.05f;  // 辺の検査間隔(Lipschitz上界, FastCollision::set_motion_stepと同じ意味)
  uint32_t seed     = 1;
  uint32_t first_warps = 256;    // 最初のラウンドで木を伸ばすワープ数
  uint32_t max_warps   = 896;   // ワープ数の上限(first_warpsから、ラウンドごとに倍々で増やす)
  uint32_t iters       = 1;     // 1ラウンドで各ワープが行うRRT-Connect反復の数
  uint32_t rounds_per_submit = 8;   // 1回のsubmitに積むラウンド数(解けたら残りのdispatchは0本になる)
  uint32_t max_submits       = 4;   // 解けなければ続けてsubmitする回数の上限
};

class VkRrtPipeline {
 public:
  static constexpr size_t kMaxDof = 7;
  static constexpr size_t kMaxGoals = 8;
  
  VkRrtPipeline(VkContext& ctx, const GpuRobotData& robot, const std::vector<float>& joint_reach, const std::vector<float>& lower, const std::vector<float>& upper);
  ~VkRrtPipeline();
  VkRrtPipeline(const VkRrtPipeline&)            = delete;
  VkRrtPipeline& operator=(const VkRrtPipeline&) = delete;

  void set_world(const GpuWorldData& world);
  // ロボット専用に生成したシェーダを使っているか(glslcが無い等で汎用版に戻っていればfalse)
  bool specialized() const { return specialized_; }
  // 解が見つかればtrueで path=[start,...,goalのどれか]。start/goalsは衝突なしであること。goalsは最大kMaxGoals個。
  // margin_used: 経路が満たしている余裕(木が伸びずにGPU側で縮めた場合は settings.margin より小さい)
  bool plan(const std::vector<float>& start, const std::vector<std::vector<float>>& goals, const GpuRrtSettings& settings, std::vector<std::vector<float>>& path, float* margin_used = nullptr);

 private:
  VkContext& ctx_;
  GpuRobotData robot_;
  uint32_t gen_ = 0;
  bool specialized_ = false;
  std::vector<float> reach_, lower_, upper_;
  int num_boxes_ = 0, num_balls_ = 0;
  std::unique_ptr<VkComputeSequence> seq_;
  std::unique_ptr<VkStorageBuffer> params_, steps_meta_, steps_data_, spheres_, cluster_meta_, cluster_bs_, group_meta_, group_bs_, pairs_, boxes_, balls_, result_, ctl_, tree_;
};

}  // namespace vkmotion
