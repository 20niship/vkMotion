#pragma once

#include <cstdint>
#include <memory>
#include <vector>

#include <vulkan/vulkan.h>

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

// バッチ探索の1問題ぶんの入力/出力。世界(箱・球)は問題ごとに別でよい(箱は32個、球は8個まで)。
struct GpuRrtProblem {
  std::vector<float> start;
  std::vector<std::vector<float>> goals;
  const GpuWorldData* world = nullptr;
};
struct GpuRrtResult {
  bool solved = false;
  std::vector<std::vector<float>> path;
  float margin = 0.0f;  // 経路が満たしている余裕(木が伸びずにGPU側で縮めた場合は settings.margin より小さい)
};

class VkRrtPipeline {
 public:
  static constexpr size_t kMaxDof = 7;
  static constexpr size_t kMaxGoals = 8;
  static constexpr size_t kMaxBoxes = 32;
  static constexpr size_t kMaxBalls = 8;
  static constexpr size_t kMaxBatch = 1024;

  VkRrtPipeline(VkContext& ctx, const GpuRobotData& robot, const std::vector<float>& joint_reach, const std::vector<float>& lower, const std::vector<float>& upper);
  ~VkRrtPipeline();
  VkRrtPipeline(const VkRrtPipeline&)            = delete;
  VkRrtPipeline& operator=(const VkRrtPipeline&) = delete;

  void set_world(const GpuWorldData& world);
  // ロボット専用に生成したシェーダを使っているか(glslcが無い等で汎用版に戻っていればfalse)
  bool specialized() const { return specialized_; }
  // 解が見つかればtrueで path=[start,...,goalのどれか]。start/goalsは衝突なしであること。goalsは最大kMaxGoals個。
  // margin_used: 経路が満たしている余裕(木が伸びずにGPU側で縮めた場合は settings.margin より小さい)
  // 複数の問題(それぞれ別の世界)を1回のsubmitの列でまとめて解く。results[i]が問題iの結果。
  // 問題あたりのワープ数は settings.first_warps..max_warps(バッチが大きいときは小さくすること。全体で数百〜千ワープが目安)。
  void plan_batch(const std::vector<GpuRrtProblem>& problems, const GpuRrtSettings& settings, std::vector<GpuRrtResult>& results);
  bool plan(const std::vector<float>& start, const std::vector<std::vector<float>>& goals, const GpuRrtSettings& settings, std::vector<std::vector<float>>& path, float* margin_used = nullptr);

 private:
  VkContext& ctx_;
  GpuRobotData robot_;
  uint32_t gen_ = 0;
  bool specialized_ = false;
  std::vector<float> reach_, lower_, upper_;
  GpuWorldData world_;  // set_world()で渡された世界(plan()が使う)
  uint32_t tree_cap_ = 4096;  // 1本の木のノード数上限(plan_batchが問題数に応じて決める)
  size_t capacity_ = 0;  // buffers_が確保している問題数
  void ensure_capacity(size_t problems);
  std::unique_ptr<VkComputeSequence> seq_;
  std::unique_ptr<VkStorageBuffer> params_, steps_meta_, steps_data_, spheres_, cluster_meta_, cluster_bs_, group_meta_, group_bs_, pairs_, boxes_, balls_, result_, ctl_, tree_;
  std::vector<VkDescriptorBufferInfo> bindings() const;
};

}  // namespace vkmotion
