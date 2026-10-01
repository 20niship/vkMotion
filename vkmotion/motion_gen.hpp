#pragma once

#include <cutil/ref.hpp>
#include <string>
#include <vector>

#include "robot_config.hpp"
#include "types.hpp"
#include "world_config.hpp"

namespace vkmotion {

struct MotionGenConfig {
  cutil::Ref<RobotConfig> robot_config;
  cutil::Ref<WorldConfig> world_config;  // nullptr: collision ignored
  int num_trajopt_seeds = 12;
  bool use_vulkan = true;  // 軌道修正時の衝突判定(RobotWorld)に適用。IKSolver内部の衝突判定は現状CPU固定

  static MotionGenConfig load_from_robot_config(cutil::Ref<RobotConfig> robot_config, cutil::Ref<WorldConfig> world_config);
};

struct MotionGenPlanConfig {
  int max_attempts         = 10;
  bool enable_graph_planner = true;  // TODO: 幾何プランナー未実装のため現状無視される
  float timeout_seconds    = 5.0f;
};

struct MotionGenResult {
  bool success = false;
  std::vector<JointState> trajectory;
  std::vector<float> interpolated_times;
  std::string status;
};

// Pose-goal motion generation (cuRobo: MotionGen). ponytail: L-BFGS+MPPIの代わりに直線補間+距離場勾配の局所repairのみ。
class MotionGen {
 public:
  explicit MotionGen(const MotionGenConfig& config) : config_(config) {}

  void warmup() {}  // no-op: Vulkanパイプラインのウォームアップは当該経路実装時に追加

  MotionGenResult plan_single(const JointState& start_state, const Pose& goal_pose, const MotionGenPlanConfig& plan_config = {});
  MotionGenResult plan_single_js(const JointState& start_state, const JointState& goal_state, const MotionGenPlanConfig& plan_config = {});

 private:
  MotionGenConfig config_;
};

}  // namespace vkmotion
