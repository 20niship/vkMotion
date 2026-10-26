#pragma once

#include <cutil/ref.hpp>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "allowed_collision_matrix.hpp"
#include "constraints.hpp"
#include "robot_config.hpp"
#include "robot_world.hpp"
#include "types.hpp"
#include "world_config.hpp"

namespace vkmotion {

class IKSolver;
class FastCollision;

struct MotionGenConfig {
  cutil::Ref<RobotConfig> robot_config;
  cutil::Ref<WorldConfig> world_config;  // nullptr: collision ignored
  int num_trajopt_seeds = 12;
  bool use_vulkan = true;  // 軌道修正時の衝突判定(RobotWorld)に適用。IKSolver内部の衝突判定は現状CPU固定
  AllowedCollisionMatrix acm;                // self-collision除外ペア(URDFの隣接リンクに加えて)
  float max_acceleration = 15.0f;            // 関節加速度の上限[rad/s^2] (時間パラメタライズ用。cuRoboのfranka既定と同じ)
  float search_step = 0.1f;                  // 大域探索(RRT-Connect)の辺の検査間隔[m]。粗くすると探索が速くなる(VAMPは約0.2〜0.4m相当)。最終経路は必ずcollision_stepで検証し直す
  float collision_step = 0.05f;              // 経路の短縮・最終検証の辺の検査間隔[m](障害物の厚み+球径をまたがない目安)
  float collision_margin = 0.005f;           // 経路の内部点に持たせる安全余裕[m]。0なら障害物に接した経路になりうる
  float gpu_handoff_s = 0.001f;              // gpu_search時、まずCPUのRRT-Connectをこの秒数だけ試し、解けなければGPU探索に渡す(簡単な問題はdispatchの固定費0.5ms以下で済むCPUの方が速い)。CPUは0.4ms前後で解けるのが多数派で、GPUはsubmitの固定費が約0.5msかかるため。0ならGPU優先
  bool gpu_search = false;                   // trueかつuse_vulkanなら、大域探索(RRT-Connect)をGPU上で並列に走らせる(shaders/rrt_gpu.comp)。解けなければCPUのRRT-Connectに戻る
  int num_threads = 1;                       // 大域探索(RRT-Connect)を独立シードで並列起動する本数。最初に解いたものを採用
  int num_ik_goals = 3;                      // 姿勢ゴール時にIKで集める衝突なし解の数(全てをゴール木の根にする)
  uint64_t seed = 0;                         // 0: 非決定的、それ以外: 再現可能
  std::optional<ConstraintSet> constraints;  // nullopt: 制約無視(既定、既存の挙動と完全に同じ)

  static MotionGenConfig load_from_robot_config(cutil::Ref<RobotConfig> robot_config, cutil::Ref<WorldConfig> world_config);
};

struct MotionGenPlanConfig {
  int max_attempts         = 10;
  bool enable_graph_planner = true;  // trueなら直線補間+局所repairが全滅した後にRRT-Connectフォールバックを試す
  float timeout_seconds    = 5.0f;
  int shortcut_iterations  = 60;  // 経路短縮(ランダムな2点を直線で結び直す)の試行回数。増やすと経路が短くなるが時間も増える
};

struct MotionGenResult {
  bool success = false;
  std::vector<JointState> trajectory;
  std::vector<float> interpolated_times;
  std::string status;
};

// Pose-goal motion generation (cuRobo: MotionGen)。RobotWorld(Vulkan込み)はインスタンス生存中1つだけ作り使い回す。
class MotionGen {
 public:
  explicit MotionGen(const MotionGenConfig& config);
  ~MotionGen();

  void warmup();
  RobotWorld& robot_world_for_test() { return robot_world(); }  // テスト用(GPU探索が使えるかの確認)

  MotionGenResult plan_single(const JointState& start_state, const Pose& goal_pose, const MotionGenPlanConfig& plan_config = {});
  MotionGenResult plan_single_js(const JointState& start_state, const JointState& goal_state, const MotionGenPlanConfig& plan_config = {});

 private:
  RobotWorld& robot_world();
  IKSolver& ik_solver();
  MotionGenResult plan_fast(const FastCollision& fc, const JointState& start, const std::vector<JointState>& goals, const MotionGenPlanConfig& plan_config);

  MotionGenConfig config_;
  std::unique_ptr<RobotWorld> robot_world_;
  std::unique_ptr<IKSolver> ik_;
  uint64_t plan_counter_ = 0;
};

}  // namespace vkmotion
