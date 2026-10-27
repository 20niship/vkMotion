#pragma once

#include <cutil/ref.hpp>
#include <memory>
#include <optional>
#include <vector>

#include "robot_config.hpp"
#include "robot_world.hpp"
#include "types.hpp"
#include "world_config.hpp"

namespace vkmotion {

struct IKSolverConfig {
  cutil::Ref<RobotConfig> robot_config;
  cutil::Ref<WorldConfig> world_config;  // nullptr: collision ignored
  int num_seeds = 512;  // 試すランダムシード数の上限(収束・衝突なしの解が見つかった時点で打ち切る)
  AllowedCollisionMatrix acm;
  uint64_t seed = 0;  // 0: std::random_device(非決定的)、それ以外: 再現可能な乱数列
};

// Numerical IK (cuRobo: curobo.wrap.reacher.ik_solver.IKSolver), CPU-only Levenberg-Marquardt with analytic Jacobian (CompiledKinematics).
class IKSolver {
 public:
  explicit IKSolver(const IKSolverConfig& config);

  // ponytail: end-effector = link_names().back() only; multi-tip robots need per-tip solving, add when needed.
  std::optional<JointState> solve_single(const Pose& goal_pose, const std::optional<JointState>& seed = std::nullopt);
  std::vector<std::optional<JointState>> solve_batch(const std::vector<Pose>& goal_poses);

  // 異なる解を最大max_solutions個集める(world_configがあれば衝突なしのものだけ)。ランダムシードをmax_seeds個まで試す。
  // 複数ゴールを同時に扱うプランナー(MotionGen::plan_single)向け。
  // first_seedを渡すと最初の試行をそこから始める(現在姿勢に近い解が得られ、経路が短くなる)。
  std::vector<JointState> solve_multiple(const Pose& goal_pose, int max_solutions, int max_seeds, const std::optional<JointState>& first_seed = std::nullopt);

 private:
  IKSolverConfig config_;
  std::unique_ptr<RobotWorld> robot_world_;  // null when config_.world_config is null

  std::optional<JointState> solve_from_seed(const Pose& goal, const std::vector<float>& seed_q) const;
  std::vector<float> random_joint_values();

  uint64_t rng_state_ = 0;
  uint32_t next_random();
};

}  // namespace vkmotion
