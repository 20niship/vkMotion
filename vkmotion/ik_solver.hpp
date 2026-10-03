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
  int num_seeds = 32;
};

// Numerical IK (cuRobo: curobo.wrap.reacher.ik_solver.IKSolver), CPU-only DLS with finite-difference Jacobian.
class IKSolver {
 public:
  explicit IKSolver(const IKSolverConfig& config);

  // ponytail: end-effector = link_names().back() only; multi-tip robots need per-tip solving, add when needed.
  std::optional<JointState> solve_single(const Pose& goal_pose, const std::optional<JointState>& seed = std::nullopt);
  std::vector<std::optional<JointState>> solve_batch(const std::vector<Pose>& goal_poses);

 private:
  IKSolverConfig config_;
  std::unique_ptr<RobotWorld> robot_world_;  // null when config_.world_config is null

  std::optional<JointState> solve_from_seed(const Pose& goal, const std::vector<float>& seed_q) const;
  std::vector<float> random_joint_values() const;
};

}  // namespace vkmotion
