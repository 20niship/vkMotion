#include "motion_gen.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <random>

#include "ik_solver.hpp"
#include "robot_world.hpp"

namespace vkmotion {

namespace {

constexpr int kNumWaypoints        = 21;
constexpr float kRepairDelta       = 1e-2f;
constexpr float kRepairStep        = 0.05f;
constexpr int kRepairMaxIterations = 80;

float min_collision_distance(const RobotWorld& rw, const JointState& js) {
  std::vector<float> d = rw.get_collision_distance(js);
  float m = std::numeric_limits<float>::infinity();
  for(float v : d) m = std::min(m, v);
  return m;
}

JointState random_state(const RobotConfig& robot) {
  static thread_local std::mt19937 rng(std::random_device{}());
  const auto& lower = robot.joint_position_limits_lower();
  const auto& upper = robot.joint_position_limits_upper();
  JointState js;
  js.position.resize(lower.size());
  for(size_t i = 0; i < lower.size(); i++) {
    std::uniform_real_distribution<float> dist(lower[i], upper[i]);
    js.position[i] = dist(rng);
  }
  return js;
}

std::vector<JointState> linear_interpolate(const JointState& a, const JointState& b, int n) {
  std::vector<JointState> out;
  out.reserve(static_cast<size_t>(n));
  const size_t dof = a.position.size();
  for(int i = 0; i < n; i++) {
    float t = n > 1 ? static_cast<float>(i) / static_cast<float>(n - 1) : 0.0f;
    JointState js;
    js.position.resize(dof);
    for(size_t j = 0; j < dof; j++) js.position[j] = a.position[j] + t * (b.position[j] - a.position[j]);
    out.push_back(std::move(js));
  }
  return out;
}

// 数値勾配でmin_collision_distanceを登る局所repair。勾配ゼロ/反復上限は失敗として呼び出し元に再試行させる。
bool repair_waypoint(const RobotWorld& rw, const RobotConfig& robot, std::vector<float>& q) {
  if(min_collision_distance(rw, JointState{q, {}}) > 0.0f) return true;

  const auto& lower = robot.joint_position_limits_lower();
  const auto& upper = robot.joint_position_limits_upper();
  const size_t dof  = q.size();

  for(int iter = 0; iter < kRepairMaxIterations; iter++) {
    std::vector<float> grad(dof, 0.0f);
    for(size_t i = 0; i < dof; i++) {
      std::vector<float> qp = q, qm = q;
      qp[i] += kRepairDelta;
      qm[i] -= kRepairDelta;
      float fp = min_collision_distance(rw, JointState{qp, {}});
      float fm = min_collision_distance(rw, JointState{qm, {}});
      grad[i]  = (fp - fm) / (2.0f * kRepairDelta);
    }
    float norm = 0.0f;
    for(float g : grad) norm += g * g;
    norm = std::sqrt(norm);
    if(norm < 1e-8f) break;
    for(size_t i = 0; i < dof; i++) {
      q[i] += kRepairStep * grad[i] / norm;
      q[i] = std::max(lower[i], std::min(upper[i], q[i]));
    }
    if(min_collision_distance(rw, JointState{q, {}}) > 0.0f) return true;
  }
  return false;
}

}  // namespace

MotionGenConfig MotionGenConfig::load_from_robot_config(cutil::Ref<RobotConfig> robot_config, cutil::Ref<WorldConfig> world_config) {
  MotionGenConfig cfg;
  cfg.robot_config = robot_config;
  cfg.world_config = world_config;
  return cfg;
}

MotionGenResult MotionGen::plan_single(const JointState& start_state, const Pose& goal_pose, const MotionGenPlanConfig& plan_config) {
  IKSolverConfig ik_cfg;
  ik_cfg.robot_config = config_.robot_config;
  ik_cfg.world_config = config_.world_config;
  ik_cfg.num_seeds    = std::max(1, config_.num_trajopt_seeds);
  IKSolver ik(ik_cfg);

  std::vector<std::optional<JointState>> ik_results = ik.solve_batch({goal_pose});
  if(ik_results.empty() || !ik_results[0]) {
    MotionGenResult result;
    result.status = "IK failed to find a collision-free goal configuration";
    return result;
  }
  return plan_single_js(start_state, *ik_results[0], plan_config);
}

MotionGenResult MotionGen::plan_single_js(const JointState& start_state, const JointState& goal_state, const MotionGenPlanConfig& plan_config) {
  MotionGenResult result;

  cutil::Ref<WorldConfig> world = config_.world_config ? config_.world_config : cutil::make_ref<WorldConfig>();
  RobotWorld rw(RobotWorldConfig{config_.robot_config, world, config_.use_vulkan});
  const RobotConfig& robot = *config_.robot_config;

  if(!rw.is_collision_free(start_state) || !rw.is_collision_free(goal_state)) {
    result.status = "start or goal state is in collision";
    return result;
  }

  int total_attempts = std::max(1, std::min(plan_config.max_attempts, std::max(1, config_.num_trajopt_seeds)));

  for(int attempt = 0; attempt < total_attempts; attempt++) {
    std::vector<JointState> traj;
    if(attempt == 0) {
      traj = linear_interpolate(start_state, goal_state, kNumWaypoints);
    } else {
      JointState via  = random_state(robot);
      int half        = kNumWaypoints / 2;
      traj            = linear_interpolate(start_state, via, half);
      auto second_half = linear_interpolate(via, goal_state, kNumWaypoints - half + 1);
      traj.insert(traj.end(), second_half.begin() + 1, second_half.end());
    }

    bool all_ok = true;
    for(size_t i = 1; i + 1 < traj.size(); i++) {
      if(!rw.is_collision_free(traj[i]) && !repair_waypoint(rw, robot, traj[i].position)) {
        all_ok = false;
        break;
      }
    }
    if(all_ok) {
      result.success   = true;
      result.trajectory = std::move(traj);
      result.interpolated_times.resize(result.trajectory.size());
      for(size_t i = 0; i < result.trajectory.size(); i++)
        result.interpolated_times[i] = plan_config.timeout_seconds * static_cast<float>(i) / static_cast<float>(result.trajectory.size());
      result.status = "success";
      return result;
    }
  }

  result.status = "failed to find a collision-free trajectory after max_attempts";
  return result;
}

}  // namespace vkmotion
