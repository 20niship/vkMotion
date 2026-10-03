#include "ik_solver.hpp"

#include <Eigen/Dense>
#include <cmath>
#include <random>

namespace vkmotion {

namespace {

// below ~1e-3 rad, float32 get_axis() loses 1-w^2 to cancellation and zeroes the rotational Jacobian.
constexpr float kDelta       = 1e-2f;
constexpr double kPosEps     = 5e-4;
constexpr double kRotEps     = 5e-4;
constexpr double kLambda     = 0.05;
constexpr float kMaxStep     = 0.2f;
constexpr int kMaxIterations = 200;

Pose ee_pose(const RobotConfig& robot, const std::vector<float>& q) {
  std::vector<cutil::Mat4f> link_world = robot.forward_kinematics(JointState{q, {}});
  const cutil::Mat4f& t                = link_world.back();

  Pose p;
  p.position = cutil::Vec3f(t(0, 3), t(1, 3), t(2, 3));
  cutil::Mat<3, 3, float> r;
  for(unsigned i = 0; i < 3; i++)
    for(unsigned j = 0; j < 3; j++) r(i, j) = t(i, j);
  p.quaternion = cutil::Quatf::from_mat3(r);
  return p;
}

// Rotation vector (axis * angle, shortest path) taking `from` to `to`.
cutil::Vec3f orientation_error(const cutil::Quatf& from, const cutil::Quatf& to) {
  cutil::Quatf err = to * from.inverse();
  if(err.w < 0.0f) err = cutil::Quatf(-err.x, -err.y, -err.z, -err.w);
  err = err.normalized();
  return err.get_axis() * static_cast<float>(err.get_angle());
}

Eigen::Matrix<double, 6, 1> pose_error(const Pose& goal, const Pose& current) {
  cutil::Vec3f pos_err = goal.position - current.position;
  cutil::Vec3f rot_err = orientation_error(current.quaternion, goal.quaternion);
  Eigen::Matrix<double, 6, 1> e;
  e << pos_err[0], pos_err[1], pos_err[2], rot_err[0], rot_err[1], rot_err[2];
  return e;
}

}  // namespace

IKSolver::IKSolver(const IKSolverConfig& config) : config_(config) {
  if(config_.world_config) {
    RobotWorldConfig wc;
    wc.robot_config = config_.robot_config;
    wc.world_config = config_.world_config;
    wc.use_vulkan   = false;
    robot_world_    = std::make_unique<RobotWorld>(wc);
  }
}

std::vector<float> IKSolver::random_joint_values() const {
  static thread_local std::mt19937 rng(std::random_device{}());
  const auto& lower = config_.robot_config->joint_position_limits_lower();
  const auto& upper = config_.robot_config->joint_position_limits_upper();
  std::vector<float> q(lower.size());
  for(size_t i = 0; i < q.size(); i++) {
    std::uniform_real_distribution<float> dist(lower[i], upper[i]);
    q[i] = dist(rng);
  }
  return q;
}

std::optional<JointState> IKSolver::solve_from_seed(const Pose& goal, const std::vector<float>& seed_q) const {
  const RobotConfig& robot = *config_.robot_config;
  const auto& lower         = robot.joint_position_limits_lower();
  const auto& upper          = robot.joint_position_limits_upper();
  const size_t dof           = robot.dof();

  std::vector<float> q = seed_q;

  for(int iter = 0; iter < kMaxIterations; iter++) {
    Pose current                           = ee_pose(robot, q);
    Eigen::Matrix<double, 6, 1> e          = pose_error(goal, current);
    if(e.head<3>().norm() < kPosEps && e.tail<3>().norm() < kRotEps) {
      return JointState{q, {}};
    }

    Eigen::MatrixXd J(6, static_cast<int>(dof));
    for(size_t i = 0; i < dof; i++) {
      std::vector<float> q_plus = q, q_minus = q;
      q_plus[i] += kDelta;
      q_minus[i] -= kDelta;
      Pose p_plus  = ee_pose(robot, q_plus);
      Pose p_minus = ee_pose(robot, q_minus);
      cutil::Vec3f dpos        = (p_plus.position - p_minus.position) / (2.0f * kDelta);
      cutil::Vec3f drot        = orientation_error(p_minus.quaternion, p_plus.quaternion) / (2.0f * kDelta);
      J(0, static_cast<int>(i)) = dpos[0];
      J(1, static_cast<int>(i)) = dpos[1];
      J(2, static_cast<int>(i)) = dpos[2];
      J(3, static_cast<int>(i)) = drot[0];
      J(4, static_cast<int>(i)) = drot[1];
      J(5, static_cast<int>(i)) = drot[2];
    }

    Eigen::MatrixXd JJt = J * J.transpose() + (kLambda * kLambda) * Eigen::MatrixXd::Identity(6, 6);
    Eigen::VectorXd y   = JJt.ldlt().solve(e);
    Eigen::VectorXd dq  = J.transpose() * y;

    for(size_t i = 0; i < dof; i++) {
      float step = static_cast<float>(dq(static_cast<int>(i)));
      step       = std::max(-kMaxStep, std::min(kMaxStep, step));
      q[i]       = std::max(lower[i], std::min(upper[i], q[i] + step));
    }
  }
  return std::nullopt;
}

std::optional<JointState> IKSolver::solve_single(const Pose& goal_pose, const std::optional<JointState>& seed) {
  std::vector<float> seed_q = seed ? seed->position : random_joint_values();
  std::optional<JointState> result = solve_from_seed(goal_pose, seed_q);
  if(result && robot_world_ && !robot_world_->is_collision_free(*result)) return std::nullopt;
  return result;
}

std::vector<std::optional<JointState>> IKSolver::solve_batch(const std::vector<Pose>& goal_poses) {
  std::vector<std::optional<JointState>> results;
  results.reserve(goal_poses.size());
  for(const Pose& goal : goal_poses) {
    std::optional<JointState> found;
    for(int s = 0; s < config_.num_seeds; s++) {
      std::optional<JointState> candidate = solve_from_seed(goal, random_joint_values());
      if(!candidate) continue;
      if(robot_world_ && !robot_world_->is_collision_free(*candidate)) continue;
      found = candidate;
      break;
    }
    results.push_back(found);
  }
  return results;
}

}  // namespace vkmotion
