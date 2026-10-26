#include "ik_solver.hpp"

#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <random>

#include "fast_kinematics.hpp"

namespace vkmotion {

namespace {

constexpr double kPosEps     = 1e-4;  // [m]
constexpr double kRotEps     = 5e-4;  // [rad]
constexpr int kMaxIterations = 60;
constexpr float kMaxStep     = 0.5f;  // 1反復で動かす関節ノルムの上限[rad]

struct GoalFrame {
  float r[9];
  float p[3];
};

GoalFrame to_frame(const Pose& pose) {
  GoalFrame g{};
  cutil::Mat<3, 3, float> m = pose.quaternion.normalized().to_mat3();
  for(int i = 0; i < 3; i++) {
    g.p[i] = pose.position[i];
    for(int j = 0; j < 3; j++) g.r[i * 3 + j] = m(static_cast<unsigned>(i), static_cast<unsigned>(j));
  }
  return g;
}

// 現在姿勢(Rc)から目標姿勢(Rg)へ向かう回転ベクトル(軸*角, world座標系)。 R = Rg * Rc^T
Eigen::Vector3d rotation_error(const float* rg, const float* rc) {
  double R[9];
  for(int i = 0; i < 3; i++)
    for(int j = 0; j < 3; j++) R[i * 3 + j] = double(rg[i * 3 + 0]) * rc[j * 3 + 0] + double(rg[i * 3 + 1]) * rc[j * 3 + 1] + double(rg[i * 3 + 2]) * rc[j * 3 + 2];
  Eigen::Vector3d v(0.5 * (R[7] - R[5]), 0.5 * (R[2] - R[6]), 0.5 * (R[3] - R[1]));  // sin(theta) * axis
  const double c = std::clamp(0.5 * (R[0] + R[4] + R[8] - 1.0), -1.0, 1.0);
  const double s = v.norm();
  if(s > 1e-9) return v / s * std::atan2(s, c);
  if(c > 0.0) return Eigen::Vector3d::Zero();
  // theta ~ pi: 軸は (R+I)/2 の対角から
  Eigen::Vector3d a(std::sqrt(std::max(0.0, 0.5 * (R[0] + 1.0))), std::sqrt(std::max(0.0, 0.5 * (R[4] + 1.0))), std::sqrt(std::max(0.0, 0.5 * (R[8] + 1.0))));
  return a * M_PI;
}

}  // namespace

IKSolver::IKSolver(const IKSolverConfig& config) : config_(config), rng_state_(config.seed ? config.seed : std::random_device{}() * 2654435761ull + 12345) {
  if(config_.world_config) {
    RobotWorldConfig wc;
    wc.robot_config = config_.robot_config;
    wc.world_config = config_.world_config;
    wc.use_vulkan   = false;
    wc.acm          = config_.acm;
    robot_world_    = std::make_unique<RobotWorld>(wc);
  }
}

uint32_t IKSolver::next_random() {
  // splitmix64 -> 上位32bit
  rng_state_ += 0x9E3779B97F4A7C15ULL;
  uint64_t z = rng_state_;
  z          = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
  z          = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
  return static_cast<uint32_t>((z ^ (z >> 31)) >> 32);
}

std::vector<float> IKSolver::random_joint_values() {
  const auto& lower = config_.robot_config->joint_position_limits_lower();
  const auto& upper = config_.robot_config->joint_position_limits_upper();
  std::vector<float> q(lower.size());
  for(size_t i = 0; i < q.size(); i++) q[i] = lower[i] + (upper[i] - lower[i]) * (static_cast<float>(next_random() >> 8) * (1.0f / 16777216.0f));
  return q;
}

// Levenberg-Marquardt。損失が下がらなければ減衰を増やして巻き戻す(特異姿勢・関節リミット際で発散しない)。
std::optional<JointState> IKSolver::solve_from_seed(const Pose& goal_pose, const std::vector<float>& seed_q) const {
  const RobotConfig& robot       = *config_.robot_config;
  const CompiledKinematics& kin  = robot.compiled_kinematics();
  const auto& lower              = robot.joint_position_limits_lower();
  const auto& upper              = robot.joint_position_limits_upper();
  const size_t dof               = kin.dof();
  const int ee                   = static_cast<int>(kin.num_links()) - 1;
  const GoalFrame goal           = to_frame(goal_pose);

  std::vector<float> q = seed_q;
  std::vector<Frame> frames(kin.num_links());
  std::vector<float> Jf(6 * dof);
  Eigen::MatrixXd J(6, static_cast<Eigen::Index>(dof));
  Eigen::Matrix<double, 6, 1> e;

  auto error = [&](const std::vector<float>& qq, Eigen::Matrix<double, 6, 1>& err) {
    kin.forward(qq.data(), frames.data());
    const Frame& t = frames[static_cast<size_t>(ee)];
    err.head<3>()  = Eigen::Vector3d(goal.p[0] - t.p[0], goal.p[1] - t.p[1], goal.p[2] - t.p[2]);
    err.tail<3>()  = rotation_error(goal.r, t.r);
  };

  error(q, e);
  double cost   = e.squaredNorm();
  double lambda = 1e-2;
  for(int iter = 0; iter < kMaxIterations; iter++) {
    if(e.head<3>().norm() < kPosEps && e.tail<3>().norm() < kRotEps) return JointState{q, {}};

    kin.jacobian(frames.data(), ee, Jf.data());  // framesはerror(q)の結果のまま
    for(int i = 0; i < 6; i++)
      for(size_t j = 0; j < dof; j++) J(i, static_cast<Eigen::Index>(j)) = Jf[static_cast<size_t>(i) * dof + j];

    Eigen::Matrix<double, 6, 6> A = J * J.transpose();
    bool accepted                 = false;
    for(int tries = 0; tries < 6 && !accepted; tries++) {
      Eigen::Matrix<double, 6, 6> M = A;
      M.diagonal().array() += lambda;
      Eigen::VectorXd dq = J.transpose() * M.ldlt().solve(e);
      double n           = dq.norm();
      if(n > kMaxStep) dq *= kMaxStep / n;
      std::vector<float> qn(dof);
      for(size_t j = 0; j < dof; j++) qn[j] = std::clamp(q[j] + static_cast<float>(dq(static_cast<Eigen::Index>(j))), lower[j], upper[j]);
      Eigen::Matrix<double, 6, 1> en;
      error(qn, en);
      double cn = en.squaredNorm();
      if(cn < cost) {
        q = qn;
        e = en;
        cost = cn;
        lambda = std::max(lambda * 0.3, 1e-8);
        accepted = true;
      } else {
        lambda *= 4.0;
        error(q, e);  // framesを現在のqに戻す(ヤコビアン再計算用)
      }
    }
    if(!accepted) break;  // 局所解(リミット際/特異)
  }
  if(e.head<3>().norm() < kPosEps && e.tail<3>().norm() < kRotEps) return JointState{q, {}};
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
    std::vector<JointState> sols = solve_multiple(goal, 1, config_.num_seeds);
    if(sols.empty()) results.push_back(std::nullopt);
    else results.push_back(sols.front());
  }
  return results;
}

std::vector<JointState> IKSolver::solve_multiple(const Pose& goal_pose, int max_solutions, int max_seeds, const std::optional<JointState>& first_seed) {
  std::vector<JointState> out;
  for(int s = 0; s < max_seeds && static_cast<int>(out.size()) < max_solutions; s++) {
    std::optional<JointState> cand = solve_from_seed(goal_pose, (s == 0 && first_seed) ? first_seed->position : random_joint_values());
    if(!cand) continue;
    if(robot_world_ && !robot_world_->is_collision_free(*cand)) continue;
    bool dup = false;
    for(const auto& o : out) {
      float m = 0.0f;
      for(size_t j = 0; j < o.position.size(); j++) m = std::max(m, std::abs(o.position[j] - cand->position[j]));
      if(m < 0.05f) dup = true;
    }
    if(!dup) out.push_back(*cand);
  }
  return out;
}

}  // namespace vkmotion
