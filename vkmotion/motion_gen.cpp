#include "motion_gen.hpp"

#include <algorithm>
#include <cmath>
#include <random>

#include "ik_solver.hpp"
#include "robot_world.hpp"

namespace vkmotion {

namespace {

constexpr int kNumWaypoints         = 21;
constexpr float kRepairDelta        = 1e-2f;
constexpr float kRepairStep         = 0.05f;
constexpr int kRepairMaxIterations  = 80;
constexpr int kShortcutIterations   = 60;
constexpr float kMinDt              = 1e-3f;

std::mt19937& rng() {
  static thread_local std::mt19937 engine(std::random_device{}());
  return engine;
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

float joint_delta_norm(const JointState& a, const JointState& b) {
  float d = 0.0f;
  for(size_t j = 0; j < a.position.size(); j++) {
    float diff = a.position[j] - b.position[j];
    d += diff * diff;
  }
  return std::sqrt(d);
}

float trajectory_length(const std::vector<JointState>& traj) {
  float len = 0.0f;
  for(size_t i = 1; i < traj.size(); i++) len += joint_delta_norm(traj[i], traj[i - 1]);
  return len;
}

// start-goal直線補間上のランダムなtに、perturbation_scale(0=直線通り、1=関節レンジ全体)に応じた摂動を加える経由点。
JointState biased_via_point(const JointState& start, const JointState& goal, const RobotConfig& robot, float perturbation_scale) {
  const auto& lower = robot.joint_position_limits_lower();
  const auto& upper = robot.joint_position_limits_upper();
  std::uniform_real_distribution<float> t_dist(0.3f, 0.7f);
  float t = t_dist(rng());

  JointState via;
  via.position.resize(start.position.size());
  for(size_t i = 0; i < via.position.size(); i++) {
    float base  = start.position[i] + t * (goal.position[i] - start.position[i]);
    float range = upper[i] - lower[i];
    std::uniform_real_distribution<float> pert(-0.5f * perturbation_scale * range, 0.5f * perturbation_scale * range);
    via.position[i] = std::max(lower[i], std::min(upper[i], base + pert(rng())));
  }
  return via;
}

std::vector<JointState> via_point_trajectory(const JointState& start, const JointState& goal, const JointState& via) {
  int half = kNumWaypoints / 2;
  auto traj = linear_interpolate(start, via, half);
  auto rest = linear_interpolate(via, goal, kNumWaypoints - half + 1);
  traj.insert(traj.end(), rest.begin() + 1, rest.end());
  return traj;
}

// 数値勾配でworld衝突距離を登る局所repair。各反復の2*dof個の摂動状態を1回のdispatchにまとめて評価する(Vulkanでは状態ごとの個別dispatchだと投入コストが支配的になるため)。勾配ゼロ/反復上限は失敗として呼び出し元に再試行させる。
bool repair_waypoint(const RobotWorld& rw, const RobotConfig& robot, std::vector<float>& q) {
  std::vector<JointState> single{JointState{q, {}}};
  if(rw.get_min_world_collision_distance_batch(single)[0] > 0.0f) return true;

  const auto& lower = robot.joint_position_limits_lower();
  const auto& upper = robot.joint_position_limits_upper();
  const size_t dof  = q.size();

  for(int iter = 0; iter < kRepairMaxIterations; iter++) {
    std::vector<JointState> perturbed;
    perturbed.reserve(dof * 2);
    for(size_t i = 0; i < dof; i++) {
      std::vector<float> qp = q, qm = q;
      qp[i] += kRepairDelta;
      qm[i] -= kRepairDelta;
      perturbed.push_back(JointState{std::move(qp), {}});
      perturbed.push_back(JointState{std::move(qm), {}});
    }
    std::vector<float> dists = rw.get_min_world_collision_distance_batch(perturbed);

    std::vector<float> grad(dof, 0.0f);
    float norm = 0.0f;
    for(size_t i = 0; i < dof; i++) {
      grad[i] = (dists[2 * i] - dists[2 * i + 1]) / (2.0f * kRepairDelta);
      norm += grad[i] * grad[i];
    }
    norm = std::sqrt(norm);
    if(norm < 1e-8f) break;
    for(size_t i = 0; i < dof; i++) {
      q[i] += kRepairStep * grad[i] / norm;
      q[i] = std::max(lower[i], std::min(upper[i], q[i]));
    }
    single[0] = JointState{q, {}};
    if(rw.get_min_world_collision_distance_batch(single)[0] > 0.0f) return true;
  }
  return false;
}

// ランダムな2点[i,j]を直線で結び直し、衝突フリーかつ短くなるなら置き換える標準的なshortcut smoothing。
void shortcut_smooth(const RobotWorld& rw, std::vector<JointState>& traj, int iterations) {
  if(traj.size() < 4) return;
  for(int iter = 0; iter < iterations; iter++) {
    std::uniform_int_distribution<int> i_dist(0, static_cast<int>(traj.size()) - 3);
    int i = i_dist(rng());
    std::uniform_int_distribution<int> j_dist(i + 2, static_cast<int>(traj.size()) - 1);
    int j = j_dist(rng());

    std::vector<JointState> candidate = linear_interpolate(traj[i], traj[j], j - i + 1);
    std::vector<JointState> interior(candidate.begin() + 1, candidate.end() - 1);
    std::vector<bool> free            = rw.is_collision_free_batch(interior);
    if(!std::all_of(free.begin(), free.end(), [](bool b) { return b; })) continue;

    float old_len = 0.0f;
    for(int k = i + 1; k <= j; k++) old_len += joint_delta_norm(traj[k], traj[k - 1]);
    float new_len = 0.0f;
    for(size_t k = 1; k < candidate.size(); k++) new_len += joint_delta_norm(candidate[k], candidate[k - 1]);
    if(new_len >= old_len - 1e-6f) continue;

    traj.erase(traj.begin() + i + 1, traj.begin() + j);
    traj.insert(traj.begin() + i + 1, candidate.begin() + 1, candidate.end() - 1);
  }
}

// 各関節の速度制限のみを守る素朴な時間パラメタライズ(加速度/jerkは考慮しない)。ponytail: 必要になったら台形/S字速度プロファイルへ拡張。
std::vector<float> time_parameterize(const std::vector<JointState>& traj, const std::vector<float>& velocity_limits) {
  std::vector<float> times(traj.size(), 0.0f);
  for(size_t i = 1; i < traj.size(); i++) {
    float dt = 0.0f;
    for(size_t j = 0; j < traj[i].position.size(); j++) {
      float v_limit = (j < velocity_limits.size() && velocity_limits[j] > 1e-6f) ? velocity_limits[j] : 1.0e6f;
      dt            = std::max(dt, std::abs(traj[i].position[j] - traj[i - 1].position[j]) / v_limit);
    }
    times[i] = times[i - 1] + std::max(dt, kMinDt);
  }
  return times;
}

}  // namespace

MotionGenConfig MotionGenConfig::load_from_robot_config(cutil::Ref<RobotConfig> robot_config, cutil::Ref<WorldConfig> world_config) {
  MotionGenConfig cfg;
  cfg.robot_config = robot_config;
  cfg.world_config = world_config;
  return cfg;
}

RobotWorld& MotionGen::robot_world() {
  if(!robot_world_) {
    cutil::Ref<WorldConfig> world = config_.world_config ? config_.world_config : cutil::make_ref<WorldConfig>();
    robot_world_                  = std::make_unique<RobotWorld>(RobotWorldConfig{config_.robot_config, world, config_.use_vulkan});
  }
  return *robot_world_;
}

void MotionGen::warmup() {
  RobotWorld& rw = robot_world();
  JointState dummy{std::vector<float>(config_.robot_config->dof(), 0.0f), {}};
  rw.is_collision_free(dummy);  // Vulkanパイプラインの初回dispatchをここで強制する
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

  RobotWorld& rw            = robot_world();
  const RobotConfig& robot = *config_.robot_config;

  if(!rw.is_collision_free(start_state) || !rw.is_collision_free(goal_state)) {
    result.status = "start or goal state is in collision";
    return result;
  }

  const int num_seeds        = std::max(1, config_.num_trajopt_seeds);
  const int max_rounds       = std::max(1, plan_config.max_attempts);
  const int interior_count   = kNumWaypoints - 2;

  for(int round = 0; round < max_rounds; round++) {
    float perturbation_scale = max_rounds > 1 ? static_cast<float>(round) / static_cast<float>(max_rounds - 1) : 0.0f;

    // K本の候補軌道の中間waypointを1回のバッチdispatchで評価する(逐次評価だとVulkanが並列性を活かせないため)。
    std::vector<std::vector<JointState>> seeds;
    seeds.reserve(num_seeds);
    seeds.push_back(linear_interpolate(start_state, goal_state, kNumWaypoints));
    for(int s = 1; s < num_seeds; s++) {
      JointState via = biased_via_point(start_state, goal_state, robot, perturbation_scale);
      seeds.push_back(via_point_trajectory(start_state, goal_state, via));
    }

    std::vector<JointState> all_interior;
    all_interior.reserve(seeds.size() * interior_count);
    for(const auto& traj : seeds)
      for(size_t i = 1; i + 1 < traj.size(); i++) all_interior.push_back(traj[i]);
    std::vector<bool> free = rw.is_collision_free_batch(all_interior);

    std::vector<int> collisions(seeds.size(), 0);
    for(size_t s = 0; s < seeds.size(); s++)
      for(int k = 0; k < interior_count; k++)
        if(!free[s * interior_count + k]) collisions[s]++;

    int best = 0;
    for(size_t s = 1; s < seeds.size(); s++) {
      bool better = collisions[s] < collisions[best] ||
                    (collisions[s] == collisions[best] && collisions[s] == 0 && trajectory_length(seeds[s]) < trajectory_length(seeds[best]));
      if(better) best = static_cast<int>(s);
    }

    std::vector<JointState>& traj = seeds[best];
    bool all_ok                    = true;
    for(size_t i = 1; i + 1 < traj.size(); i++) {
      if(!free[best * interior_count + (i - 1)] && !repair_waypoint(rw, robot, traj[i].position)) {
        all_ok = false;
        break;
      }
    }

    if(all_ok) {
      shortcut_smooth(rw, traj, kShortcutIterations);
      result.success            = true;
      result.trajectory         = std::move(traj);
      result.interpolated_times = time_parameterize(result.trajectory, robot.joint_velocity_limits());
      result.status             = "success";
      return result;
    }
  }

  result.status = "failed to find a collision-free trajectory after max_attempts";
  return result;
}

}  // namespace vkmotion
