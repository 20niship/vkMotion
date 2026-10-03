#include "motion_gen.hpp"

#include <algorithm>
#include <cmath>
#include <random>

#include "ik_solver.hpp"
#include "robot_world.hpp"

namespace vkmotion {

namespace {

constexpr float kRepairDelta             = 1e-2f;
constexpr float kRepairStep              = 0.05f;
constexpr int kRepairMaxIterations       = 40;
constexpr int kShortcutIterations        = 15;  // kCandidatesPerRound(5)個/roundなので総候補数は旧60回相当
constexpr float kMinDt                   = 1e-3f;
// OMPLのlongest_valid_segment_fraction相当。固定waypoint数だと長距離移動で間隔が粗くなり薄い障害物を見逃すため関節空間距離に応じて点数を決める。
constexpr float kCollisionCheckResolution = 0.08f;  // rad, 隣接waypoint間の最大関節角ステップ
constexpr int kMinWaypoints               = 11;
constexpr int kMaxWaypoints               = 400;

int adaptive_waypoint_count(const JointState& a, const JointState& b) {
  float max_delta = 0.0f;
  for(size_t j = 0; j < a.position.size(); j++) max_delta = std::max(max_delta, std::abs(a.position[j] - b.position[j]));
  int n = static_cast<int>(std::ceil(max_delta / kCollisionCheckResolution)) + 1;
  return std::max(kMinWaypoints, std::min(kMaxWaypoints, n));
}

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

std::vector<JointState> via_point_trajectory(const JointState& start, const JointState& goal, const JointState& via, int num_waypoints) {
  int half = num_waypoints / 2;
  auto traj = linear_interpolate(start, via, half);
  auto rest = linear_interpolate(via, goal, num_waypoints - half + 1);
  traj.insert(traj.end(), rest.begin() + 1, rest.end());
  return traj;
}

// ランダムな2点[i,j]を直線で結び直し、衝突フリーかつ短くなるなら置き換える標準的なshortcut smoothing。
// 候補を1個ずつdispatchすると往復コストが支配的になるため、ラウンドごとにkCandidatesPerRound個をまとめて1 dispatchで評価し最良のものだけ採用する。
constexpr int kCandidatesPerRound = 5;

void shortcut_smooth(const RobotWorld& rw, std::vector<JointState>& traj, int rounds) {
  if(traj.size() < 4) return;
  for(int round = 0; round < rounds; round++) {
    int n = static_cast<int>(traj.size());
    if(n < 4) break;

    struct Candidate {
      int i, j;
      std::vector<JointState> points;  // i..j inclusive, straight-line replacement
    };
    std::vector<Candidate> candidates;
    std::vector<JointState> all_interior;
    std::vector<size_t> interior_counts;

    for(int c = 0; c < kCandidatesPerRound; c++) {
      std::uniform_int_distribution<int> i_dist(0, n - 3);
      int i = i_dist(rng());
      std::uniform_int_distribution<int> j_dist(i + 2, n - 1);
      int j = j_dist(rng());
      std::vector<JointState> pts = linear_interpolate(traj[i], traj[j], j - i + 1);
      interior_counts.push_back(pts.size() - 2);
      all_interior.insert(all_interior.end(), pts.begin() + 1, pts.end() - 1);
      candidates.push_back({i, j, std::move(pts)});
    }

    std::vector<bool> free = rw.is_collision_free_batch(all_interior);

    int best            = -1;
    float best_reduction = 0.0f;
    size_t offset         = 0;
    for(size_t c = 0; c < candidates.size(); c++) {
      size_t cnt = interior_counts[c];
      bool all_free = std::all_of(free.begin() + offset, free.begin() + offset + cnt, [](bool b) { return b; });
      offset += cnt;
      if(!all_free) continue;

      const Candidate& cand = candidates[c];
      float old_len = 0.0f;
      for(int k = cand.i + 1; k <= cand.j; k++) old_len += joint_delta_norm(traj[k], traj[k - 1]);
      float new_len = 0.0f;
      for(size_t k = 1; k < cand.points.size(); k++) new_len += joint_delta_norm(cand.points[k], cand.points[k - 1]);
      float reduction = old_len - new_len;
      if(reduction > best_reduction) {
        best_reduction = reduction;
        best            = static_cast<int>(c);
      }
    }

    if(best < 0) continue;
    const Candidate& cand = candidates[static_cast<size_t>(best)];
    traj.erase(traj.begin() + cand.i + 1, traj.begin() + cand.j);
    traj.insert(traj.begin() + cand.i + 1, cand.points.begin() + 1, cand.points.end() - 1);
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
  const int num_waypoints    = adaptive_waypoint_count(start_state, goal_state);
  const int interior_count   = num_waypoints - 2;

  for(int round = 0; round < max_rounds; round++) {
    float perturbation_scale = max_rounds > 1 ? static_cast<float>(round) / static_cast<float>(max_rounds - 1) : 0.0f;

    // K本の候補軌道の中間waypointを1回のバッチdispatchで評価する(逐次評価だとVulkanが並列性を活かせないため)。
    std::vector<std::vector<JointState>> seeds;
    seeds.reserve(num_seeds);
    seeds.push_back(linear_interpolate(start_state, goal_state, num_waypoints));
    for(int s = 1; s < num_seeds; s++) {
      JointState via = biased_via_point(start_state, goal_state, robot, perturbation_scale);
      seeds.push_back(via_point_trajectory(start_state, goal_state, via, num_waypoints));
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
    std::vector<std::vector<float>*> failing;
    for(size_t i = 1; i + 1 < traj.size(); i++)
      if(!free[best * interior_count + (i - 1)]) failing.push_back(&traj[i].position);
    std::vector<bool> repaired = rw.repair_waypoints(failing, kRepairStep, kRepairDelta, kRepairMaxIterations);
    bool all_ok                = std::all_of(repaired.begin(), repaired.end(), [](bool b) { return b; });

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
