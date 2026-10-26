#include "motion_gen.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <future>
#include <random>

#include "fast_planner.hpp"
#include "vk_rrt.hpp"
#include "fast_collision.hpp"
#include "ik_solver.hpp"
#include "robot_world.hpp"
#include "rrt_connect.hpp"

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

// Constraint違反を数値勾配降下で下げる。ponytail: 制約チェック自体が軽量なCPU計算なのでGPU化しない。
bool repair_constraint_violation(const ConstraintSet& constraints, const RobotConfig& robot, std::vector<float>& q, int max_iterations) {
  constexpr float kDelta = 1e-2f;
  constexpr float kStep  = 0.05f;
  const auto& lower = robot.joint_position_limits_lower();
  const auto& upper = robot.joint_position_limits_upper();
  const size_t dof  = q.size();

  if(constraints.total_violation(JointState{q, {}}, robot) <= 0.0f) return true;
  for(int iter = 0; iter < max_iterations; iter++) {
    std::vector<float> grad(dof, 0.0f);
    float norm = 0.0f;
    for(size_t j = 0; j < dof; j++) {
      std::vector<float> qp = q, qm = q;
      qp[j] += kDelta;
      qm[j] -= kDelta;
      float vp = constraints.total_violation(JointState{qp, {}}, robot);
      float vm = constraints.total_violation(JointState{qm, {}}, robot);
      grad[j]  = (vp - vm) / (2.0f * kDelta);
      norm += grad[j] * grad[j];
    }
    norm = std::sqrt(norm);
    if(norm < 1e-8f) break;
    for(size_t j = 0; j < dof; j++) {
      q[j] -= kStep * grad[j] / norm;  // 違反量を下る方向
      q[j] = std::max(lower[j], std::min(upper[j], q[j]));
    }
    if(constraints.total_violation(JointState{q, {}}, robot) <= 0.0f) return true;
  }
  return false;
}

// shortcut平滑化+制約repair+制約repair後の再衝突チェックをまとめる(直線補間経路・RRT-Connect経路どちらの後処理にも使う)。
bool finalize_trajectory(const RobotWorld& rw, const RobotConfig& robot, std::vector<JointState>& traj, const std::optional<ConstraintSet>& constraints) {
  shortcut_smooth(rw, traj, kShortcutIterations);
  if(!constraints.has_value() || constraints->empty()) return true;

  bool ok = true;
  for(size_t i = 1; i + 1 < traj.size() && ok; i++) {
    if(!constraints->is_satisfied(traj[i], robot)) ok = repair_constraint_violation(*constraints, robot, traj[i].position, kRepairMaxIterations);
  }
  if(!ok) return false;

  std::vector<JointState> interior(traj.begin() + 1, traj.end() - 1);
  std::vector<bool> still_free = rw.is_collision_free_batch(interior);
  return std::all_of(still_free.begin(), still_free.end(), [](bool b) { return b; });
}

// 折れ線(関節空間)上の時間パラメタライズ: 各関節の速度制限と加速度制限を守る。
//  - 経路長sに沿った速度 sdot を、区間ごとの上限(速度制限/加速度制限 / |単位方向|)と、角での関節速度ジャンプ上限で抑える
//  - 前進パス・後退パスで区間ごとに一定加速度で到達できる速度へ丸める(両端は静止)
// ponytail: jerk制限は未対応(加速度は区間ごとに不連続)。必要になればS字プロファイルへ拡張。
std::vector<float> time_parameterize(const std::vector<JointState>& traj, const std::vector<float>& velocity_limits, float max_acceleration) {
  const size_t n = traj.size();
  std::vector<float> times(n, 0.0f);
  if(n < 2) return times;
  const size_t dof = traj[0].position.size();
  auto vmax = [&](size_t j) { return (j < velocity_limits.size() && velocity_limits[j] > 1e-6f) ? velocity_limits[j] : 1.0e6f; };
  const float amax_total = max_acceleration > 1e-6f ? max_acceleration : 1.0e6f;
  // 加速度の予算: 区間内の加速度 0.6 + 角での速度ジャンプ 0.4 (合算しても上限を超えない)
  const float amax        = 0.6f * amax_total;
  const float amax_corner = 0.4f * amax_total;

  std::vector<float> len(n - 1), vseg(n - 1), aseg(n - 1);
  std::vector<std::vector<float>> dir(n - 1, std::vector<float>(dof));
  for(size_t k = 0; k + 1 < n; k++) {
    float l2 = 0.0f;
    for(size_t j = 0; j < dof; j++) {
      float d = traj[k + 1].position[j] - traj[k].position[j];
      dir[k][j] = d;
      l2 += d * d;
    }
    len[k] = std::sqrt(l2);
    float v = 1.0e6f, a = 1.0e6f;
    if(len[k] > 1e-9f) {
      for(size_t j = 0; j < dof; j++) {
        dir[k][j] /= len[k];
        float ad = std::abs(dir[k][j]);
        if(ad > 1e-9f) {
          v = std::min(v, vmax(j) / ad);
          a = std::min(a, amax / ad);
        }
      }
    }
    vseg[k] = v;
    aseg[k] = a;
  }

  // 各点での速度上限。角では関節速度が不連続にジャンプするので、隣り合う短い方の区間を時間 L/sdot で通過するあいだに
  // 加速度の予算(0.4*amax)ぶんでしか速度が変えられないとみなし、 jump * sdot^2 / L <= 0.4*amax を満たす速度に抑える。
  std::vector<float> cap(n, 0.0f);
  for(size_t i = 1; i + 1 < n; i++) {
    float c = std::min(vseg[i - 1], vseg[i]);
    const float lmin = std::min(len[i - 1], len[i]);
    for(size_t j = 0; j < dof; j++) {
      float jump = std::abs(dir[i][j] - dir[i - 1][j]);
      if(jump > 1e-6f) c = std::min(c, std::sqrt(amax_corner * lmin / jump));
    }
    cap[i] = c;
  }
  std::vector<float> v2(n);  // sdot^2
  for(size_t i = 0; i < n; i++) v2[i] = cap[i] * cap[i];
  for(size_t i = 0; i + 1 < n; i++) v2[i + 1] = std::min(v2[i + 1], v2[i] + 2.0f * aseg[i] * len[i]);
  for(size_t i = n - 1; i > 0; i--) v2[i - 1] = std::min(v2[i - 1], v2[i] + 2.0f * aseg[i - 1] * len[i - 1]);

  for(size_t k = 0; k + 1 < n; k++) {
    float v0 = std::sqrt(v2[k]), v1 = std::sqrt(v2[k + 1]);
    float dt;
    if(v0 + v1 > 1e-4f) {
      dt = 2.0f * len[k] / (v0 + v1);
    } else {
      dt = 2.0f * std::sqrt(len[k] / aseg[k]);  // 両端が静止する単一区間: 三角速度プロファイル
    }
    times[k + 1] = times[k] + std::max(dt, kMinDt);
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
    robot_world_                  = std::make_unique<RobotWorld>(RobotWorldConfig{config_.robot_config, world, config_.use_vulkan, config_.acm});
  }
  return *robot_world_;
}

void MotionGen::warmup() {
  RobotWorld& rw = robot_world();
  rw.warmup();  // コンパイル済み衝突判定・Vulkanパイプラインの初回構築をここで済ませる
}

MotionGen::MotionGen(const MotionGenConfig& config) : config_(config) {}
MotionGen::~MotionGen() = default;

IKSolver& MotionGen::ik_solver() {
  if(!ik_) {
    IKSolverConfig ik_cfg;
    ik_cfg.robot_config = config_.robot_config;
    ik_cfg.world_config = config_.world_config;
    ik_cfg.num_seeds    = std::max(1, config_.num_trajopt_seeds);
    ik_cfg.acm          = config_.acm;
    ik_cfg.seed         = config_.seed ? config_.seed * 7919 + 1 : 0;
    ik_                 = std::make_unique<IKSolver>(ik_cfg);
  }
  return *ik_;
}

MotionGenResult MotionGen::plan_single(const JointState& start_state, const Pose& goal_pose, const MotionGenPlanConfig& plan_config) {
  IKSolver& ik = ik_solver();
  // 衝突なしIK解を複数集め、全部をゴール木の根にして1回の探索で解く(どの解に着くかはRRTが決める)。
  std::vector<JointState> goals = ik.solve_multiple(goal_pose, std::max(1, config_.num_ik_goals), std::max(config_.num_trajopt_seeds, 1) * 48, start_state);
  if(goals.empty()) {
    MotionGenResult result;
    result.status = "IK failed to find a collision-free goal configuration";
    return result;
  }
  RobotWorld& rw = robot_world();
  if(const FastCollision* fc = rw.fast_collision()) {
    if(!rw.is_collision_free(start_state)) {
      MotionGenResult result;
      result.status = "start or goal state is in collision";
      return result;
    }
    return plan_fast(*fc, start_state, goals, plan_config);
  }
  return plan_single_js(start_state, goals.front(), plan_config);
}

namespace {

// 区間ごとに最大関節ステップがkCollisionCheckResolution以下になるよう等分割(直線区間なので経路長は変わらない)。
std::vector<JointState> densify(const ConfigPath& path) {
  std::vector<JointState> out;
  if(path.empty()) return out;
  out.push_back(JointState{path.front(), {}});
  for(size_t i = 1; i < path.size(); i++) {
    float max_delta = 0.0f;
    for(size_t j = 0; j < path[i].size(); j++) max_delta = std::max(max_delta, std::abs(path[i][j] - path[i - 1][j]));
    int n = std::max(1, static_cast<int>(std::ceil(max_delta / kCollisionCheckResolution)));
    for(int k = 1; k <= n; k++) {
      float t = static_cast<float>(k) / static_cast<float>(n);
      JointState js;
      js.position.resize(path[i].size());
      for(size_t j = 0; j < path[i].size(); j++) js.position[j] = path[i - 1][j] + t * (path[i][j] - path[i - 1][j]);
      out.push_back(std::move(js));
    }
  }
  return out;
}

}  // namespace

MotionGenResult MotionGen::plan_fast(const FastCollision& fc, const JointState& start, const std::vector<JointState>& goals, const MotionGenPlanConfig& plan_config) {
  MotionGenResult result;
  const float margin = config_.collision_margin;
  const RobotConfig& robot = *config_.robot_config;
  const uint64_t seed      = (config_.seed ? config_.seed : std::random_device{}()) + 0x9E3779B9ull * (++plan_counter_);

  std::vector<std::vector<float>> goal_q;
  for(const auto& g : goals) goal_q.push_back(g.position);

  const auto t0 = std::chrono::steady_clock::now();
  auto remaining = [&] { return std::max(0.01, plan_config.timeout_seconds - std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count()); };

  // まず直線(start->どれかのgoal)。通れば探索不要。
  ConfigPath direct;
  for(const auto& g : goal_q) {
    if(fc.motion_free(start.position.data(), g.data(), margin, false, config_.collision_step)) {
      direct = {start.position, g};
      break;
    }
  }
  if(direct.empty() && !plan_config.enable_graph_planner) {
    result.status = "failed to find a collision-free trajectory after max_attempts";
    return result;
  }

  const float fine_step   = config_.collision_step;
  const float coarse_step = std::max(config_.search_step, fine_step);
  const bool via_search   = direct.empty();
  const int threads       = std::max(1, config_.num_threads);

  // 大域探索(step: 辺の検査間隔)。見つからなければ空の経路を返す。
  float used_margin = margin;  // 見つかった経路が満たしている余裕(端点がすれすれの問題では縮める)
  auto search = [&](float step) -> ConfigPath {
    FastRrtResult rrt;
    used_margin = margin;
    if(config_.gpu_search && robot_world().gpu_rrt_available()) {
      if(config_.gpu_handoff_s > 0.0f) {
        FastRrtSettings quick;
        quick.margin      = margin;
        quick.motion_step = step;
        quick.timeout_s   = config_.gpu_handoff_s;
        quick.seed        = seed ^ 0x5A5A5A5Aull;
        FastRrtResult qr  = fast_rrt_connect(fc, robot.joint_position_limits_lower(), robot.joint_position_limits_upper(), start.position, goal_q, quick);
        if(qr.success) {
          used_margin = qr.margin;
          return std::move(qr.path);
        }
      }
      GpuRrtSettings gs;
      gs.motion_step = step;
      gs.max_submits = 6;
      gs.margin = margin;  // 端点がすれすれで木が伸びないときはGPU側が余裕を縮める
      for(int attempt = 0; attempt < 12 && remaining() > 0.02; attempt++) {
        gs.seed = static_cast<uint32_t>(seed + 7919ull * static_cast<uint64_t>(attempt));
        ConfigPath gp;
        float m  = margin;
        auto tg  = std::chrono::steady_clock::now();
        bool gok = robot_world().gpu_rrt_connect(start.position, goal_q, gs, gp, &m);
        if(gok) {
          used_margin = m;
          return gp;
        }
      }
    }
    if(threads > 1) {
      // 大半の問題は数msで解けるので、スレッド起動のオーバーヘッドを払う前に1スレッドで短く試す。
      FastRrtSettings quick;
      quick.margin      = margin;
      quick.motion_step = step;
      quick.timeout_s   = 0.0005;
      quick.seed        = seed ^ 0xA5A5A5A5ull;
      rrt               = fast_rrt_connect(fc, robot.joint_position_limits_lower(), robot.joint_position_limits_upper(), start.position, goal_q, quick);
      if(rrt.success) used_margin = rrt.margin;
    }
    // RRTは実行時間の分布が重い裾を持つ(数倍〜10倍遅い"運の悪い"シードがある)ので、短い制限時間で打ち切って別シードで再開する。
    // 制限時間は20msから試行ごとに2倍にして、本当に難しい問題でも総時間が(再開の無駄を含めても)2倍程度で済むようにする(以前の3msからの1.5倍では、10〜20msかかる問題が途中で何度も打ち切られていた)。
    // 余裕(margin)は、木が伸びないときだけ探索の途中で縮める(fast_rrt_connect)。
    constexpr int kMaxRestarts = 64;
    for(int attempt = 0; attempt < kMaxRestarts && !rrt.success; attempt++) {
      FastRrtSettings st;
      st.margin      = margin;
      st.motion_step = step;
      st.timeout_s   = std::min(remaining(), std::min(1.0, 0.02 * std::pow(2.0, attempt)));
      if(threads == 1) {
        st.seed = seed + static_cast<uint64_t>(attempt) * 1000003ull;
        rrt     = fast_rrt_connect(fc, robot.joint_position_limits_lower(), robot.joint_position_limits_upper(), start.position, goal_q, st);
      } else {
        std::atomic<bool> cancel{false};
        std::vector<std::future<FastRrtResult>> futs;
        for(int t = 0; t < threads; t++) {
          FastRrtSettings s = st;
          s.seed            = seed + static_cast<uint64_t>(attempt) * 1000003ull + static_cast<uint64_t>(t) * 7777ull;
          s.cancel          = &cancel;
          futs.push_back(std::async(std::launch::async, [&, s] {
            FastRrtResult r = fast_rrt_connect(fc, robot.joint_position_limits_lower(), robot.joint_position_limits_upper(), start.position, goal_q, s);
            if(r.success) cancel.store(true);
            return r;
          }));
        }
        for(auto& f : futs) {
          FastRrtResult r = f.get();
          if(r.success && (!rrt.success || r.path.size() < rrt.path.size())) rrt = std::move(r);
        }
      }
      if(rrt.success) used_margin = rrt.margin;
      if(remaining() <= 0.011) break;
    }
    return rrt.success ? std::move(rrt.path) : ConfigPath{};
  };

  // 経路の全辺・内部点を細かい間隔で検証する(探索を粗い間隔で行った場合の安全網)
  auto verify = [&](const ConfigPath& p, float m) {
    for(size_t i = 0; i + 1 < p.size(); i++) {
      if(i + 1 < p.size() - 1 && !fc.is_free_margin(p[i + 1].data(), m)) return false;
      if(!fc.motion_free(p[i].data(), p[i + 1].data(), m, false, fine_step)) return false;
    }
    return true;
  };

  ConfigPath path;
  if(!via_search) {
    path = std::move(direct);
  } else {
    auto step_of_pass = [&](int pass) { return pass == 0 ? coarse_step : fine_step; };
    for(int pass = 0; pass < 2 && path.empty(); pass++) {
      if(pass == 1 && coarse_step <= fine_step) break;
      ConfigPath cand = search(step_of_pass(pass));
      if(cand.empty()) break;
      const float m = used_margin;  // 縮めた余裕で見つかった経路は、その余裕で短縮・検証する
      fast_reduce(fc, cand, m, fine_step);
      fast_shortcut(fc, cand, plan_config.shortcut_iterations, seed, m, fine_step);
      // 粗い間隔で探索した経路だけ、細かい間隔で検証し直す(余裕0。余裕込みで検証すると、短縮で辺が分割されて検査点がずれた結果、余裕ぎりぎりの辺が不合格になりうる)。
      // 細かい間隔で探索した経路は探索時の検査をそのまま信頼する。
      const bool trusted = step_of_pass(pass) <= fine_step;
      if(trusted || verify(cand, 0.0f)) {
        path = std::move(cand);
      }
    }
    if(path.empty()) {
      result.status = "failed to find a collision-free trajectory after max_attempts";
      return result;
    }
  }
  {
    // shortcutが挿入する点が既存の頂点とほぼ重なることがある。ほぼ同一の点は時間パラメタライズが不安定になるので潰す。
    ConfigPath cleaned{path.front()};
    for(size_t i = 1; i < path.size(); i++) {
      float d2 = 0.0f;
      for(size_t j = 0; j < path[i].size(); j++) d2 += (path[i][j] - cleaned.back()[j]) * (path[i][j] - cleaned.back()[j]);
      if(d2 > 1e-6f || i + 1 == path.size()) cleaned.push_back(path[i]);
    }
    path = std::move(cleaned);
  }

  std::vector<JointState> traj = densify(path);
  if(config_.constraints.has_value() && !config_.constraints->empty()) {
    if(!finalize_trajectory(robot_world(), robot, traj, config_.constraints)) {
      result.status = "failed to satisfy constraints";
      return result;
    }
  }
  result.success            = true;
  result.trajectory         = std::move(traj);
  result.interpolated_times = time_parameterize(result.trajectory, robot.joint_velocity_limits(), config_.max_acceleration);
  result.status             = via_search ? "success (RRT-Connect fallback)" : "success";
  return result;
}

MotionGenResult MotionGen::plan_single_js(const JointState& start_state, const JointState& goal_state, const MotionGenPlanConfig& plan_config) {
  MotionGenResult result;

  RobotWorld& rw            = robot_world();
  const RobotConfig& robot = *config_.robot_config;

  if(!rw.is_collision_free(start_state) || !rw.is_collision_free(goal_state)) {
    result.status = "start or goal state is in collision";
    return result;
  }
  if(const FastCollision* fc = rw.fast_collision()) return plan_fast(*fc, start_state, {goal_state}, plan_config);

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

    if(all_ok && finalize_trajectory(rw, robot, traj, config_.constraints)) {
      result.success            = true;
      result.trajectory         = std::move(traj);
      result.interpolated_times = time_parameterize(result.trajectory, robot.joint_velocity_limits(), config_.max_acceleration);
      result.status             = "success";
      return result;
    }
  }

  // 直線補間+局所repairが全ラウンド失敗した場合の大域探索フォールバック(cuRoboの並列幾何プランナー相当)。
  if(plan_config.enable_graph_planner) {
    cutil::Ref<WorldConfig> world = config_.world_config ? config_.world_config : cutil::make_ref<WorldConfig>();
    RrtConnectPlanner rrt(config_.robot_config, world, config_.use_vulkan, config_.acm);
    RrtConnectResult rrt_result = rrt.plan(start_state, goal_state);
    if(rrt_result.success && finalize_trajectory(rw, robot, rrt_result.trajectory, config_.constraints)) {
      result.success            = true;
      result.trajectory         = std::move(rrt_result.trajectory);
      result.interpolated_times = time_parameterize(result.trajectory, robot.joint_velocity_limits(), config_.max_acceleration);
      result.status             = "success (RRT-Connect fallback)";
      return result;
    }
  }

  result.status = "failed to find a collision-free trajectory after max_attempts";
  return result;
}

}  // namespace vkmotion
