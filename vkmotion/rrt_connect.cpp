#include "rrt_connect.hpp"

#include <algorithm>
#include <cmath>
#include <random>

#include "fast_planner.hpp"
#include "robot_world.hpp"

namespace vkmotion {

namespace {

struct Node {
  std::vector<float> q;
  int parent;  // -1 for root
};

float dist(const std::vector<float>& a, const std::vector<float>& b) {
  float d = 0.0f;
  for(size_t i = 0; i < a.size(); i++) {
    float diff = a[i] - b[i];
    d += diff * diff;
  }
  return std::sqrt(d);
}

std::vector<float> steer(const std::vector<float>& from, const std::vector<float>& to, float step) {
  float d = dist(from, to);
  if(d <= step) return to;
  std::vector<float> out(from.size());
  for(size_t i = 0; i < from.size(); i++) out[i] = from[i] + (to[i] - from[i]) * (step / d);
  return out;
}

int nearest(const std::vector<Node>& tree, const std::vector<float>& q) {
  int best    = 0;
  float best_d = dist(tree[0].q, q);
  for(size_t i = 1; i < tree.size(); i++) {
    float d = dist(tree[i].q, q);
    if(d < best_d) {
      best_d = d;
      best   = static_cast<int>(i);
    }
  }
  return best;
}

std::vector<float> random_q(std::mt19937& rng, const RobotConfig& robot) {
  const auto& lo = robot.joint_position_limits_lower();
  const auto& hi = robot.joint_position_limits_upper();
  std::vector<float> q(lo.size());
  for(size_t i = 0; i < q.size(); i++) {
    std::uniform_real_distribution<float> d(lo[i], hi[i]);
    q[i] = d(rng);
  }
  return q;
}

std::vector<JointState> extract_path(const std::vector<Node>& tree, int idx) {
  std::vector<JointState> path;
  while(idx >= 0) {
    path.push_back(JointState{tree[idx].q, {}});
    idx = tree[idx].parent;
  }
  std::reverse(path.begin(), path.end());
  return path;
}

// ランダムサンプルmax_samples_per_batch個を1歩steerした候補にし、1回のバッチ衝突チェックで最初に衝突フリーだったものをtree_aに追加する。
bool extend_batch(std::vector<Node>& tree_a, const RobotConfig& robot, const RobotWorld& rw, std::mt19937& rng, int batch_size, float step_size) {
  std::vector<int> nearest_idx(static_cast<size_t>(batch_size));
  std::vector<std::vector<float>> candidates(static_cast<size_t>(batch_size));
  std::vector<JointState> states;
  states.reserve(static_cast<size_t>(batch_size));
  for(int s = 0; s < batch_size; s++) {
    std::vector<float> sample = random_q(rng, robot);
    int ni                    = nearest(tree_a, sample);
    nearest_idx[static_cast<size_t>(s)] = ni;
    candidates[static_cast<size_t>(s)]  = steer(tree_a[static_cast<size_t>(ni)].q, sample, step_size);
    states.push_back(JointState{candidates[static_cast<size_t>(s)], {}});
  }
  std::vector<bool> free = rw.is_collision_free_batch(states);
  for(size_t s = 0; s < free.size(); s++) {
    if(free[s]) {
      tree_a.push_back({candidates[s], nearest_idx[s]});
      return true;
    }
  }
  return false;
}

// tree_bをtargetに向けて1歩ずつ伸ばす(衝突したら打ち切り)。targetに到達できたそのノードindexを返す、無理なら-1。
int try_connect(std::vector<Node>& tree_b, const std::vector<float>& target, const RobotWorld& rw, float step_size) {
  int cur = nearest(tree_b, target);
  for(int step = 0; step < 1000; step++) {
    float d = dist(tree_b[static_cast<size_t>(cur)].q, target);
    std::vector<float> next_q = d <= step_size ? target : steer(tree_b[static_cast<size_t>(cur)].q, target, step_size);
    if(!rw.is_collision_free(JointState{next_q, {}})) return -1;
    tree_b.push_back({next_q, cur});
    cur = static_cast<int>(tree_b.size()) - 1;
    if(d <= step_size) return cur;
  }
  return -1;
}

}  // namespace

RrtConnectPlanner::RrtConnectPlanner(cutil::Ref<RobotConfig> robot_config, cutil::Ref<WorldConfig> world_config, bool use_vulkan, AllowedCollisionMatrix acm)
    : robot_config_(robot_config), world_config_(world_config), use_vulkan_(use_vulkan), acm_(std::move(acm)) {}

RrtConnectResult RrtConnectPlanner::plan(const JointState& start, const JointState& goal, const RrtConnectConfig& config) {
  RrtConnectResult result;
  RobotWorld rw(RobotWorldConfig{robot_config_, world_config_, use_vulkan_, acm_});
  const RobotConfig& robot = *robot_config_;

  if(!rw.is_collision_free(start) || !rw.is_collision_free(goal)) {
    result.status = "start or goal state is in collision";
    return result;
  }

  if(const FastCollision* fc = rw.fast_collision()) {
    FastRrtSettings st;
    st.seed = std::random_device{}();
    FastRrtResult r = fast_rrt_connect(*fc, robot.joint_position_limits_lower(), robot.joint_position_limits_upper(), start.position, {goal.position}, st);
    if(r.success) {
      result.success = true;
      for(auto& q : r.path) result.trajectory.push_back(JointState{std::move(q), {}});
      result.status = "success";
    } else {
      result.status = "RRT-Connect failed to find a path within max_iterations";
    }
    return result;
  }

  std::mt19937 rng(std::random_device{}());
  std::vector<Node> tree_start{{start.position, -1}};
  std::vector<Node> tree_goal{{goal.position, -1}};
  std::vector<Node>* tree_a = &tree_start;
  std::vector<Node>* tree_b = &tree_goal;
  bool a_is_start            = true;

  for(int iter = 0; iter < config.max_iterations; iter++) {
    if(!extend_batch(*tree_a, robot, rw, rng, config.max_samples_per_batch, config.step_size)) {
      std::swap(tree_a, tree_b);
      a_is_start = !a_is_start;
      continue;
    }
    int a_new_idx                   = static_cast<int>(tree_a->size()) - 1;
    const std::vector<float>& q_new = (*tree_a)[static_cast<size_t>(a_new_idx)].q;

    int connect_idx = try_connect(*tree_b, q_new, rw, config.step_size);
    if(connect_idx >= 0) {
      std::vector<JointState> path_a = extract_path(*tree_a, a_new_idx);
      std::vector<JointState> path_b = extract_path(*tree_b, connect_idx);
      std::vector<JointState> full;
      if(a_is_start) {
        full = std::move(path_a);
        std::reverse(path_b.begin(), path_b.end());
        full.insert(full.end(), path_b.begin() + 1, path_b.end());
      } else {
        full = std::move(path_b);
        std::reverse(path_a.begin(), path_a.end());
        full.insert(full.end(), path_a.begin() + 1, path_a.end());
      }
      result.success    = true;
      result.trajectory = std::move(full);
      result.status     = "success";
      return result;
    }

    std::swap(tree_a, tree_b);
    a_is_start = !a_is_start;
  }

  result.status = "RRT-Connect failed to find a path within max_iterations";
  return result;
}

}  // namespace vkmotion
