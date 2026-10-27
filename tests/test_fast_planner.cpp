#include <cmath>
#include <doctest/doctest.h>
#include <random>
#include <vkmotion/fast_collision.hpp>
#include <vkmotion/fast_planner.hpp>
#include <vkmotion/ik_solver.hpp>
#include <vkmotion/motion_gen.hpp>
#include <vkmotion/robot_world.hpp>

using namespace vkmotion;

namespace {

cutil::Ref<RobotConfig> ur5e() { return RobotConfig::from_file(std::string(VKMOTION_TEST_ASSET_DIR) + "/urdf/ur5e.urdf"); }

const std::vector<float> kStart{0.0f, -1.57f, 1.57f, -1.57f, -1.57f, 0.0f};
const std::vector<float> kGoal{1.5f, -1.0f, 1.0f, -1.0f, -1.0f, 0.5f};

// start-goalの関節空間直線補間の中間のEE位置に大きな球を置き、直線では通れなくする
cutil::Ref<WorldConfig> wall_world(const RobotConfig& robot) {
  auto world = cutil::make_ref<WorldConfig>();
  JointState mid;
  mid.position.resize(kStart.size());
  for(size_t j = 0; j < kStart.size(); j++) mid.position[j] = 0.5f * (kStart[j] + kGoal[j]);
  const auto m = robot.forward_kinematics(mid).back();
  Sphere blocker;
  blocker.name          = "blocker";
  blocker.pose.position = cutil::Vec3f(m(0, 3), m(1, 3), m(2, 3));
  blocker.radius        = 0.2f;
  world->add_obstacle(blocker);
  return world;
}

float path_length(const ConfigPath& p) {
  float len = 0.0f;
  for(size_t i = 1; i < p.size(); i++) {
    float d = 0.0f;
    for(size_t j = 0; j < p[i].size(); j++) d += (p[i][j] - p[i - 1][j]) * (p[i][j] - p[i - 1][j]);
    len += std::sqrt(d);
  }
  return len;
}

}  // namespace

TEST_CASE("fast_rrt_connect: solves around an obstacle and every edge is collision-free at fine resolution") {
  auto robot = ur5e();
  auto world = wall_world(*robot);
  FastCollision fc(*robot, AllowedCollisionMatrix{});
  REQUIRE(fc.set_world(*world));

  std::vector<float> start = kStart, goal = kGoal;
  REQUIRE(fc.is_free(start.data()));
  REQUIRE(fc.is_free(goal.data()));
  REQUIRE_FALSE(fc.motion_free(start.data(), goal.data()));  // 直線は壁を貫く

  FastRrtSettings st;
  st.seed          = 7;
  FastRrtResult r = fast_rrt_connect(fc, robot->joint_position_limits_lower(), robot->joint_position_limits_upper(), start, {goal}, st);
  REQUIRE(r.success);
  CHECK(r.path.front() == start);
  CHECK(r.path.back() == goal);

  // 実装とは独立に、0.005rad刻みの密なサンプルで参照実装(RobotWorld距離)による検証
  RobotWorld rw(RobotWorldConfig{robot, world, false});
  for(size_t i = 1; i < r.path.size(); i++) {
    float maxd = 0.0f;
    for(size_t j = 0; j < start.size(); j++) maxd = std::max(maxd, std::abs(r.path[i][j] - r.path[i - 1][j]));
    int n = std::max(1, static_cast<int>(std::ceil(maxd / 0.005f)));
    for(int k = 0; k <= n; k++) {
      JointState js;
      js.position.resize(start.size());
      for(size_t j = 0; j < start.size(); j++) js.position[j] = r.path[i - 1][j] + (r.path[i][j] - r.path[i - 1][j]) * (static_cast<float>(k) / static_cast<float>(n));
      bool free = true;
      for(float d : rw.get_collision_distance(js)) free = free && d > 0.0f;
      for(float d : rw.get_self_collision_distance(js)) free = free && d > 0.0f;
      REQUIRE(free);
    }
  }
}

TEST_CASE("fast_rrt_connect: multiple goals reach one of them, and respects the cancel flag") {
  auto robot = ur5e();
  auto world = cutil::make_ref<WorldConfig>();
  FastCollision fc(*robot, AllowedCollisionMatrix{});
  fc.set_world(*world);
  std::vector<float> start{0.0f, -1.2f, 1.0f, -1.0f, -1.0f, 0.0f};
  std::vector<std::vector<float>> goals{{0.5f, -1.0f, 1.0f, -1.0f, -1.0f, 0.2f}, {-0.5f, -1.4f, 1.2f, -1.0f, -1.0f, 0.1f}};
  FastRrtResult r = fast_rrt_connect(fc, robot->joint_position_limits_lower(), robot->joint_position_limits_upper(), start, goals, {});
  REQUIRE(r.success);
  CHECK((r.path.back() == goals[0] || r.path.back() == goals[1]));

  // 解けない問題(始点が壁に囲まれている)でもcancelで即座に戻る
  std::atomic<bool> cancel{true};
  FastRrtSettings st;
  st.cancel = &cancel;
  // 直線で結べない設定にするため壁を置く
  auto w2 = wall_world(*robot);
  FastCollision fc2(*robot, AllowedCollisionMatrix{});
  fc2.set_world(*w2);
  std::vector<float> a = kStart, b = kGoal;
  FastRrtResult c = fast_rrt_connect(fc2, robot->joint_position_limits_lower(), robot->joint_position_limits_upper(), a, {b}, st);
  CHECK_FALSE(c.success);
}

TEST_CASE("fast_shortcut / fast_reduce never lengthen a path and keep it collision-free") {
  auto robot = ur5e();
  auto world = wall_world(*robot);
  FastCollision fc(*robot, AllowedCollisionMatrix{});
  fc.set_world(*world);
  std::vector<float> start = kStart, goal = kGoal;
  FastRrtSettings st;
  st.seed          = 3;
  FastRrtResult r = fast_rrt_connect(fc, robot->joint_position_limits_lower(), robot->joint_position_limits_upper(), start, {goal}, st);
  REQUIRE(r.success);

  ConfigPath path = r.path;
  float before    = path_length(path);
  fast_reduce(fc, path);
  CHECK(path_length(path) <= before + 1e-4f);
  CHECK(path.size() <= r.path.size());
  float mid = path_length(path);
  fast_shortcut(fc, path, 100, 5);
  CHECK(path_length(path) <= mid + 1e-4f);
  CHECK(path.front() == start);
  CHECK(path.back() == goal);
  for(size_t i = 1; i < path.size(); i++) CHECK(fc.motion_free(path[i - 1].data(), path[i].data()));
}

TEST_CASE("IKSolver::solve_multiple: returns distinct converged solutions, all collision-free") {
  auto robot = ur5e();
  auto world = cutil::make_ref<WorldConfig>();
  Sphere blocker;
  blocker.name          = "blocker";
  blocker.pose.position = cutil::Vec3f(0.0f, 0.5f, 0.1f);
  blocker.radius        = 0.15f;
  world->add_obstacle(blocker);

  JointState known{{0.3f, -1.2f, 1.1f, -0.7f, 0.5f, 0.2f}, {}};
  auto fk       = robot->forward_kinematics(known).back();
  Pose goal;
  goal.position = cutil::Vec3f(fk(0, 3), fk(1, 3), fk(2, 3));
  cutil::Mat<3, 3, float> r3;
  for(unsigned i = 0; i < 3; i++)
    for(unsigned j = 0; j < 3; j++) r3(i, j) = fk(i, j);
  goal.quaternion = cutil::Quatf::from_mat3(r3);

  IKSolverConfig cfg;
  cfg.robot_config = robot;
  cfg.world_config = world;
  cfg.seed         = 42;
  IKSolver ik(cfg);
  auto sols = ik.solve_multiple(goal, 4, 400);
  REQUIRE(sols.size() >= 2);  // UR5eの手先姿勢には複数のIK解がある
  RobotWorld rw(RobotWorldConfig{robot, world, false});
  for(size_t a = 0; a < sols.size(); a++) {
    CHECK(rw.is_collision_free(sols[a]));
    auto m = robot->forward_kinematics(sols[a]).back();
    CHECK(std::sqrt(std::pow(m(0, 3) - goal.position[0], 2) + std::pow(m(1, 3) - goal.position[1], 2) + std::pow(m(2, 3) - goal.position[2], 2)) < 1e-3f);
    for(size_t b = a + 1; b < sols.size(); b++) {
      float maxd = 0.0f;
      for(size_t j = 0; j < 6; j++) maxd = std::max(maxd, std::abs(sols[a].position[j] - sols[b].position[j]));
      CHECK(maxd >= 0.05f);
    }
  }
  // seedを固定すれば再現する
  IKSolver ik2(cfg);
  auto sols2 = ik2.solve_multiple(goal, 4, 400);
  REQUIRE(sols2.size() == sols.size());
  CHECK(sols2[0].position == sols[0].position);
}

TEST_CASE("MotionGen: pose goal behind an obstacle is planned with a fixed seed and the multi-threaded planner agrees on success") {
  auto robot = ur5e();
  auto world = wall_world(*robot);
  JointState start{kStart, {}};
  JointState goal_q{kGoal, {}};
  auto fk       = robot->forward_kinematics(goal_q).back();
  Pose goal;
  goal.position = cutil::Vec3f(fk(0, 3), fk(1, 3), fk(2, 3));
  cutil::Mat<3, 3, float> r3;
  for(unsigned i = 0; i < 3; i++)
    for(unsigned j = 0; j < 3; j++) r3(i, j) = fk(i, j);
  goal.quaternion = cutil::Quatf::from_mat3(r3);

  for(int threads : {1, 4}) {
    MotionGenConfig cfg = MotionGenConfig::load_from_robot_config(robot, world);
    cfg.seed            = 9;
    cfg.num_threads     = threads;
    MotionGen gen(cfg);
    MotionGenResult res = gen.plan_single(start, goal);
    REQUIRE(res.success);
    auto end_fk = robot->forward_kinematics(res.trajectory.back()).back();
    CHECK(std::abs(end_fk(0, 3) - goal.position[0]) < 2e-3f);
    CHECK(std::abs(end_fk(1, 3) - goal.position[1]) < 2e-3f);
    CHECK(std::abs(end_fk(2, 3) - goal.position[2]) < 2e-3f);
  }
}

TEST_CASE("gpu_search: MotionGen solves around an obstacle on the GPU and the path is collision-free") {
  auto robot = ur5e();
  auto world = wall_world(*robot);
  MotionGenConfig cfg = MotionGenConfig::load_from_robot_config(robot, world);
  cfg.gpu_search      = true;
  cfg.seed            = 5;
  MotionGen mg(cfg);
  try {
    mg.warmup();
  } catch(const std::exception&) {
    MESSAGE("Vulkan unavailable; skipping");
    return;
  }
  if(!mg.robot_world_for_test().gpu_rrt_available()) {
    MESSAGE("GPU RRT not available on this device; skipping");
    return;
  }
  JointState start{kStart, {}}, goal{kGoal, {}};
  auto r = mg.plan_single_js(start, goal);
  REQUIRE(r.success);
  FastCollision fc(*robot, AllowedCollisionMatrix{});
  REQUIRE(fc.set_world(*world));
  for(const auto& s : r.trajectory) CHECK(fc.is_free(s.position.data()));
  for(size_t j = 0; j < kStart.size(); j++) CHECK(r.trajectory.front().position[j] == doctest::Approx(kStart[j]).epsilon(1e-4));
  for(size_t j = 0; j < kGoal.size(); j++) CHECK(r.trajectory.back().position[j] == doctest::Approx(kGoal[j]).epsilon(1e-4));
}

TEST_CASE("plan_batch_js: solves several problems with different worlds at once and every path is collision-free") {
  auto robot = ur5e();
  MotionGenConfig cfg = MotionGenConfig::load_from_robot_config(robot, cutil::make_ref<WorldConfig>());
  cfg.seed            = 9;
  MotionGen mg(cfg);
  try {
    mg.warmup();
  } catch(const std::exception&) {
    MESSAGE("Vulkan unavailable; skipping");
    return;
  }
  std::vector<MotionGen::BatchProblem> problems;
  std::vector<cutil::Ref<WorldConfig>> worlds;
  for(int k = 0; k < 4; k++) {
    auto world = wall_world(*robot);
    if(k == 1) world = cutil::make_ref<WorldConfig>();  // 障害物なし(直線で通る)
    worlds.push_back(world);
    problems.push_back({JointState{kStart, {}}, JointState{kGoal, {}}, world});
  }
  auto results = mg.plan_batch_js(problems);
  REQUIRE(results.size() == problems.size());
  for(size_t k = 0; k < problems.size(); k++) {
    REQUIRE(results[k].success);
    FastCollision fc(*robot, AllowedCollisionMatrix{});
    REQUIRE(fc.set_world(*worlds[k]));
    for(const auto& s : results[k].trajectory) CHECK(fc.is_free(s.position.data()));
    for(size_t j = 0; j < kGoal.size(); j++) CHECK(results[k].trajectory.back().position[j] == doctest::Approx(kGoal[j]).epsilon(1e-4));
  }
}
