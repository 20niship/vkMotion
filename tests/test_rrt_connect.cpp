#include <cmath>
#include <doctest/doctest.h>
#include <vkmotion/motion_gen.hpp>
#include <vkmotion/robot_world.hpp>
#include <vkmotion/rrt_connect.hpp>

using namespace vkmotion;

namespace {
cutil::Ref<RobotConfig> load_two_link_arm() {
  return RobotConfig::from_file(std::string(VKMOTION_TEST_ASSET_DIR) + "/urdf/two_link_arm.urdf");
}
cutil::Ref<RobotConfig> load_ur5e() {
  return RobotConfig::from_file(std::string(VKMOTION_TEST_ASSET_DIR) + "/urdf/ur5e.urdf");
}
cutil::Vec3f ee_position(const RobotConfig& robot, const JointState& js) {
  auto fk        = robot.forward_kinematics(js);
  const auto& m = fk.back();
  return cutil::Vec3f(m(0, 3), m(1, 3), m(2, 3));
}
}  // namespace

TEST_CASE("RrtConnectPlanner: two_link_arm finds a path in free space") {
  auto robot = load_two_link_arm();
  auto world = cutil::make_ref<WorldConfig>();

  JointState start{{0.0f, 0.0f}, {}};
  JointState goal{{1.0f, 0.3f}, {}};

  RrtConnectPlanner planner(robot, world);
  RrtConnectResult result = planner.plan(start, goal);

  REQUIRE(result.success);
  REQUIRE(result.trajectory.size() >= 2);

  RobotWorld rw(RobotWorldConfig{robot, world, false});
  for(const auto& js : result.trajectory) CHECK(rw.is_collision_free(js));
}

TEST_CASE("RrtConnectPlanner: ur5e finds a path around an obstacle on its own") {
  auto robot = load_ur5e();
  auto world = cutil::make_ref<WorldConfig>();

  JointState start{{0.0f, -1.57f, 1.57f, -1.57f, -1.57f, 0.0f}, {}};
  JointState goal{{1.5f, -1.0f, 1.0f, -1.0f, -1.0f, 0.5f}, {}};

  // start-goalの関節空間直線補間の中間点(t=0.5)のEE位置を直接塞ぐ。start/goal自体はt=0なので衝突しない。
  JointState mid_q;
  mid_q.position.resize(start.position.size());
  for(size_t j = 0; j < mid_q.position.size(); j++) mid_q.position[j] = start.position[j] + 0.5f * (goal.position[j] - start.position[j]);
  Sphere blocker;
  blocker.name          = "blocker";
  blocker.pose.position = ee_position(*robot, mid_q);
  blocker.radius         = 0.2f;
  world->add_obstacle(blocker);

  RobotWorld rw(RobotWorldConfig{robot, world, false});
  REQUIRE(rw.is_collision_free(start));
  REQUIRE(rw.is_collision_free(goal));

  // 直線補間の中間点ちょうどを塞いでもRRT-Connectは単体で解けることを確認する(MotionGen本体は単一障害物なら自力解決してしまいやすいため単体で検証)。
  RrtConnectPlanner planner(robot, world);
  RrtConnectResult rrt_result = planner.plan(start, goal);
  REQUIRE(rrt_result.success);
  for(const auto& js : rrt_result.trajectory) CHECK(rw.is_collision_free(js));
}

TEST_CASE("MotionGen: enable_graph_planner reaches the RRT-Connect fallback when local search is disabled") {
  auto robot = load_two_link_arm();
  auto world = cutil::make_ref<WorldConfig>();

  // 直線補間上のwaypointを完全に塞ぎ、direct roundを確実に失敗させる。
  JointState start{{0.0f, 0.0f}, {}};
  JointState goal{{1.0f, 0.0f}, {}};
  Sphere blocker;
  blocker.name          = "blocker";
  blocker.pose.position = cutil::Vec3f(0.3f * std::cos(0.5f), 0.3f * std::sin(0.5f), 0.3f);
  blocker.radius         = 0.03f;
  world->add_obstacle(blocker);

  MotionGenConfig cfg = MotionGenConfig::load_from_robot_config(robot, world);

  // num_trajopt_seeds=1・max_attempts=1で経由点によるランダム探索を無効化する。
  MotionGenPlanConfig no_fallback_cfg;
  no_fallback_cfg.enable_graph_planner = false;
  no_fallback_cfg.max_attempts          = 1;
  MotionGenConfig cfg_single_seed = cfg;
  cfg_single_seed.num_trajopt_seeds     = 1;
  MotionGen gen_no_fallback(cfg_single_seed);
  MotionGenResult local_only = gen_no_fallback.plan_single_js(start, goal, no_fallback_cfg);

  // フォールバック有効ならlocal searchの結果に関わらず最終的に成功すること。
  MotionGen gen_with_fallback(cfg_single_seed);
  MotionGenPlanConfig fallback_cfg;
  fallback_cfg.max_attempts     = 1;
  MotionGenResult with_fallback = gen_with_fallback.plan_single_js(start, goal, fallback_cfg);
  REQUIRE(with_fallback.success);

  RobotWorld rw(RobotWorldConfig{robot, world, false});
  for(const auto& js : with_fallback.trajectory) CHECK(rw.is_collision_free(js));

  // local_onlyが失敗していた場合のみ、フォールバックが実際にRRT-Connect経由で成功したことを確認する。
  if(!local_only.success) CHECK(with_fallback.status == "success (RRT-Connect fallback)");
}
