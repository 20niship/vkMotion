#include <cmath>
#include <doctest/doctest.h>
#include <vkmotion/motion_gen.hpp>
#include <vkmotion/robot_world.hpp>

using namespace vkmotion;

namespace {
cutil::Ref<RobotConfig> load_two_link_arm() {
  return RobotConfig::from_file(std::string(VKMOTION_TEST_ASSET_DIR) + "/urdf/two_link_arm.urdf");
}
}  // namespace

TEST_CASE("MotionGen: two_link_arm plans an obstacle-free trajectory to a reachable goal pose") {
  auto robot = load_two_link_arm();
  auto world = cutil::make_ref<WorldConfig>();

  JointState start{{0.0f, 0.0f}, {}};
  JointState goal_q{{1.0f, 0.3f}, {}};
  auto goal_fk  = robot->forward_kinematics(goal_q);
  const auto& m = goal_fk.back();
  Pose goal;
  goal.position = cutil::Vec3f(m(0, 3), m(1, 3), m(2, 3));
  cutil::Mat<3, 3, float> r;
  for(unsigned i = 0; i < 3; i++)
    for(unsigned j = 0; j < 3; j++) r(i, j) = m(i, j);
  goal.quaternion = cutil::Quatf::from_mat3(r);

  MotionGen gen(MotionGenConfig::load_from_robot_config(robot, world));
  gen.warmup();
  MotionGenResult result = gen.plan_single(start, goal);

  REQUIRE(result.success);
  REQUIRE(result.trajectory.size() >= 2);
  CHECK(result.trajectory.front().position[0] == doctest::Approx(start.position[0]));
  CHECK(result.trajectory.front().position[1] == doctest::Approx(start.position[1]));

  auto final_fk     = robot->forward_kinematics(result.trajectory.back());
  const auto& fm     = final_fk.back();
  cutil::Vec3f got(fm(0, 3), fm(1, 3), fm(2, 3));
  cutil::Vec3f d = got - goal.position;
  CHECK(std::sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2]) < 1e-2f);
}

TEST_CASE("MotionGen: two_link_arm routes around an obstacle blocking the straight-line joint path") {
  auto robot = load_two_link_arm();
  auto world = cutil::make_ref<WorldConfig>();

  JointState start{{0.0f, 0.0f}, {}};
  JointState goal_q{{1.0f, 0.0f}, {}};

  // link2's sphere traces an arc of radius 0.3 at z=0.3 as joint1 sweeps; block the arc midpoint.
  Sphere blocker;
  blocker.name          = "blocker";
  blocker.pose.position = cutil::Vec3f(0.3f * std::cos(0.5f), 0.3f * std::sin(0.5f), 0.3f);
  blocker.radius         = 0.03f;
  world->add_obstacle(blocker);

  auto goal_fk  = robot->forward_kinematics(goal_q);
  const auto& m = goal_fk.back();
  Pose goal;
  goal.position = cutil::Vec3f(m(0, 3), m(1, 3), m(2, 3));
  cutil::Mat<3, 3, float> r;
  for(unsigned i = 0; i < 3; i++)
    for(unsigned j = 0; j < 3; j++) r(i, j) = m(i, j);
  goal.quaternion = cutil::Quatf::from_mat3(r);

  MotionGenConfig cfg = MotionGenConfig::load_from_robot_config(robot, world);
  MotionGen gen(cfg);
  MotionGenResult result = gen.plan_single_js(start, goal_q);

  REQUIRE(result.success);

  RobotWorld rw(RobotWorldConfig{robot, world, false});
  for(const auto& js : result.trajectory) CHECK(rw.is_collision_free(js));
}

TEST_CASE("MotionGen: ur5e plans a trajectory to a reachable goal pose") {
  auto robot = RobotConfig::from_file(std::string(VKMOTION_TEST_ASSET_DIR) + "/urdf/ur5e.urdf");
  auto world = cutil::make_ref<WorldConfig>();

  JointState start{{0.0f, -1.2f, 1.0f, -0.5f, 0.3f, 0.0f}, {}};
  JointState goal_q{{0.3f, -1.0f, 1.1f, -0.7f, 0.5f, 0.2f}, {}};
  auto goal_fk  = robot->forward_kinematics(goal_q);
  const auto& m = goal_fk.back();
  Pose goal;
  goal.position = cutil::Vec3f(m(0, 3), m(1, 3), m(2, 3));
  cutil::Mat<3, 3, float> r;
  for(unsigned i = 0; i < 3; i++)
    for(unsigned j = 0; j < 3; j++) r(i, j) = m(i, j);
  goal.quaternion = cutil::Quatf::from_mat3(r);

  MotionGen gen(MotionGenConfig::load_from_robot_config(robot, world));
  MotionGenResult result = gen.plan_single(start, goal);

  REQUIRE(result.success);
  REQUIRE_FALSE(result.trajectory.empty());
}
