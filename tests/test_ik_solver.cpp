#include <doctest/doctest.h>
#include <vkmotion/ik_solver.hpp>
#include <vkmotion/robot_config.hpp>

namespace {

float pos_distance(const cutil::Vec3f& a, const cutil::Vec3f& b) {
  cutil::Vec3f d = a - b;
  return std::sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2]);
}

}  // namespace

TEST_CASE("IKSolver: two_link_arm converges to a known-reachable goal from a different seed") {
  auto robot = vkmotion::RobotConfig::from_file(std::string(VKMOTION_TEST_ASSET_DIR) + "/urdf/two_link_arm.urdf");

  vkmotion::JointState goal_q{{0.6f, -0.4f}, {}};
  auto world  = robot->forward_kinematics(goal_q);
  int ee_idx  = static_cast<int>(robot->link_names().size()) - 1;
  const auto& m = world[static_cast<size_t>(ee_idx)];
  vkmotion::Pose goal;
  goal.position = cutil::Vec3f(m(0, 3), m(1, 3), m(2, 3));
  cutil::Mat<3, 3, float> goal_r;
  for(unsigned i = 0; i < 3; i++)
    for(unsigned j = 0; j < 3; j++) goal_r(i, j) = m(i, j);
  goal.quaternion = cutil::Quatf::from_mat3(goal_r);

  vkmotion::IKSolverConfig cfg;
  cfg.robot_config = robot;
  vkmotion::IKSolver solver(cfg);

  vkmotion::JointState seed{{0.0f, 0.0f}, {}};
  auto result = solver.solve_single(goal, seed);
  REQUIRE(result.has_value());

  auto check_world = robot->forward_kinematics(*result);
  const auto& cm    = check_world[static_cast<size_t>(ee_idx)];
  cutil::Vec3f got(cm(0, 3), cm(1, 3), cm(2, 3));
  CHECK(pos_distance(got, goal.position) < 1e-2f);
}

TEST_CASE("IKSolver: ur5e converges to a known-reachable full-pose goal from a different seed") {
  auto robot = vkmotion::RobotConfig::from_file(std::string(VKMOTION_TEST_ASSET_DIR) + "/urdf/ur5e.urdf");

  vkmotion::JointState goal_q{{0.3f, -1.2f, 1.1f, -0.7f, 0.5f, 0.2f}, {}};
  auto world  = robot->forward_kinematics(goal_q);
  int ee_idx  = static_cast<int>(robot->link_names().size()) - 1;
  const auto& m = world[static_cast<size_t>(ee_idx)];
  vkmotion::Pose goal;
  goal.position = cutil::Vec3f(m(0, 3), m(1, 3), m(2, 3));
  cutil::Mat<3, 3, float> r;
  for(unsigned i = 0; i < 3; i++)
    for(unsigned j = 0; j < 3; j++) r(i, j) = m(i, j);
  goal.quaternion = cutil::Quatf::from_mat3(r);

  vkmotion::IKSolverConfig cfg;
  cfg.robot_config = robot;
  vkmotion::IKSolver solver(cfg);

  vkmotion::JointState seed{{0.2f, -1.0f, 1.0f, -0.6f, 0.4f, 0.1f}, {}};
  auto result = solver.solve_single(goal, seed);
  REQUIRE(result.has_value());

  auto check_world = robot->forward_kinematics(*result);
  const auto& cm    = check_world[static_cast<size_t>(ee_idx)];
  cutil::Vec3f got(cm(0, 3), cm(1, 3), cm(2, 3));
  CHECK(pos_distance(got, goal.position) < 1e-2f);
}

TEST_CASE("IKSolver: unreachable goal returns nullopt") {
  auto robot = vkmotion::RobotConfig::from_file(std::string(VKMOTION_TEST_ASSET_DIR) + "/urdf/two_link_arm.urdf");

  vkmotion::Pose goal;
  goal.position = cutil::Vec3f(1000.0f, 1000.0f, 1000.0f);

  vkmotion::IKSolverConfig cfg;
  cfg.robot_config = robot;
  vkmotion::IKSolver solver(cfg);

  vkmotion::JointState seed{{0.0f, 0.0f}, {}};
  auto result = solver.solve_single(goal, seed);
  CHECK_FALSE(result.has_value());
}
