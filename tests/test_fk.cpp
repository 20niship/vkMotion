#include <doctest/doctest.h>
#include <vkmotion/robot_config.hpp>

namespace {
constexpr float kHalfPi = 1.5707963267948966f;
}

TEST_CASE("forward_kinematics: all-zero joint state places link2 along the Z chain") {
  auto robot = vkmotion::RobotConfig::from_file(std::string(VKMOTION_TEST_ASSET_DIR) + "/urdf/two_link_arm.urdf");
  REQUIRE(robot->dof() == 2);

  vkmotion::JointState js{{0.0f, 0.0f}, {}};
  auto world = robot->forward_kinematics(js);

  int link2_idx = robot->urdf_model().find_link("link2");
  REQUIRE(link2_idx >= 0);
  const auto& m = world[static_cast<size_t>(link2_idx)];
  CHECK(m(0, 3) == doctest::Approx(0.3f));
  CHECK(m(1, 3) == doctest::Approx(0.0f));
  CHECK(m(2, 3) == doctest::Approx(0.3f));
}

TEST_CASE("forward_kinematics: rotating joint1 by 90deg carries link2's X offset into Y") {
  auto robot = vkmotion::RobotConfig::from_file(std::string(VKMOTION_TEST_ASSET_DIR) + "/urdf/two_link_arm.urdf");

  vkmotion::JointState js{{kHalfPi, 0.0f}, {}};
  auto world = robot->forward_kinematics(js);

  int link2_idx = robot->urdf_model().find_link("link2");
  const auto& m = world[static_cast<size_t>(link2_idx)];
  CHECK(m(0, 3) == doctest::Approx(0.0f).epsilon(0.001));
  CHECK(m(1, 3) == doctest::Approx(0.3f));
  CHECK(m(2, 3) == doctest::Approx(0.3f));
}
