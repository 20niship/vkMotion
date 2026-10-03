#include <doctest/doctest.h>

#include <algorithm>
#include <vkmotion/robot_config.hpp>
#include <vkmotion/robot_world.hpp>
#include <vkmotion/world_config.hpp>

// World positions below: independently computed by test_assets/gt/generate_gt.py (no vkMotion/cuRobo code).

namespace {

std::string asset_path(const char* relative) {
  return std::string(VKMOTION_TEST_ASSET_DIR) + "/" + relative;
}

vkmotion::Sphere sphere_obstacle(const cutil::Vec3f& position, float radius) {
  vkmotion::Sphere s;
  s.name     = "probe";
  s.pose     = vkmotion::Pose{position, cutil::Quatf::identity()};
  s.radius   = radius;
  return s;
}

}  // namespace

TEST_CASE("GT: OpenArm link_tip at zero joint state matches independent FK") {
  auto robot = vkmotion::RobotConfig::from_file(asset_path("urdf/openarm.urdf"));
  auto world = cutil::make_ref<vkmotion::WorldConfig>();
  const cutil::Vec3f tip_gt(0.000001f, 0.0205f, 0.5585f);
  world->add_obstacle(sphere_obstacle(tip_gt, 0.02f));

  vkmotion::RobotWorld rw(vkmotion::RobotWorldConfig{robot, world});
  vkmotion::JointState js{std::vector<float>(7, 0.0f), {}};

  CHECK_FALSE(rw.is_collision_free(js));
  const auto distances = rw.get_collision_distance(js);
  float min_distance    = *std::min_element(distances.begin(), distances.end());
  CHECK(min_distance == doctest::Approx(-(0.022f + 0.02f)).epsilon(0.01));
}

TEST_CASE("GT: OpenArm link_tip at non-zero joint state, obstacle far away is collision-free") {
  auto robot = vkmotion::RobotConfig::from_file(asset_path("urdf/openarm.urdf"));
  auto world = cutil::make_ref<vkmotion::WorldConfig>();
  world->add_obstacle(sphere_obstacle(cutil::Vec3f(10.0f, 10.0f, 10.0f), 0.02f));

  vkmotion::RobotWorld rw(vkmotion::RobotWorldConfig{robot, world});
  vkmotion::JointState js{{0.5f, -0.3f, 0.2f, 0.0f, 0.4f, -0.6f, 0.1f}, {}};
  CHECK(rw.is_collision_free(js));
}

TEST_CASE("GT: SO-101 moving_jaw at zero joint state matches independent FK") {
  auto robot = vkmotion::RobotConfig::from_file(asset_path("urdf/so101.urdf"));
  auto world = cutil::make_ref<vkmotion::WorldConfig>();
  const cutil::Vec3f tip_gt(0.31663371f, 0.0176182f, 0.25546141f);
  world->add_obstacle(sphere_obstacle(tip_gt, 0.01f));

  vkmotion::RobotWorld rw(vkmotion::RobotWorldConfig{robot, world});
  vkmotion::JointState js{std::vector<float>(6, 0.0f), {}};

  CHECK_FALSE(rw.is_collision_free(js));
  const auto distances = rw.get_collision_distance(js);
  float min_distance    = *std::min_element(distances.begin(), distances.end());
  CHECK(min_distance == doctest::Approx(-(0.02f + 0.01f)).epsilon(0.01));
}

TEST_CASE("GT: SO-101 non-zero joint state, obstacle far away is collision-free") {
  auto robot = vkmotion::RobotConfig::from_file(asset_path("urdf/so101.urdf"));
  auto world = cutil::make_ref<vkmotion::WorldConfig>();
  world->add_obstacle(sphere_obstacle(cutil::Vec3f(-10.0f, -10.0f, -10.0f), 0.01f));

  vkmotion::RobotWorld rw(vkmotion::RobotWorldConfig{robot, world});
  vkmotion::JointState js{{0.4f, -0.5f, 0.6f, -0.2f, 0.3f, 0.0f}, {}};
  CHECK(rw.is_collision_free(js));
}

TEST_CASE("GT: UR5e wrist_3_link at zero joint state matches independent FK") {
  auto robot = vkmotion::RobotConfig::from_file(asset_path("urdf/ur5e.urdf"));
  auto world = cutil::make_ref<vkmotion::WorldConfig>();
  const cutil::Vec3f tip_gt(-0.8172f, -0.2329f, 0.0628f);
  world->add_obstacle(sphere_obstacle(tip_gt, 0.02f));

  vkmotion::RobotWorld rw(vkmotion::RobotWorldConfig{robot, world});
  vkmotion::JointState js{std::vector<float>(6, 0.0f), {}};

  CHECK_FALSE(rw.is_collision_free(js));
  const auto distances = rw.get_collision_distance(js);
  float min_distance    = *std::min_element(distances.begin(), distances.end());
  CHECK(min_distance == doctest::Approx(-(0.035f + 0.02f)).epsilon(0.01));
}

TEST_CASE("GT: UR5e non-zero joint state, obstacle far away is collision-free") {
  auto robot = vkmotion::RobotConfig::from_file(asset_path("urdf/ur5e.urdf"));
  auto world = cutil::make_ref<vkmotion::WorldConfig>();
  world->add_obstacle(sphere_obstacle(cutil::Vec3f(10.0f, -10.0f, 10.0f), 0.02f));

  vkmotion::RobotWorld rw(vkmotion::RobotWorldConfig{robot, world});
  vkmotion::JointState js{{0.1f, -1.0f, 1.2f, -0.5f, 0.7f, 0.0f}, {}};
  CHECK(rw.is_collision_free(js));
}
