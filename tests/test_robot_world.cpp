#include <doctest/doctest.h>
#include <vkmotion/robot_world.hpp>

using namespace vkmotion;

namespace {
cutil::Ref<RobotConfig> load_two_link_arm() {
  return RobotConfig::from_file(std::string(VKMOTION_TEST_ASSET_DIR) + "/urdf/two_link_arm.urdf");
}
}  // namespace

TEST_CASE("RobotWorld: empty world is always collision-free") {
  auto robot = load_two_link_arm();
  auto world = cutil::make_ref<WorldConfig>();
  RobotWorld rw(RobotWorldConfig{robot, world, false});

  CHECK(rw.is_collision_free(JointState{{0.0f, 0.0f}, {}}));
}

TEST_CASE("RobotWorld: obstacle overlapping the workspace is detected") {
  auto robot = load_two_link_arm();
  auto world = cutil::make_ref<WorldConfig>();

  // link2's sphere sits at (0.3, 0, 0.3) in the all-zero joint state (see test_fk.cpp).
  Cuboid blocker;
  blocker.name          = "blocker";
  blocker.pose.position = cutil::Vec3f(0.3f, 0.0f, 0.3f);
  blocker.dims           = cutil::Vec3f(0.5f, 0.5f, 0.5f);
  world->add_obstacle(blocker);

  RobotWorld rw(RobotWorldConfig{robot, world, false});
  CHECK_FALSE(rw.is_collision_free(JointState{{0.0f, 0.0f}, {}}));
}

TEST_CASE("RobotWorld: adjacent-link spheres are excluded from self-collision") {
  auto robot = load_two_link_arm();
  auto world = cutil::make_ref<WorldConfig>();
  RobotWorld rw(RobotWorldConfig{robot, world, false});

  // Only the non-adjacent base_link<->link2 sphere pair should remain.
  auto distances = rw.get_self_collision_distance(JointState{{0.0f, 0.0f}, {}});
  CHECK(distances.size() == 1);
  CHECK(distances[0] > 0.0f);
}
