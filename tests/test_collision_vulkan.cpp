#include <cmath>
#include <doctest/doctest.h>
#include <vkmotion/robot_world.hpp>

using namespace vkmotion;

namespace {
cutil::Ref<RobotConfig> load_two_link_arm() { return RobotConfig::from_file(std::string(VKMOTION_TEST_ASSET_DIR) + "/urdf/two_link_arm.urdf"); }

void check_distances_match(const std::vector<float>& gpu_d, const std::vector<float>& cpu_d) {
  REQUIRE(gpu_d.size() == cpu_d.size());
  for(size_t i = 0; i < cpu_d.size(); i++) {
    if(std::isinf(cpu_d[i])) {
      CHECK(std::isinf(gpu_d[i]));
    } else {
      CHECK(gpu_d[i] == doctest::Approx(cpu_d[i]).epsilon(1e-4));
    }
  }
}
}  // namespace

TEST_CASE("RobotWorld Vulkan: matches CPU on an empty world") {
  auto robot = load_two_link_arm();
  auto world = cutil::make_ref<WorldConfig>();

  RobotWorld cpu(RobotWorldConfig{robot, world, false});
  RobotWorld gpu(RobotWorldConfig{robot, world, true});

  JointState js{{0.3f, -0.6f}, {}};
  check_distances_match(gpu.get_collision_distance(js), cpu.get_collision_distance(js));

  CHECK(gpu.is_collision_free(js) == cpu.is_collision_free(js));
}

TEST_CASE("RobotWorld Vulkan: matches CPU when an obstacle overlaps the workspace") {
  auto robot = load_two_link_arm();
  auto world = cutil::make_ref<WorldConfig>();

  Cuboid blocker;
  blocker.name          = "blocker";
  blocker.pose.position = cutil::Vec3f(0.3f, 0.0f, 0.3f);
  blocker.dims           = cutil::Vec3f(0.5f, 0.5f, 0.5f);
  world->add_obstacle(blocker);

  RobotWorld cpu(RobotWorldConfig{robot, world, false});
  RobotWorld gpu(RobotWorldConfig{robot, world, true});

  JointState js{{0.0f, 0.0f}, {}};
  check_distances_match(gpu.get_collision_distance(js), cpu.get_collision_distance(js));

  CHECK_FALSE(gpu.is_collision_free(js));
  CHECK(gpu.is_collision_free(js) == cpu.is_collision_free(js));
}

TEST_CASE("RobotWorld Vulkan: matches CPU for plane/convex/sdf obstacles") {
  auto robot = load_two_link_arm();
  auto world = cutil::make_ref<WorldConfig>();

  // link2's sphere sits at (0.3, 0, 0.3) at the all-zero joint state (see test_fk.cpp).
  Plane floor;
  floor.name           = "floor";
  floor.pose.position = cutil::Vec3f(0.0f, 0.0f, -0.5f);  // normal = +Z, far below the arm
  world->add_obstacle(floor);

  ConvexMesh hull;
  hull.name          = "hull";
  hull.pose.position = cutil::Vec3f(0.3f, 0.0f, 0.3f);
  for(float sx : {-1.0f, 1.0f})
    for(float sy : {-1.0f, 1.0f})
      for(float sz : {-1.0f, 1.0f}) hull.vertices.push_back(cutil::Vec3f(sx * 0.1f, sy * 0.1f, sz * 0.1f));
  world->add_obstacle(hull);

  SDFGrid grid;
  grid.name            = "grid";
  grid.pose.position = cutil::Vec3f(-5.0f, -5.0f, -5.0f);  // far from the arm, all distances large (no collision)
  grid.dims            = cutil::Vec3f(2.0f, 2.0f, 2.0f);
  grid.resolution[0] = grid.resolution[1] = grid.resolution[2] = 2;
  grid.distances                                                   = {5.0f, 5.0f, 5.0f, 5.0f, 5.0f, 5.0f, 5.0f, 5.0f};
  world->add_obstacle(grid);

  RobotWorld cpu(RobotWorldConfig{robot, world, false});
  RobotWorld gpu(RobotWorldConfig{robot, world, true});

  JointState js{{0.0f, 0.0f}, {}};
  check_distances_match(gpu.get_collision_distance(js), cpu.get_collision_distance(js));
  CHECK_FALSE(gpu.is_collision_free(js));  // the convex hull around link2's sphere collides
  CHECK(gpu.is_collision_free(js) == cpu.is_collision_free(js));
}

TEST_CASE("RobotWorld Vulkan: batch dispatch matches per-state CPU results") {
  auto robot = load_two_link_arm();
  auto world = cutil::make_ref<WorldConfig>();

  Cuboid blocker;
  blocker.name          = "blocker";
  blocker.pose.position = cutil::Vec3f(0.3f, 0.0f, 0.3f);
  blocker.dims           = cutil::Vec3f(0.2f, 0.2f, 0.2f);
  world->add_obstacle(blocker);

  RobotWorld cpu(RobotWorldConfig{robot, world, false});
  RobotWorld gpu(RobotWorldConfig{robot, world, true});

  std::vector<JointState> states = {
      JointState{{0.0f, 0.0f}, {}},
      JointState{{1.0f, 0.5f}, {}},
      JointState{{-0.8f, 1.2f}, {}},
  };

  std::vector<bool> cpu_results = cpu.is_collision_free_batch(states);
  std::vector<bool> gpu_results = gpu.is_collision_free_batch(states);

  REQUIRE(cpu_results.size() == gpu_results.size());
  for(size_t i = 0; i < cpu_results.size(); i++) CHECK(gpu_results[i] == cpu_results[i]);
}
