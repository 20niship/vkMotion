#include <cmath>
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

TEST_CASE("RobotWorld: GPU repair_waypoints (GPU常駐FK+勾配降下) matches CPU convergence") {
  auto robot = load_two_link_arm();
  auto world = cutil::make_ref<WorldConfig>();

  // link2's sphere traces an arc of radius 0.3 at z=0.3 as joint1 sweeps; block the arc midpoint.
  Sphere blocker;
  blocker.name          = "blocker";
  blocker.pose.position = cutil::Vec3f(0.3f * std::cos(0.5f), 0.3f * std::sin(0.5f), 0.3f);
  blocker.radius         = 0.03f;
  world->add_obstacle(blocker);

  RobotWorld cpu(RobotWorldConfig{robot, world, false});
  RobotWorld gpu(RobotWorldConfig{robot, world, true});

  std::vector<float> q_cpu = {0.5f, 0.0f};  // on the blocked arc midpoint, in collision
  std::vector<float> q_gpu = q_cpu;
  std::vector<std::vector<float>*> qs_cpu = {&q_cpu};
  std::vector<std::vector<float>*> qs_gpu = {&q_gpu};

  std::vector<bool> cpu_ok = cpu.repair_waypoints(qs_cpu, 0.05f, 1e-2f, 80);
  std::vector<bool> gpu_ok = gpu.repair_waypoints(qs_gpu, 0.05f, 1e-2f, 80);

  REQUIRE(cpu_ok.size() == 1);
  REQUIRE(gpu_ok.size() == 1);
  CHECK(cpu_ok[0]);
  CHECK(gpu_ok[0]);

  // GPU解がCPU参照の衝突判定でも実際にcollision-freeであることを確認(数値誤差で経路がずれても結果の妥当性は保証する)。
  CHECK(cpu.is_collision_free(JointState{q_gpu, {}}));
}

TEST_CASE("RobotWorld: GPU repair_waypoints handles multiple waypoints in one dispatch") {
  auto robot = load_two_link_arm();
  auto world = cutil::make_ref<WorldConfig>();

  Sphere blocker;
  blocker.name          = "blocker";
  blocker.pose.position = cutil::Vec3f(0.3f * std::cos(0.5f), 0.3f * std::sin(0.5f), 0.3f);
  blocker.radius         = 0.03f;
  world->add_obstacle(blocker);

  RobotWorld gpu(RobotWorldConfig{robot, world, true});

  std::vector<float> q0 = {0.5f, 0.0f};   // in collision
  std::vector<float> q1 = {-1.0f, 0.4f};  // far from the obstacle, already free
  std::vector<std::vector<float>*> qs = {&q0, &q1};

  std::vector<bool> ok = gpu.repair_waypoints(qs, 0.05f, 1e-2f, 80);
  REQUIRE(ok.size() == 2);
  CHECK(ok[0]);
  CHECK(ok[1]);

  RobotWorld cpu(RobotWorldConfig{robot, world, false});
  CHECK(cpu.is_collision_free(JointState{q0, {}}));
  CHECK(cpu.is_collision_free(JointState{q1, {}}));
}
