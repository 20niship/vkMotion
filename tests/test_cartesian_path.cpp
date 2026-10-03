#include <doctest/doctest.h>
#include <vkmotion/cartesian_path.hpp>
#include <vkmotion/robot_world.hpp>

using namespace vkmotion;

namespace {

Pose ee_pose_of(const RobotConfig& robot, const JointState& js) {
  auto world    = robot.forward_kinematics(js);
  const auto& m = world.back();
  Pose p;
  p.position = cutil::Vec3f(m(0, 3), m(1, 3), m(2, 3));
  cutil::Mat<3, 3, float> r;
  for(unsigned i = 0; i < 3; i++)
    for(unsigned j = 0; j < 3; j++) r(i, j) = m(i, j);
  p.quaternion = cutil::Quatf::from_mat3(r);
  return p;
}

cutil::Ref<RobotConfig> load_ur5e() {
  return RobotConfig::from_file(std::string(VKMOTION_TEST_ASSET_DIR) + "/urdf/ur5e.urdf");
}

// two_link_arm has only 2 DOF and can't generally satisfy a full 6-DoF Cartesian target; use ur5e (6 DOF).
JointState ur5e_start() {
  return JointState{{0.2f, -1.2f, 1.2f, -1.0f, -1.0f, 0.3f}, {}};
}

}  // namespace

TEST_CASE("CartesianPathPlanner: ur5e reaches a short obstacle-free straight line") {
  auto robot = load_ur5e();
  auto world = cutil::make_ref<WorldConfig>();

  JointState start = ur5e_start();
  Pose start_pose  = ee_pose_of(*robot, start);
  CartesianWaypoint wp;
  wp.pose           = start_pose;
  wp.pose.position += cutil::Vec3f(0.0f, 0.03f, 0.0f);

  CartesianPathPlanner planner(robot, world, false);
  CartesianPathResult result = planner.compute_cartesian_path(start, {wp});

  CHECK(result.success);
  CHECK(result.fraction_achieved == doctest::Approx(1.0f));
  CHECK(result.trajectory.size() > 1);
}

TEST_CASE("CartesianPathPlanner: stops partway when an obstacle blocks the straight line") {
  auto robot = load_ur5e();
  auto world = cutil::make_ref<WorldConfig>();

  JointState start = ur5e_start();
  Pose start_pose  = ee_pose_of(*robot, start);
  CartesianWaypoint wp;
  wp.pose           = start_pose;
  wp.pose.position += cutil::Vec3f(0.0f, 0.15f, 0.0f);

  Cuboid blocker;
  blocker.name          = "blocker";
  blocker.pose.position = start_pose.position + cutil::Vec3f(0.0f, 0.07f, 0.0f);
  blocker.dims           = cutil::Vec3f(0.1f, 0.1f, 0.1f);
  world->add_obstacle(blocker);

  CartesianPathPlanner planner(robot, world, false);
  CartesianPathResult result = planner.compute_cartesian_path(start, {wp});

  CHECK_FALSE(result.success);
  CHECK(result.fraction_achieved < 1.0f);

  RobotWorld rw(RobotWorldConfig{robot, world, false});
  for(const auto& js : result.trajectory) CHECK(rw.is_collision_free(js));
}

TEST_CASE("CartesianPathPlanner: ur5e solves a short straight-line move via Vulkan collision backend") {
  auto robot = load_ur5e();
  auto world = cutil::make_ref<WorldConfig>();

  JointState start = ur5e_start();
  Pose start_pose  = ee_pose_of(*robot, start);
  CartesianWaypoint wp;
  wp.pose           = start_pose;
  wp.pose.position += cutil::Vec3f(0.03f, 0.0f, 0.0f);

  CartesianPathPlanner planner(robot, world, true);
  CartesianPathResult result = planner.compute_cartesian_path(start, {wp});

  CHECK(result.success);
  CHECK(result.fraction_achieved == doctest::Approx(1.0f));
}
