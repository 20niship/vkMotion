#include <doctest/doctest.h>
#include <vkmotion/constraints.hpp>
#include <vkmotion/motion_gen.hpp>
#include <vkmotion/robot_config.hpp>

using namespace vkmotion;

namespace {
cutil::Ref<RobotConfig> load_ur5e() {
  return RobotConfig::from_file(std::string(VKMOTION_TEST_ASSET_DIR) + "/urdf/ur5e.urdf");
}
cutil::Ref<RobotConfig> load_two_link_arm() {
  return RobotConfig::from_file(std::string(VKMOTION_TEST_ASSET_DIR) + "/urdf/two_link_arm.urdf");
}
cutil::Vec3f ee_position(const RobotConfig& robot, const JointState& js) {
  const cutil::Mat4f& t = robot.forward_kinematics(js).back();
  return cutil::Vec3f(t(0, 3), t(1, 3), t(2, 3));
}
cutil::Quatf ee_orientation(const RobotConfig& robot, const JointState& js) {
  const cutil::Mat4f& t = robot.forward_kinematics(js).back();
  cutil::Mat<3, 3, float> r;
  for(unsigned i = 0; i < 3; i++)
    for(unsigned j = 0; j < 3; j++) r(i, j) = t(i, j);
  return cutil::Quatf::from_mat3(r);
}
}  // namespace

TEST_CASE("JointConstraint: satisfied inside range, violated outside") {
  auto robot = load_two_link_arm();
  JointConstraint c{0, -0.5f, 0.5f};
  CHECK(c.is_satisfied(JointState{{0.0f, 0.0f}, {}}, *robot));
  CHECK_FALSE(c.is_satisfied(JointState{{1.0f, 0.0f}, {}}, *robot));
  CHECK(c.violation(JointState{{1.0f, 0.0f}, {}}, *robot) > 0.0f);
}

TEST_CASE("PositionConstraint: satisfied near target, violated far away") {
  auto robot = load_ur5e();
  JointState q{{0.2f, -1.0f, 1.0f, -0.5f, -0.3f, 0.1f}, {}};
  cutil::Vec3f ee = ee_position(*robot, q);

  PositionConstraint tight{ee, cutil::Vec3f(0.02f, 0.02f, 0.02f)};
  CHECK(tight.is_satisfied(q, *robot));

  PositionConstraint far{cutil::Vec3f(ee[0] + 1.0f, ee[1], ee[2]), cutil::Vec3f(0.02f, 0.02f, 0.02f)};
  CHECK_FALSE(far.is_satisfied(q, *robot));
  CHECK(far.violation(q, *robot) > 0.0f);
}

TEST_CASE("OrientationConstraint: satisfied at target, violated after a large rotation") {
  auto robot = load_ur5e();
  JointState q{{0.2f, -1.0f, 1.0f, -0.5f, -0.3f, 0.1f}, {}};
  cutil::Quatf ee_q = ee_orientation(*robot, q);

  OrientationConstraint tight{ee_q, 0.05f};
  CHECK(tight.is_satisfied(q, *robot));

  JointState q_rotated{{0.2f, -1.0f, 1.0f, -0.5f, -0.3f, 0.1f + 1.5f}, {}};
  OrientationConstraint still_target_orig{ee_q, 0.05f};
  CHECK_FALSE(still_target_orig.is_satisfied(q_rotated, *robot));
}

TEST_CASE("ConstraintSet: combines multiple constraints with AND semantics") {
  auto robot = load_two_link_arm();
  ConstraintSet set;
  set.add(JointConstraint{0, -0.5f, 0.5f});
  set.add(JointConstraint{1, -0.5f, 0.5f});

  CHECK(set.is_satisfied(JointState{{0.0f, 0.0f}, {}}, *robot));
  CHECK_FALSE(set.is_satisfied(JointState{{1.0f, 0.0f}, {}}, *robot));
  CHECK_FALSE(set.is_satisfied(JointState{{0.0f, 1.0f}, {}}, *robot));
  CHECK(set.total_violation(JointState{{1.0f, 1.0f}, {}}, *robot) > 0.0f);
}

TEST_CASE("MotionGen: PositionConstraint is satisfied along the generated trajectory") {
  auto robot = load_ur5e();
  auto world = cutil::make_ref<WorldConfig>();

  JointState start{{0.0f, -1.2f, 1.2f, -0.3f, -0.2f, 0.0f}, {}};
  JointState goal{{0.3f, -1.0f, 1.0f, -0.5f, -0.3f, 0.2f}, {}};
  cutil::Vec3f start_ee = ee_position(*robot, start);
  cutil::Vec3f goal_ee   = ee_position(*robot, goal);
  cutil::Vec3f mid((start_ee[0] + goal_ee[0]) * 0.5f, (start_ee[1] + goal_ee[1]) * 0.5f, (start_ee[2] + goal_ee[2]) * 0.5f);
  cutil::Vec3f half_extent(std::abs(start_ee[0] - goal_ee[0]) * 0.5f + 0.3f, std::abs(start_ee[1] - goal_ee[1]) * 0.5f + 0.3f,
                            std::abs(start_ee[2] - goal_ee[2]) * 0.5f + 0.3f);

  ConstraintSet constraints;
  constraints.add(PositionConstraint{mid, half_extent});

  MotionGenConfig cfg = MotionGenConfig::load_from_robot_config(robot, world);
  cfg.use_vulkan        = false;
  cfg.constraints       = constraints;
  MotionGen gen(cfg);

  MotionGenResult result = gen.plan_single_js(start, goal);
  REQUIRE(result.success);
  for(const auto& js : result.trajectory) CHECK(constraints.is_satisfied(js, *robot));
}
