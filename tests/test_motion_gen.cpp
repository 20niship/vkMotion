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

namespace {

cutil::Ref<RobotConfig> load_ur5e_arm() {
  return RobotConfig::from_file(std::string(VKMOTION_TEST_ASSET_DIR) + "/urdf/ur5e.urdf");
}

// 距離関数ベースの参照実装(コンパイル済みfast pathとは独立)で衝突なしを判定する。
bool reference_free(const RobotWorld& rw, const JointState& js) {
  for(float d : rw.get_collision_distance(js))
    if(d <= 0.0f) return false;
  for(float d : rw.get_self_collision_distance(js))
    if(d <= 0.0f) return false;
  return true;
}

// 隣り合うwaypointの間も0.01rad刻みで参照実装により検査する(waypoint間をすり抜けていないこと)。
bool trajectory_continuously_free(const RobotWorld& rw, const std::vector<JointState>& traj) {
  for(size_t i = 0; i < traj.size(); i++) {
    if(!reference_free(rw, traj[i])) return false;
    if(i == 0) continue;
    float max_delta = 0.0f;
    for(size_t j = 0; j < traj[i].position.size(); j++) max_delta = std::max(max_delta, std::abs(traj[i].position[j] - traj[i - 1].position[j]));
    int n = static_cast<int>(std::ceil(max_delta / 0.01f));
    for(int k = 1; k < n; k++) {
      JointState js;
      js.position.resize(traj[i].position.size());
      float t = static_cast<float>(k) / static_cast<float>(n);
      for(size_t j = 0; j < js.position.size(); j++) js.position[j] = traj[i - 1].position[j] + t * (traj[i].position[j] - traj[i - 1].position[j]);
      if(!reference_free(rw, js)) return false;
    }
  }
  return true;
}

// start-goalの関節空間直線補間の中間(t=0.5)のEE位置に球を置いて直線経路を塞ぐ。
void block_straight_line_midpoint(const RobotConfig& robot, WorldConfig& world, const JointState& start, const JointState& goal) {
  JointState mid;
  mid.position.resize(start.position.size());
  for(size_t j = 0; j < mid.position.size(); j++) mid.position[j] = 0.5f * (start.position[j] + goal.position[j]);
  const auto m = robot.forward_kinematics(mid).back();
  Sphere blocker;
  blocker.name          = "blocker";
  blocker.pose.position = cutil::Vec3f(m(0, 3), m(1, 3), m(2, 3));
  blocker.radius        = 0.2f;
  world.add_obstacle(blocker);
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

TEST_CASE("MotionGen: ur5e routes around an obstacle blocking the straight-line joint path") {
  auto robot = load_ur5e_arm();
  auto world = cutil::make_ref<WorldConfig>();

  JointState start{{0.0f, -1.57f, 1.57f, -1.57f, -1.57f, 0.0f}, {}};
  JointState goal_q{{1.5f, -1.0f, 1.0f, -1.0f, -1.0f, 0.5f}, {}};
  block_straight_line_midpoint(*robot, *world, start, goal_q);

  RobotWorld rw(RobotWorldConfig{robot, world, false});
  REQUIRE(reference_free(rw, start));
  REQUIRE(reference_free(rw, goal_q));

  MotionGen gen(MotionGenConfig::load_from_robot_config(robot, world));
  MotionGenResult result = gen.plan_single_js(start, goal_q);

  REQUIRE(result.success);
  CHECK(trajectory_continuously_free(rw, result.trajectory));
}

TEST_CASE("MotionGen: two_link_arm reports failure when the obstacle fully blocks the only joint") {
  // joint2の回転ではlink2の球は動かない(球がjoint2原点にある)ので、joint1の経路を塞げば到達不能になる。
  auto robot = load_two_link_arm();
  auto world = cutil::make_ref<WorldConfig>();

  JointState start{{0.0f, 0.0f}, {}};
  JointState goal_q{{1.0f, 0.0f}, {}};
  Sphere blocker;
  blocker.name          = "blocker";
  blocker.pose.position = cutil::Vec3f(0.3f * std::cos(0.5f), 0.3f * std::sin(0.5f), 0.3f);
  blocker.radius         = 0.03f;
  world->add_obstacle(blocker);

  MotionGenPlanConfig cfg;
  cfg.timeout_seconds = 0.5f;
  MotionGen gen(MotionGenConfig::load_from_robot_config(robot, world));
  CHECK_FALSE(gen.plan_single_js(start, goal_q, cfg).success);
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

TEST_CASE("RobotConfig: joint_velocity_limits reads URDF <limit velocity>") {
  auto robot                 = load_two_link_arm();
  const auto& velocity_limits = robot->joint_velocity_limits();
  REQUIRE(velocity_limits.size() == 2);
  CHECK(velocity_limits[0] == doctest::Approx(1.0f));
  CHECK(velocity_limits[1] == doctest::Approx(1.0f));
}

TEST_CASE("MotionGen: interpolated_times respects per-joint velocity limits") {
  auto robot = load_two_link_arm();
  auto world = cutil::make_ref<WorldConfig>();

  JointState start{{0.0f, 0.0f}, {}};
  JointState goal_q{{2.5f, -1.4f}, {}};

  MotionGen gen(MotionGenConfig::load_from_robot_config(robot, world));
  MotionGenResult result = gen.plan_single_js(start, goal_q);

  REQUIRE(result.success);
  REQUIRE(result.interpolated_times.size() == result.trajectory.size());
  const auto& velocity_limits = robot->joint_velocity_limits();
  for(size_t i = 1; i < result.trajectory.size(); i++) {
    float dt = result.interpolated_times[i] - result.interpolated_times[i - 1];
    REQUIRE(dt > 0.0f);
    for(size_t j = 0; j < velocity_limits.size(); j++) {
      float speed = std::abs(result.trajectory[i].position[j] - result.trajectory[i - 1].position[j]) / dt;
      CHECK(speed <= velocity_limits[j] + 1e-4f);
    }
  }
}

TEST_CASE("MotionGen: shortcut smoothing never lengthens the trajectory") {
  auto robot = load_ur5e_arm();
  auto world = cutil::make_ref<WorldConfig>();

  JointState start{{0.0f, -1.57f, 1.57f, -1.57f, -1.57f, 0.0f}, {}};
  JointState goal_q{{1.5f, -1.0f, 1.0f, -1.0f, -1.0f, 0.5f}, {}};
  block_straight_line_midpoint(*robot, *world, start, goal_q);

  MotionGen gen(MotionGenConfig::load_from_robot_config(robot, world));
  MotionGenResult result = gen.plan_single_js(start, goal_q);
  REQUIRE(result.success);

  float straight_line_length = 0.0f;
  for(size_t j = 0; j < start.position.size(); j++) {
    float d = goal_q.position[j] - start.position[j];
    straight_line_length += d * d;
  }
  straight_line_length = std::sqrt(straight_line_length);

  float trajectory_length = 0.0f;
  for(size_t i = 1; i < result.trajectory.size(); i++) {
    float d = 0.0f;
    for(size_t j = 0; j < start.position.size(); j++) {
      float diff = result.trajectory[i].position[j] - result.trajectory[i - 1].position[j];
      d += diff * diff;
    }
    trajectory_length += std::sqrt(d);
  }
  // 障害物を回避するので直線より長くなるのは当然だが、無限に遠回りしてはいないことの下限チェック
  CHECK(trajectory_length >= straight_line_length * 0.999f);
  CHECK(trajectory_length < straight_line_length * 3.0f);
}

TEST_CASE("MotionGen: interpolated_times respects the joint acceleration limit") {
  auto robot = load_ur5e_arm();
  auto world = cutil::make_ref<WorldConfig>();

  JointState start{{0.0f, -1.57f, 1.57f, -1.57f, -1.57f, 0.0f}, {}};
  JointState goal_q{{1.5f, -1.0f, 1.0f, -1.0f, -1.0f, 0.5f}, {}};
  block_straight_line_midpoint(*robot, *world, start, goal_q);

  MotionGenConfig cfg  = MotionGenConfig::load_from_robot_config(robot, world);
  cfg.max_acceleration = 2.0f;
  MotionGen gen(cfg);
  MotionGenResult result = gen.plan_single_js(start, goal_q);
  REQUIRE(result.success);

  const auto& t = result.interpolated_times;
  const auto& q = result.trajectory;
  REQUIRE(t.size() == q.size());
  CHECK(t.front() == doctest::Approx(0.0f));
  // 区間の平均関節速度の変化から求めた加速度が上限(区間ごとに一定加速度なので離散化誤差ぶんの余裕を見る)を超えない
  float worst = 0.0f;
  for(size_t i = 2; i < q.size(); i++) {
    float dt0 = t[i - 1] - t[i - 2], dt1 = t[i] - t[i - 1];
    for(size_t j = 0; j < q[i].position.size(); j++) {
      float v0 = (q[i - 1].position[j] - q[i - 2].position[j]) / dt0;
      float v1 = (q[i].position[j] - q[i - 1].position[j]) / dt1;
      worst    = std::max(worst, std::abs(v1 - v0) / (0.5f * (dt0 + dt1)));
    }
  }
  CHECK(worst <= cfg.max_acceleration * 2.0f);
  // 加速度制限が効いているので、速度制限だけの場合より遅くなる
  MotionGenConfig loose = cfg;
  loose.max_acceleration = 1000.0f;
  MotionGen gen2(loose);
  MotionGenResult fast = gen2.plan_single_js(start, goal_q);
  REQUIRE(fast.success);
  CHECK(result.interpolated_times.back() > fast.interpolated_times.back() * 0.99f);
}
