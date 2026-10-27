#include <cmath>
#include <doctest/doctest.h>
#include <random>
#include <vkmotion/fast_collision.hpp>
#include <vkmotion/fast_kinematics.hpp>
#include <vkmotion/kinematics.hpp>
#include <vkmotion/robot_world.hpp>

using namespace vkmotion;

namespace {

std::string urdf(const char* name) { return std::string(VKMOTION_TEST_ASSET_DIR) + "/urdf/" + name; }

std::vector<float> random_q(std::mt19937& rng, const RobotConfig& robot) {
  std::vector<float> q(robot.dof());
  for(size_t i = 0; i < q.size(); i++) {
    std::uniform_real_distribution<float> d(robot.joint_position_limits_lower()[i], robot.joint_position_limits_upper()[i]);
    q[i] = d(rng);
  }
  return q;
}

cutil::Ref<WorldConfig> random_world(std::mt19937& rng, int n) {
  auto world = cutil::make_ref<WorldConfig>();
  std::uniform_real_distribution<float> pos(-0.6f, 0.6f), size(0.05f, 0.3f), ang(0.0f, 6.28f);
  std::normal_distribution<float> nrm(0.0f, 1.0f);
  for(int i = 0; i < n; i++) {
    cutil::Quatf q = cutil::Quatf::from_axis_angle(cutil::Vec3f(nrm(rng), nrm(rng), nrm(rng) + 0.1f), ang(rng));
    Pose pose{cutil::Vec3f(pos(rng), pos(rng), pos(rng) + 0.3f), q};
    switch(i % 4) {
      case 0: {
        Cuboid c;
        c.name = "o" + std::to_string(i);
        c.pose = pose;
        c.dims = cutil::Vec3f(size(rng), size(rng), size(rng));
        world->add_obstacle(c);
        break;
      }
      case 1: {
        Sphere s;
        s.name   = "o" + std::to_string(i);
        s.pose   = pose;
        s.radius = size(rng) * 0.5f;
        world->add_obstacle(s);
        break;
      }
      case 2: {
        Cylinder c;
        c.name   = "o" + std::to_string(i);
        c.pose   = pose;
        c.radius = size(rng) * 0.5f;
        c.height = size(rng);
        world->add_obstacle(c);
        break;
      }
      default: {
        Capsule c;
        c.name   = "o" + std::to_string(i);
        c.pose   = pose;
        c.radius = size(rng) * 0.4f;
        c.base   = cutil::Vec3f(0, 0, -size(rng));
        c.tip    = cutil::Vec3f(0, 0, size(rng));
        world->add_obstacle(c);
      }
    }
  }
  return world;
}

}  // namespace

TEST_CASE("CompiledKinematics: matches reference forward_kinematics") {
  for(const char* name : {"ur5e.urdf", "openarm.urdf", "so101.urdf", "two_link_arm.urdf"}) {
    auto robot = RobotConfig::from_file(urdf(name));
    CompiledKinematics kin(robot->urdf_model());
    std::mt19937 rng(1);
    std::vector<Frame> frames(kin.num_links());
    for(int t = 0; t < 20; t++) {
      auto q = random_q(rng, *robot);
      kin.forward(q.data(), frames.data());
      auto ref = vkmotion::forward_kinematics(robot->urdf_model(), q);  // 参照実装(hash mapベース)
      for(size_t l = 0; l < ref.size(); l++) {
        for(int i = 0; i < 3; i++) {
          CHECK(frames[l].p[i] == doctest::Approx(ref[l](i, 3)).epsilon(1e-4));
          for(int j = 0; j < 3; j++) CHECK(frames[l].r[i * 3 + j] == doctest::Approx(ref[l](i, j)).epsilon(1e-4));
        }
      }
    }
  }
}

TEST_CASE("CompiledKinematics: Jacobian matches finite differences") {
  auto robot = RobotConfig::from_file(urdf("ur5e.urdf"));
  CompiledKinematics kin(robot->urdf_model());
  const int tip = static_cast<int>(kin.num_links()) - 1;
  std::mt19937 rng(2);
  auto q = random_q(rng, *robot);
  std::vector<Frame> f(kin.num_links()), fp(kin.num_links());
  kin.forward(q.data(), f.data());
  std::vector<float> J(6 * kin.dof());
  kin.jacobian(f.data(), tip, J.data());
  for(size_t j = 0; j < kin.dof(); j++) {
    auto qp = q;
    qp[j] += 1e-3f;
    kin.forward(qp.data(), fp.data());
    for(int i = 0; i < 3; i++) CHECK(J[i * kin.dof() + j] == doctest::Approx((fp[tip].p[i] - f[tip].p[i]) / 1e-3f).epsilon(2e-2).scale(1.0));
  }
}

TEST_CASE("FastCollision: agrees with the reference RobotWorld on random worlds") {
  for(const char* name : {"ur5e.urdf", "openarm.urdf", "so101.urdf"}) {
    auto robot = RobotConfig::from_file(urdf(name));
    std::mt19937 rng(3);
    int compared = 0, collided = 0;
    for(int w = 0; w < 6; w++) {
      auto world = random_world(rng, 8);
      RobotWorld ref(RobotWorldConfig{robot, world, false});
      FastCollision fast(*robot, AllowedCollisionMatrix{});
      REQUIRE(fast.set_world(*world));
      for(int t = 0; t < 300; t++) {
        auto q = random_q(rng, *robot);
        JointState js{q, {}};
        // 境界すれすれ(float誤差で判定が割れる)ケースは除外
        float margin = std::numeric_limits<float>::infinity();
        for(float d : ref.get_collision_distance(js)) margin = std::min(margin, std::abs(d));
        for(float d : ref.get_self_collision_distance(js)) margin = std::min(margin, std::abs(d));
        if(margin < 1e-4f) continue;
        bool r = ref.is_collision_free(js);
        compared++;
        collided += r ? 0 : 1;
        CHECK(fast.is_free(q.data()) == r);
      }
    }
    CHECK(compared > 500);
    CHECK(collided > 50);  // 衝突ケースが十分含まれていなければ一致検証にならない
  }
}

TEST_CASE("FastCollision: ACM and unsupported obstacle types") {
  auto robot = RobotConfig::from_file(urdf("two_link_arm.urdf"));
  auto world = cutil::make_ref<WorldConfig>();
  Cuboid blocker;
  blocker.name          = "blocker";
  blocker.pose.position = cutil::Vec3f(0.3f, 0.0f, 0.3f);
  blocker.dims           = cutil::Vec3f(0.5f, 0.5f, 0.5f);
  world->add_obstacle(blocker);
  float q[2] = {0.0f, 0.0f};

  FastCollision plain(*robot, AllowedCollisionMatrix{});
  REQUIRE(plain.set_world(*world));
  CHECK_FALSE(plain.is_free(q));

  AllowedCollisionMatrix acm;
  acm.disable_collision("link2", "blocker");
  FastCollision allowed(*robot, acm);
  REQUIRE(allowed.set_world(*world));
  CHECK(allowed.is_free(q));

  ConvexMesh hull;
  hull.name = "hull";
  hull.vertices.push_back(cutil::Vec3f(0, 0, 0));
  world->add_obstacle(hull);
  CHECK_FALSE(plain.set_world(*world));
}

TEST_CASE("FastCollision: motion_free rejects a path through an obstacle and accepts a free one") {
  auto robot = RobotConfig::from_file(urdf("ur5e.urdf"));
  std::mt19937 rng(5);
  auto world = random_world(rng, 6);
  FastCollision fast(*robot, AllowedCollisionMatrix{});
  REQUIRE(fast.set_world(*world));

  int checked = 0;
  for(int t = 0; t < 400; t++) {
    auto a = random_q(rng, *robot), b = random_q(rng, *robot);
    if(!fast.is_free(a.data()) || !fast.is_free(b.data())) continue;
    // 同じ分割数で全点を順に検査した結果と一致するか(二分法の検査順は結果に影響しない)
    int n = fast.motion_steps(a.data(), b.data());
    bool all_free = true;
    std::vector<float> q(a.size());
    for(int i = 1; i < n; i++) {
      for(size_t j = 0; j < q.size(); j++) q[j] = a[j] + (b[j] - a[j]) * (static_cast<float>(i) / static_cast<float>(n));
      all_free = all_free && fast.is_free(q.data());
    }
    CHECK(fast.motion_free(a.data(), b.data()) == all_free);
    checked++;
  }
  CHECK(checked > 20);
}

TEST_CASE("FastCollision: SIMD free_mask8 matches the scalar is_free (with and without margin)") {
  for(const char* name : {"ur5e.urdf", "openarm.urdf", "so101.urdf"}) {
    auto robot = RobotConfig::from_file(urdf(name));
    std::mt19937 rng(11);
    int compared = 0, collided = 0;
    for(int w = 0; w < 5; w++) {
      auto world = cutil::make_ref<WorldConfig>();
      std::uniform_real_distribution<float> pos(-0.6f, 0.6f), size(0.05f, 0.3f), ang(0.0f, 6.28f);
      std::normal_distribution<float> nrm(0.0f, 1.0f);
      for(int i = 0; i < 8; i++) {
        Pose pose{cutil::Vec3f(pos(rng), pos(rng), pos(rng) + 0.3f), cutil::Quatf::from_axis_angle(cutil::Vec3f(nrm(rng), nrm(rng), nrm(rng) + 0.1f), ang(rng))};
        if(i % 3 == 2) {
          Sphere s;
          s.name   = "s" + std::to_string(i);
          s.pose   = pose;
          s.radius = size(rng) * 0.5f;
          world->add_obstacle(s);
        } else {
          Cuboid c;
          c.name = "c" + std::to_string(i);
          c.pose = pose;
          c.dims = cutil::Vec3f(size(rng), size(rng), size(rng));
          world->add_obstacle(c);
        }
      }
      FastCollision fc(*robot, AllowedCollisionMatrix{});
      REQUIRE(fc.set_world(*world));
      for(float margin : {0.0f, 0.004f}) {
        for(int t = 0; t < 60; t++) {
          std::vector<float> block;
          std::vector<std::vector<float>> qs;
          for(int lane = 0; lane < 8; lane++) {
            qs.push_back(random_q(rng, *robot));
            block.insert(block.end(), qs.back().begin(), qs.back().end());
          }
          uint32_t m = fc.free_mask8(block.data(), margin);
          for(int lane = 0; lane < 8; lane++) {
            bool scalar = fc.is_free_margin(qs[lane].data(), margin);
            compared++;
            collided += scalar ? 0 : 1;
            CHECK(((m >> lane) & 1u) == (scalar ? 1u : 0u));
          }
        }
      }
    }
    CHECK(compared > 2000);
    CHECK(collided > 200);
  }
}

TEST_CASE("RobotWorld: GPU robot_validate batch agrees with the CPU SIMD batch") {
  int compared_worlds = 0;
  for(const char* name : {"ur5e.urdf", "openarm.urdf"}) {
    auto robot = RobotConfig::from_file(urdf(name));
    std::mt19937 rng(21);
    std::uniform_real_distribution<float> pos(-0.8f, 0.8f), size(0.05f, 0.25f), ang(0.0f, 6.28f);
    std::normal_distribution<float> nrm(0.0f, 1.0f);
    std::vector<JointState> states;
    for(size_t i = 0; i < RobotWorld::kGpuBatchThreshold + 100; i++) states.push_back(JointState{random_q(rng, *robot), {}});

    // 自由空間の割合が適度な世界が引けるまで障害物配置を作り直す(ロボットの根元を塞ぐ配置だと全て衝突になり比較にならない)
    for(int attempt = 0; attempt < 30; attempt++) {
      auto world = cutil::make_ref<WorldConfig>();
      for(int i = 0; i < 4; i++) {
        Pose pose{cutil::Vec3f(pos(rng), pos(rng), pos(rng) + 0.4f), cutil::Quatf::from_axis_angle(cutil::Vec3f(nrm(rng), nrm(rng), nrm(rng) + 0.1f), ang(rng))};
        if(i % 3 == 2) {
          Sphere s;
          s.name   = "s" + std::to_string(i);
          s.pose   = pose;
          s.radius = size(rng) * 0.5f;
          world->add_obstacle(s);
        } else {
          Cuboid c;
          c.name = "c" + std::to_string(i);
          c.pose = pose;
          c.dims = cutil::Vec3f(size(rng), size(rng), size(rng));
          world->add_obstacle(c);
        }
      }
      RobotWorld cpu(RobotWorldConfig{robot, world, false});
      auto b = cpu.is_collision_free_batch(states);
      size_t free_count = 0;
      for(bool v : b) free_count += v;
      if(free_count < 300 || free_count > states.size() - 300) continue;

      RobotWorld gpu(RobotWorldConfig{robot, world, true});
      auto a = gpu.is_collision_free_batch(states);
      size_t diff = 0;
      for(size_t i = 0; i < states.size(); i++) diff += a[i] != b[i];
      CHECK(diff <= 2);  // float演算順の違いで境界すれすれだけ割れうる
      compared_worlds++;
      break;
    }
  }
  CHECK(compared_worlds == 2);
}
