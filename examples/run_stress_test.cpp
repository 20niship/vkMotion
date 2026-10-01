// Usage: run_stress_test <urdf_path> <num_trials> <json_output_path> (JSON schema: see README.md)

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>
#include <random>
#include <string>

#include <vkmotion/motion_gen.hpp>
#include <vkmotion/robot_world.hpp>
#include <vkmotion/urdf_loader.hpp>

using namespace vkmotion;
using json = nlohmann::json;

namespace {

constexpr int kObstacleSetupAttempts = 8;

json pose_json(const cutil::Vec3f& pos, const cutil::Quatf& q) {
  return json::array({pos[0], pos[1], pos[2], q.w, q.x, q.y, q.z});
}

// ponytail: sum of joint origin offsets, a crude chain-length reach bound (ignores branching) just to size/place obstacles.
float estimate_reach(const UrdfModel& model) {
  float reach = 0.2f;
  for(const auto& j : model.joints) reach += j.origin_xyz.length();
  return reach;
}

cutil::Vec3f ee_position(const RobotConfig& robot, const JointState& js) {
  std::vector<cutil::Mat4f> link_world = robot.forward_kinematics(js);
  const cutil::Mat4f& t                = link_world.back();
  return cutil::Vec3f(t(0, 3), t(1, 3), t(2, 3));
}

// MotionGen interpolates in JOINT space (a Cartesian-space curve, not a line), so block it using the EE position of an actual interpolated joint state, not a lerp of the Cartesian endpoints.
cutil::Vec3f ee_position_on_joint_lerp(const RobotConfig& robot, const JointState& start, const JointState& goal, float t) {
  JointState mid;
  mid.position.resize(start.position.size());
  for(size_t i = 0; i < mid.position.size(); i++) mid.position[i] = start.position[i] + t * (goal.position[i] - start.position[i]);
  return ee_position(robot, mid);
}

cutil::Quatf random_quaternion(std::mt19937& rng) {
  std::normal_distribution<float> n(0.0f, 1.0f);
  cutil::Vec3f axis(n(rng), n(rng), n(rng));
  if(axis.length() < 1e-6f) axis = cutil::Vec3f(1.0f, 0.0f, 0.0f);
  std::uniform_real_distribution<float> angle(0.0f, 6.28318530718f);
  return cutil::Quatf::from_axis_angle(axis, angle(rng));
}

cutil::Vec3f random_direction(std::mt19937& rng) {
  std::normal_distribution<float> n(0.0f, 1.0f);
  cutil::Vec3f v(n(rng), n(rng), n(rng));
  if(v.length() < 1e-6f) return cutil::Vec3f(1.0f, 0.0f, 0.0f);
  return v.normalized();
}

// Obstacle of random shape/size (scaled off `reach`) centered at `center`.
void add_obstacle_at(std::mt19937& rng, const cutil::Vec3f& center, float reach, int index, WorldConfig& world, json& obstacles_json) {
  std::uniform_int_distribution<int> type_pick(0, 3);
  std::uniform_real_distribution<float> small(reach * 0.05f, reach * 0.18f);
  Pose pose{center, random_quaternion(rng)};
  std::string name = "obs" + std::to_string(index);

  json o;
  o["name"] = name;
  o["pose"] = pose_json(pose.position, pose.quaternion);

  switch(type_pick(rng)) {
    case 0: {
      Cuboid c;
      c.name = name;
      c.pose = pose;
      c.dims = cutil::Vec3f(small(rng), small(rng), small(rng));
      world.add_obstacle(c);
      o["type"] = "cuboid";
      o["dims"] = json::array({c.dims[0], c.dims[1], c.dims[2]});
      break;
    }
    case 1: {
      Sphere s;
      s.name   = name;
      s.pose   = pose;
      s.radius = small(rng);
      world.add_obstacle(s);
      o["type"]   = "sphere";
      o["radius"] = s.radius;
      break;
    }
    case 2: {
      Cylinder c;
      c.name   = name;
      c.pose   = pose;
      c.radius = small(rng);
      c.height = small(rng) * 2.0f;
      world.add_obstacle(c);
      o["type"]   = "cylinder";
      o["radius"] = c.radius;
      o["height"] = c.height;
      break;
    }
    default: {
      Capsule c;
      c.name    = name;
      c.pose    = pose;
      c.radius  = small(rng);
      float h   = small(rng) * 2.0f;
      c.base    = cutil::Vec3f(0.0f, 0.0f, -h * 0.5f);
      c.tip     = cutil::Vec3f(0.0f, 0.0f, h * 0.5f);
      world.add_obstacle(c);
      o["type"]   = "capsule";
      o["radius"] = c.radius;
      o["base"]   = json::array({c.base[0], c.base[1], c.base[2]});
      o["tip"]    = json::array({c.tip[0], c.tip[1], c.tip[2]});
      break;
    }
  }
  obstacles_json.push_back(o);
}

// Center sits on the robot's actual joint-space-interpolated path (+small jitter) so it must pass near it.
void add_blocking_obstacle(std::mt19937& rng, const RobotConfig& robot, const JointState& start, const JointState& goal, float reach, int index,
                            WorldConfig& world, json& obstacles_json) {
  std::uniform_real_distribution<float> t_dist(0.25f, 0.75f);
  std::uniform_real_distribution<float> jitter_dist(0.0f, reach * 0.015f);
  cutil::Vec3f on_path = ee_position_on_joint_lerp(robot, start, goal, t_dist(rng));
  cutil::Vec3f center  = on_path + random_direction(rng) * jitter_dist(rng);
  add_obstacle_at(rng, center, reach, index, world, obstacles_json);
}

// Center is sampled within the robot's reachable shell so obstacles aren't placed somewhere the arm can never reach.
void add_scattered_obstacle(std::mt19937& rng, float reach, int index, WorldConfig& world, json& obstacles_json) {
  std::uniform_real_distribution<float> r_dist(reach * 0.25f, reach * 0.95f);
  cutil::Vec3f center = random_direction(rng) * r_dist(rng);
  add_obstacle_at(rng, center, reach, index, world, obstacles_json);
}

std::vector<float> random_joint_values(std::mt19937& rng, const RobotConfig& robot) {
  const auto& lo = robot.joint_position_limits_lower();
  const auto& hi = robot.joint_position_limits_upper();
  std::vector<float> q(robot.dof());
  for(size_t i = 0; i < q.size(); i++) {
    float a = lo[i], b = hi[i];
    if(a > b) std::swap(a, b);
    std::uniform_real_distribution<float> d(a, b);
    q[i] = d(rng);
  }
  return q;
}

bool find_collision_free_state(std::mt19937& rng, const RobotConfig& robot, const RobotWorld& rw, JointState& out, int max_tries = 200) {
  for(int i = 0; i < max_tries; i++) {
    JointState js{random_joint_values(rng, robot), {}};
    if(rw.is_collision_free(js)) {
      out = js;
      return true;
    }
  }
  return false;
}

json run_motion_gen(MotionGen& gen, const JointState& start, const JointState& goal) {
  auto t0                = std::chrono::steady_clock::now();
  MotionGenResult result = gen.plan_single_js(start, goal);
  auto t1                = std::chrono::steady_clock::now();
  double ms              = std::chrono::duration<double, std::milli>(t1 - t0).count();

  json r;
  r["success"]    = result.success;
  r["elapsed_ms"] = ms;
  r["status"]     = result.status;
  json traj       = json::array();
  for(const auto& js : result.trajectory) traj.push_back(js.position);
  r["trajectory"] = traj;
  return r;
}

}  // namespace

int main(int argc, char** argv) {
  if(argc != 4) {
    std::cerr << "usage: run_stress_test <urdf_path> <num_trials> <json_output_path>\n";
    return 1;
  }
  std::string urdf_path = argv[1];
  int num_trials         = std::max(1, std::atoi(argv[2]));
  std::string json_out   = argv[3];

  auto robot  = RobotConfig::from_file(urdf_path);
  float reach = estimate_reach(robot->urdf_model());

  // WorldConfig/RobotWorld/MotionGenは全試行で使い回し、warmup()後は純粋な経路探索時間だけを計測する。
  auto world = cutil::make_ref<WorldConfig>();
  RobotWorld rw(RobotWorldConfig{robot, world, false});

  MotionGenConfig cpu_cfg = MotionGenConfig::load_from_robot_config(robot, world);
  cpu_cfg.use_vulkan       = false;
  MotionGen cpu_gen(cpu_cfg);
  cpu_gen.warmup();

  MotionGenConfig gpu_cfg = MotionGenConfig::load_from_robot_config(robot, world);
  gpu_cfg.use_vulkan       = true;
  MotionGen gpu_gen(gpu_cfg);
  gpu_gen.warmup();

  json root;
  root["robot_urdf"] = urdf_path;
  root["num_trials"] = num_trials;
  json trials         = json::array();

  std::random_device rd;
  std::uniform_int_distribution<int> n_obs_dist(5, 10);
  std::uniform_int_distribution<int> n_blocker_dist(2, 3);

  for(int t = 0; t < num_trials; t++) {
    unsigned seed = rd();
    std::mt19937 rng(seed);

    json trial;
    trial["trial_index"] = t;
    trial["seed"]         = seed;

    world->clear();
    JointState start, goal;
    bool start_goal_ok = find_collision_free_state(rng, *robot, rw, start) && find_collision_free_state(rng, *robot, rw, goal);
    if(!start_goal_ok) {
      trial["setup_failed"] = true;
      trials.push_back(trial);
      std::cerr << "[trial " << t << "] setup failed (no collision-free start/goal found)\n";
      continue;
    }

    json obstacles_json;
    bool obstacles_ok = false;
    for(int attempt = 0; attempt < kObstacleSetupAttempts && !obstacles_ok; attempt++) {
      world->clear();
      obstacles_json  = json::array();
      int n_obs       = n_obs_dist(rng);
      int n_blockers  = std::min(n_obs, n_blocker_dist(rng));
      for(int i = 0; i < n_blockers; i++) add_blocking_obstacle(rng, *robot, start, goal, reach, i, *world, obstacles_json);
      for(int i = n_blockers; i < n_obs; i++) add_scattered_obstacle(rng, reach, i, *world, obstacles_json);
      obstacles_ok = rw.is_collision_free(start) && rw.is_collision_free(goal);
    }
    if(!obstacles_ok) {
      trial["setup_failed"] = true;
      trials.push_back(trial);
      std::cerr << "[trial " << t << "] setup failed (obstacles kept blocking start/goal)\n";
      continue;
    }

    trial["obstacles"]          = obstacles_json;
    trial["setup_failed"]      = false;
    trial["start_joint_state"] = start.position;
    trial["goal_joint_state"]  = goal.position;
    trial["cpu"]                = run_motion_gen(cpu_gen, start, goal);
    trial["gpu"]                = run_motion_gen(gpu_gen, start, goal);

    std::cerr << "[trial " << t << "] cpu=" << (trial["cpu"]["success"].get<bool>() ? "ok" : "fail") << " (" << trial["cpu"]["elapsed_ms"].get<double>()
               << "ms)  gpu=" << (trial["gpu"]["success"].get<bool>() ? "ok" : "fail") << " (" << trial["gpu"]["elapsed_ms"].get<double>() << "ms)\n";

    trials.push_back(trial);
  }

  root["trials"] = trials;

  std::ofstream out(json_out);
  if(!out) {
    std::cerr << "failed to open output path: " << json_out << "\n";
    return 1;
  }
  out << root.dump(2);
  std::cerr << "wrote " << json_out << "\n";
  return 0;
}
