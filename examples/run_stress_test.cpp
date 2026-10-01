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

json pose_json(const cutil::Vec3f& pos, const cutil::Quatf& q) {
  return json::array({pos[0], pos[1], pos[2], q.w, q.x, q.y, q.z});
}

// ponytail: sum of joint origin offsets, a crude chain-length reach bound (ignores branching) just to size the obstacle workspace box.
float estimate_reach(const UrdfModel& model) {
  float reach = 0.2f;
  for(const auto& j : model.joints) reach += j.origin_xyz.length();
  return reach;
}

cutil::Quatf random_quaternion(std::mt19937& rng) {
  std::normal_distribution<float> n(0.0f, 1.0f);
  cutil::Vec3f axis(n(rng), n(rng), n(rng));
  if(axis.length() < 1e-6f) axis = cutil::Vec3f(1.0f, 0.0f, 0.0f);
  std::uniform_real_distribution<float> angle(0.0f, 6.28318530718f);
  return cutil::Quatf::from_axis_angle(axis, angle(rng));
}

cutil::Vec3f random_point_in_cube(std::mt19937& rng, float half_extent) {
  std::uniform_real_distribution<float> d(-half_extent, half_extent);
  return cutil::Vec3f(d(rng), d(rng), d(rng));
}

void add_random_obstacle(std::mt19937& rng, float reach, int index, WorldConfig& world, json& obstacles_json) {
  std::uniform_int_distribution<int> type_pick(0, 3);
  std::uniform_real_distribution<float> small(reach * 0.03f, reach * 0.15f);
  Pose pose{random_point_in_cube(rng, reach), random_quaternion(rng)};
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

json run_motion_gen(cutil::Ref<RobotConfig> robot, cutil::Ref<WorldConfig> world, const JointState& start, const JointState& goal, bool use_vulkan) {
  MotionGenConfig cfg = MotionGenConfig::load_from_robot_config(robot, world);
  cfg.use_vulkan       = use_vulkan;
  MotionGen gen(cfg);

  auto t0                    = std::chrono::steady_clock::now();
  MotionGenResult result     = gen.plan_single_js(start, goal);
  auto t1                    = std::chrono::steady_clock::now();
  double ms                  = std::chrono::duration<double, std::milli>(t1 - t0).count();

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

  json root;
  root["robot_urdf"] = urdf_path;
  root["num_trials"] = num_trials;
  json trials         = json::array();

  std::random_device rd;
  std::uniform_int_distribution<int> n_obs_dist(5, 10);

  for(int t = 0; t < num_trials; t++) {
    unsigned seed = rd();
    std::mt19937 rng(seed);

    json trial;
    trial["trial_index"] = t;
    trial["seed"]         = seed;

    auto world          = cutil::make_ref<WorldConfig>();
    json obstacles_json = json::array();
    int n_obs           = n_obs_dist(rng);
    for(int i = 0; i < n_obs; i++) add_random_obstacle(rng, reach, i, *world, obstacles_json);
    trial["obstacles"] = obstacles_json;

    RobotWorld rw(RobotWorldConfig{robot, world, false});

    JointState start, goal;
    bool ok = find_collision_free_state(rng, *robot, rw, start) && find_collision_free_state(rng, *robot, rw, goal);
    if(!ok) {
      trial["setup_failed"] = true;
      trials.push_back(trial);
      std::cerr << "[trial " << t << "] setup failed (no collision-free start/goal found)\n";
      continue;
    }

    trial["setup_failed"]      = false;
    trial["start_joint_state"] = start.position;
    trial["goal_joint_state"]  = goal.position;
    trial["cpu"]                = run_motion_gen(robot, world, start, goal, false);
    trial["gpu"]                = run_motion_gen(robot, world, start, goal, true);

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
