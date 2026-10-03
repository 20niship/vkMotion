// Usage: run_batch_benchmark <urdf_path> <json_output_path> -- CPU(sequential) vs GPU(1 batched dispatch) collision-check throughput by batch size.

#include <algorithm>
#include <chrono>
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>
#include <random>
#include <vector>

#include <vkmotion/robot_world.hpp>
#include <vkmotion/urdf_loader.hpp>

using namespace vkmotion;
using json = nlohmann::json;

namespace {

float estimate_reach(const UrdfModel& model) {
  float reach = 0.2f;
  for(const auto& j : model.joints) reach += j.origin_xyz.length();
  return reach;
}

void add_scattered_obstacle(std::mt19937& rng, float reach, int index, WorldConfig& world) {
  std::uniform_real_distribution<float> r_dist(reach * 0.25f, reach * 0.95f);
  std::normal_distribution<float> n(0.0f, 1.0f);
  cutil::Vec3f dir(n(rng), n(rng), n(rng));
  if(dir.length() < 1e-6f) dir = cutil::Vec3f(1.0f, 0.0f, 0.0f);
  cutil::Vec3f center = dir.normalized() * r_dist(rng);

  std::uniform_real_distribution<float> small(reach * 0.05f, reach * 0.18f);
  std::uniform_int_distribution<int> type_pick(0, 3);
  Pose pose{center, cutil::Quatf::identity()};
  std::string name = "obs" + std::to_string(index);
  switch(type_pick(rng)) {
    case 0: {
      Cuboid c;
      c.name = name;
      c.pose = pose;
      c.dims = cutil::Vec3f(small(rng), small(rng), small(rng));
      world.add_obstacle(c);
      break;
    }
    case 1: {
      Sphere s;
      s.name   = name;
      s.pose   = pose;
      s.radius = small(rng);
      world.add_obstacle(s);
      break;
    }
    case 2: {
      Cylinder c;
      c.name   = name;
      c.pose   = pose;
      c.radius = small(rng);
      c.height = small(rng) * 2.0f;
      world.add_obstacle(c);
      break;
    }
    default: {
      Capsule c;
      c.name   = name;
      c.pose   = pose;
      c.radius = small(rng);
      float h  = small(rng) * 2.0f;
      c.base   = cutil::Vec3f(0.0f, 0.0f, -h * 0.5f);
      c.tip    = cutil::Vec3f(0.0f, 0.0f, h * 0.5f);
      world.add_obstacle(c);
      break;
    }
  }
}

std::vector<JointState> random_states(std::mt19937& rng, const RobotConfig& robot, int n) {
  const auto& lo = robot.joint_position_limits_lower();
  const auto& hi = robot.joint_position_limits_upper();
  std::vector<JointState> out(n);
  for(auto& js : out) {
    js.position.resize(robot.dof());
    for(size_t i = 0; i < js.position.size(); i++) {
      float a = lo[i], b = hi[i];
      if(a > b) std::swap(a, b);
      std::uniform_real_distribution<float> d(a, b);
      js.position[i] = d(rng);
    }
  }
  return out;
}

// Runs `fn()` a few times and returns the median wall-clock ms, to smooth out scheduling noise.
template <typename Fn> double median_ms(Fn&& fn, int reps = 3) {
  std::vector<double> samples;
  samples.reserve(reps);
  for(int i = 0; i < reps; i++) {
    auto t0 = std::chrono::steady_clock::now();
    fn();
    auto t1 = std::chrono::steady_clock::now();
    samples.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
  }
  std::sort(samples.begin(), samples.end());
  return samples[samples.size() / 2];
}

}  // namespace

int main(int argc, char** argv) {
  if(argc != 3) {
    std::cerr << "usage: run_batch_benchmark <urdf_path> <json_output_path>\n";
    return 1;
  }
  std::string urdf_path = argv[1];
  std::string json_out   = argv[2];

  auto robot  = RobotConfig::from_file(urdf_path);
  float reach = estimate_reach(robot->urdf_model());

  std::mt19937 rng(12345);
  auto world = cutil::make_ref<WorldConfig>();
  for(int i = 0; i < 12; i++) add_scattered_obstacle(rng, reach, i, *world);

  RobotWorld cpu_rw(RobotWorldConfig{robot, world, false});
  RobotWorld gpu_rw(RobotWorldConfig{robot, world, true});
  gpu_rw.is_collision_free_batch(random_states(rng, *robot, 1));  // warmup: force Vulkan init outside the measured loop

  json root;
  root["robot_urdf"] = urdf_path;
  json results        = json::array();

  for(int batch_size : {1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096}) {
    std::vector<JointState> states = random_states(rng, *robot, batch_size);

    double cpu_ms = median_ms([&] { cpu_rw.is_collision_free_batch(states); });
    double gpu_ms = median_ms([&] { gpu_rw.is_collision_free_batch(states); });

    json r;
    r["batch_size"] = batch_size;
    r["cpu_ms"]      = cpu_ms;
    r["gpu_ms"]      = gpu_ms;
    results.push_back(r);
    std::cerr << "[batch " << batch_size << "] cpu=" << cpu_ms << "ms  gpu=" << gpu_ms << "ms\n";
  }

  root["results"] = results;
  std::ofstream out(json_out);
  if(!out) {
    std::cerr << "failed to open output path: " << json_out << "\n";
    return 1;
  }
  out << root.dump(2);
  std::cerr << "wrote " << json_out << "\n";
  return 0;
}
