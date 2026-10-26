// Usage: bench_vkmotion <data_dir> <out_json> [--limit N] [--skip fkcol,js,pose,ik] [--backend cpu|vulkan] [--threads N] [--gpu-search]
// vamp / cuRobo と同じ入力(bench/prepare.py の出力)で FK・衝突判定・IK・軌道生成を計測し、同一スキーマのJSONを書く。

#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <map>
#include <nlohmann/json.hpp>
#include <set>
#include <string>
#include <vkmotion/fast_collision.hpp>
#include <vkmotion/fast_kinematics.hpp>
#include <vkmotion/ik_solver.hpp>
#include <vkmotion/motion_gen.hpp>
#include <vkmotion/robot_world.hpp>
#include <vkmotion/vk_context.hpp>
#include <vkmotion/vk_validate.hpp>

using namespace vkmotion;
using json = nlohmann::json;

namespace {

using Clock = std::chrono::steady_clock;
double ms_since(Clock::time_point t0) { return std::chrono::duration<double, std::milli>(Clock::now() - t0).count(); }

json read_json(const std::string& path) {
  std::ifstream f(path);
  if(!f) throw std::runtime_error("cannot open " + path);
  return json::parse(f);
}

void fill_world(WorldConfig& world, const json& objects) {
  world.clear();
  for(const auto& o : objects) {
    Pose pose = Pose::from_list(o["pose"].get<std::vector<float>>());
    if(o["type"] == "cuboid") {
      Cuboid c;
      c.name = o["name"];
      c.pose = pose;
      auto d = o["dims"].get<std::vector<float>>();
      c.dims = cutil::Vec3f(d[0], d[1], d[2]);
      world.add_obstacle(c);
    } else if(o["type"] == "sphere") {
      Sphere s;
      s.name   = o["name"];
      s.pose   = pose;
      s.radius = o["radius"];
      world.add_obstacle(s);
    } else {
      throw std::runtime_error("unsupported obstacle type in bench: " + o["type"].get<std::string>());
    }
  }
}

AllowedCollisionMatrix load_acm(const std::string& path) {
  AllowedCollisionMatrix acm;
  for(const auto& p : read_json(path)) acm.disable_collision(p[0], p[1]);
  return acm;
}

JointState js_of(const json& q) { return JointState{q.get<std::vector<float>>(), {}}; }

// 末端リンク(link_names().back() = IKSolverと同じ規約)のFK
Pose ee_pose(const RobotConfig& robot, const std::vector<float>& q) {
  std::vector<cutil::Mat4f> link_world = robot.forward_kinematics(JointState{q, {}});
  const cutil::Mat4f& t                = link_world.back();
  Pose p;
  p.position = cutil::Vec3f(t(0, 3), t(1, 3), t(2, 3));
  cutil::Mat<3, 3, float> r;
  for(unsigned i = 0; i < 3; i++)
    for(unsigned j = 0; j < 3; j++) r(i, j) = t(i, j);
  p.quaternion = cutil::Quatf::from_mat3(r);
  return p;
}

double rot_err_deg(const cutil::Quatf& a, const cutil::Quatf& b) {
  double d = std::abs(double(a.w) * b.w + double(a.x) * b.x + double(a.y) * b.y + double(a.z) * b.z);
  return 2.0 * std::acos(std::min(1.0, d)) * 180.0 / M_PI;
}

double path_len(const std::vector<JointState>& traj) {
  double len = 0;
  for(size_t i = 1; i < traj.size(); i++) {
    double s = 0;
    for(size_t j = 0; j < traj[i].position.size(); j++) {
      double d = traj[i].position[j] - traj[i - 1].position[j];
      s += d * d;
    }
    len += std::sqrt(s);
  }
  return len;
}

json bench_fk_collision(const json& samples, const cutil::Ref<RobotConfig>& robot, const AllowedCollisionMatrix& acm, bool use_vulkan, bool want_fk) {
  json out;
  const auto& qs = samples["q"];
  std::vector<JointState> states;
  for(const auto& q : qs) states.push_back(js_of(q));

  if(want_fk) {
    // 精度: 公開API(RobotConfig::forward_kinematics)の末端位置 / 速度: コンパイル済みFK(全リンク姿勢, 1スレッド)
    json fk = json::array();
    for(const auto& s : states) {
      Pose p = ee_pose(*robot, s.position);
      fk.push_back({p.position[0], p.position[1], p.position[2]});
    }
    const CompiledKinematics& kin = robot->compiled_kinematics();
    std::vector<Frame> frames(kin.num_links());
    std::vector<float> qflat;
    for(const auto& s : states) qflat.insert(qflat.end(), s.position.begin(), s.position.end());
    volatile float sink = 0;
    const int reps      = 50;
    auto t0             = Clock::now();
    for(int r = 0; r < reps; r++) {
      for(size_t i = 0; i < states.size(); i++) {
        kin.forward(qflat.data() + i * kin.dof(), frames.data());
        sink = sink + frames.back().p[0];
      }
    }
    out["fk_us_per_call"] = ms_since(t0) * 1e3 / (double(reps) * states.size());
    out["fk"]             = fk;
  }

  out["collision"] = json::object();
  for(auto& [name, sc] : samples["scenes"].items()) {
    auto world = cutil::make_ref<WorldConfig>();
    fill_world(*world, sc["objects"]);
    RobotWorld rw(RobotWorldConfig{robot, world, use_vulkan, acm});
    rw.warmup();  // Vulkanパイプライン作成・世界のコンパイルなど初回コストを計測から外す

    // 精度用: 単発APIで全サンプルを判定 / 速度用: バッチAPI
    auto t0                  = Clock::now();
    std::vector<bool> valid  = rw.is_collision_free_batch(states);
    double batch_ms          = ms_since(t0);
    // 大バッチスループット(100k configs相当を繰り返しで作る)
    std::vector<JointState> big;
    const size_t big_n = 100000;
    big.reserve(big_n);
    for(size_t i = 0; i < big_n; i++) big.push_back(states[i % states.size()]);
    t0 = Clock::now();
    rw.is_collision_free_batch(big);
    double big_ms = ms_since(t0);
    json v        = json::array();
    for(bool b : valid) v.push_back(b ? 1 : 0);
    // 1スレッドで1configずつ(計画ループの実態に近いレイテンシ)
    t0 = Clock::now();
    size_t n_single = std::min<size_t>(big.size(), 20000);
    size_t n_ok     = 0;
    for(size_t i = 0; i < n_single; i++) n_ok += rw.is_collision_free(big[i]) ? 1 : 0;
    double single_ms = ms_since(t0);
    out["collision"][name] = {{"us_per_check", big_ms * 1e3 / big.size()}, {"us_per_check_1thread", single_ms * 1e3 / n_single},
                              {"us_per_check_2k_batch", batch_ms * 1e3 / states.size()}, {"valid", v}, {"n_ok_dummy", n_ok}};
  }
  return out;
}

json plan_row(const json& p) { return {{"scene", p["scene"]}, {"index", p["index"]}, {"valid", true}}; }

json bench_plan(const json& problems, const cutil::Ref<RobotConfig>& robot, const AllowedCollisionMatrix& acm, bool use_vulkan, bool pose_mode, int threads, float margin) {
  auto world = cutil::make_ref<WorldConfig>();
  MotionGenConfig cfg = MotionGenConfig::load_from_robot_config(robot, world);
  cfg.use_vulkan      = use_vulkan;
  cfg.acm             = acm;
  cfg.num_threads     = threads;
  cfg.collision_margin = margin;
  if(const char* e = std::getenv("BENCH_IK_GOALS")) cfg.num_ik_goals = std::atoi(e);
  cfg.seed            = 1234;
  cfg.gpu_search      = std::getenv("BENCH_GPU_SEARCH") != nullptr;
  if(const char* e = std::getenv("BENCH_HANDOFF")) cfg.gpu_handoff_s = std::atof(e);
  if(const char* e = std::getenv("BENCH_SEARCH_STEP")) cfg.search_step = std::atof(e);
  if(const char* e = std::getenv("BENCH_COLLISION_STEP")) cfg.collision_step = std::atof(e);
  MotionGen mg(cfg);
  mg.warmup();

  json rows = json::array();
  for(const auto& p : problems) {
    fill_world(*world, p["objects"]);
    json row = plan_row(p);
    JointState start = js_of(p["start"]);
    MotionGenPlanConfig plan_cfg;
    const uint64_t checks0 = FastCollision::debug_check_count(), blocks0 = FastCollision::debug_block_count(), edges0 = FastCollision::debug_edge_count();
    auto t0 = Clock::now();
    MotionGenResult res;
    Pose goal_pose;
    if(pose_mode) {
      goal_pose = Pose::from_list(p["goal_pose"].get<std::vector<float>>());
      res       = mg.plan_single(start, goal_pose, plan_cfg);
    } else {
      res = mg.plan_single_js(start, js_of(p["goal"]), plan_cfg);
    }
    double dt = ms_since(t0);
    row["success"] = res.success;
    row["time_ms"] = dt;
    row["checks"]  = FastCollision::debug_check_count() - checks0;
    row["blocks"]  = FastCollision::debug_block_count() - blocks0;
    row["edges"]   = FastCollision::debug_edge_count() - edges0;
    if(res.success) {
      row["path_len"]    = path_len(res.trajectory);
      row["n_waypoints"] = res.trajectory.size();
      json path          = json::array();
      for(const auto& s : res.trajectory) path.push_back(s.position);
      row["path"] = path;
      if(!res.interpolated_times.empty()) {
        row["motion_time_s"] = res.interpolated_times.back();
        row["times"]         = res.interpolated_times;
      }
      if(pose_mode) {
        Pose reached        = ee_pose(*robot, res.trajectory.back().position);
        row["pos_err_mm"]   = (reached.position - goal_pose.position).length() * 1e3;
        row["rot_err_deg"]  = rot_err_deg(reached.quaternion, goal_pose.quaternion);
      }
    }
    rows.push_back(row);
  }
  return rows;
}

json bench_ik(const json& problems, const cutil::Ref<RobotConfig>& robot, const AllowedCollisionMatrix& acm) {
  auto world = cutil::make_ref<WorldConfig>();
  IKSolverConfig cfg;
  cfg.robot_config = robot;
  cfg.world_config = world;
  cfg.acm          = acm;
  cfg.seed         = 1234;
  if(const char* e = std::getenv("BENCH_IK_SEEDS")) cfg.num_seeds = std::atoi(e);
  json rows        = json::array();
  for(const auto& p : problems) {
    fill_world(*world, p["objects"]);
    Pose goal = Pose::from_list(p["goal_pose"].get<std::vector<float>>());
    IKSolver ik(cfg);
    auto t0 = Clock::now();
    auto r  = ik.solve_batch({goal});
    double dt = ms_since(t0);
    json row  = plan_row(p);
    row["time_ms"] = dt;
    row["success"] = !r.empty() && r[0].has_value();
    if(row["success"]) {
      Pose reached       = ee_pose(*robot, r[0]->position);
      row["pos_err_mm"]  = (reached.position - goal.position).length() * 1e3;
      row["rot_err_deg"] = rot_err_deg(reached.quaternion, goal.quaternion);
    }
    rows.push_back(row);
  }
  return rows;
}

}  // namespace

// デバッグ用: [{"scene","index","q":[...]}...] を読み、各configの判定(fast path / 参照実装)を書き出す。
int check_mode(const std::string& data, const std::string& in_path, const std::string& out_path) {
  auto robot = RobotConfig::from_file(data + "/panda.urdf");
  auto acm   = load_acm(data + "/acm.json");
  std::map<std::pair<std::string, int>, json> scenes;
  json all = read_json(data + "/problems.json");
  for(const auto& p : all["problems"]) scenes[{p["scene"].get<std::string>(), p["index"].get<int>()}] = p["objects"];
  json out = json::array();
  json checks = read_json(in_path);
  for(const auto& c : checks) {
    auto world = cutil::make_ref<WorldConfig>();
    fill_world(*world, scenes.at({c["scene"].get<std::string>(), c["index"].get<int>()}));
    RobotWorld rw(RobotWorldConfig{robot, world, false, acm});
    JointState js = js_of(c["q"]);
    bool ref      = true;
    float min_d   = 1e9f;
    for(float d : rw.get_collision_distance(js)) { ref = ref && d > 0; min_d = std::min(min_d, d); }
    float min_self = 1e9f;
    for(float d : rw.get_self_collision_distance(js)) { ref = ref && d > 0; min_self = std::min(min_self, d); }
    out.push_back({{"fast", rw.is_collision_free(js)}, {"ref", ref}, {"min_world", min_d}, {"min_self", min_self}});
  }
  std::ofstream(out_path) << out.dump();
  return 0;
}

// GPU衝突判定1回のレイテンシ/スループットの切り分け用: n個のコンフィグを1 dispatchで判定する時間をnを変えて計る。
int micro_mode(const std::string& data) {
  auto robot = RobotConfig::from_file(data + "/panda.urdf");
  auto acm   = load_acm(data + "/acm.json");
  json problems = read_json(data + "/problems.json")["problems"];
  json samples  = read_json(data + "/samples.json");
  for(const char* scene : {"bookshelf_thin", "cage"}) {
    auto world = cutil::make_ref<WorldConfig>();
    for(const auto& p : problems)
      if(p["scene"] == scene) { fill_world(*world, p["objects"]); break; }
    FastCollision fc(*robot, acm);
    fc.set_world(*world);
    GpuRobotData rd;
    GpuWorldData wd;
    fc.pack_gpu_robot(rd);
    fc.pack_gpu_world(wd);
    VkContext ctx;
    VkValidatePipeline vp(ctx, rd);
    vp.set_world(wd);
    const auto& qs = samples["q"];
    const size_t dof = fc.dof();
    for(size_t n : {32, 64, 1792, 3584, 7168, 14336, 28672, 100000}) {
      std::vector<float> q(n * dof);
      for(size_t i = 0; i < n; i++)
        for(size_t j = 0; j < dof; j++) q[i * dof + j] = qs[i % qs.size()][j];
      std::vector<uint8_t> out(n);
      vp.validate(q.data(), n, 0.0f, out.data());
      std::vector<double> t;
      for(int r = 0; r < 30; r++) {
        auto t0 = Clock::now();
        vp.validate(q.data(), n, 0.0f, out.data());
        t.push_back(ms_since(t0));
      }
      std::sort(t.begin(), t.end());
      std::cout << scene << " n=" << n << " median " << t[t.size() / 2] * 1e3 << " us  (" << t[t.size() / 2] * 1e3 / n << " us/check)\n";
    }
  }
  return 0;
}

int main(int argc, char** argv) {
  if(argc >= 3 && std::string(argv[2]) == "--micro") return micro_mode(argv[1]);
  if(argc >= 5 && std::string(argv[3]) == "--check") return check_mode(argv[1], argv[4], argv[2]);
  if(argc < 3) {
    std::cerr << "usage: bench_vkmotion <data_dir> <out_json> [--limit N] [--skip a,b] [--backend cpu|vulkan]\n";
    return 2;
  }
  std::string data = argv[1], out_path = argv[2];
  int limit = 0;
  std::set<std::string> skip;
  std::string backend = "vulkan";
  int threads         = 1;
  float margin        = 0.005f;
  std::string only;  // "scene:index"
  for(int i = 3; i < argc; i++) {
    std::string a = argv[i];
    if(a == "--limit" && i + 1 < argc) limit = std::stoi(argv[++i]);
    else if(a == "--backend" && i + 1 < argc) backend = argv[++i];
    else if(a == "--threads" && i + 1 < argc) threads = std::stoi(argv[++i]);
    else if(a == "--margin" && i + 1 < argc) margin = std::stof(argv[++i]);
    else if(a == "--only" && i + 1 < argc) only = argv[++i];
    else if(a == "--gpu-search") setenv("BENCH_GPU_SEARCH", "1", 1);  // 大域探索(RRT-Connect)をGPUで行う(MotionGenConfig::gpu_search)
    else if(a == "--skip" && i + 1 < argc) {
      std::string s = argv[++i], tok;
      for(char c : s + ",") {
        if(c == ',') { skip.insert(tok); tok.clear(); } else tok += c;
      }
    }
  }
  const bool vulkan = backend == "vulkan";

  auto robot = RobotConfig::from_file(data + "/panda.urdf");
  auto acm   = load_acm(data + "/acm.json");

  json problems_all = read_json(data + "/problems.json")["problems"];
  json problems     = json::array();
  std::map<std::string, int> per_scene;
  for(const auto& p : problems_all) {
    if(!only.empty() && only != p["scene"].get<std::string>() + ":" + std::to_string(p["index"].get<int>())) continue;
    if(limit > 0 && ++per_scene[p["scene"]] > limit) continue;
    problems.push_back(p);
  }

  json res;
  res["lib"]    = "vkmotion-" + backend;
  res["planner_threads"] = threads;
  res["device"] = vulkan ? "Vulkan compute (GPU) + CPU" : "CPU (reference, scalar)";
  if(!skip.count("fkcol")) {
    json fc = bench_fk_collision(read_json(data + "/samples.json"), robot, acm, vulkan, true);
    for(auto& [k, v] : fc.items()) res[k] = v;
  }
  if(!skip.count("js")) res["plan_js"] = bench_plan(problems, robot, acm, vulkan, false, threads, margin);
  if(!skip.count("pose")) res["plan_pose"] = bench_plan(problems, robot, acm, vulkan, true, threads, margin);
  if(!skip.count("ik")) res["ik"] = bench_ik(problems, robot, acm);

  std::ofstream(out_path) << res.dump();
  std::cout << "saved " << out_path << "\n";
  return 0;
}
