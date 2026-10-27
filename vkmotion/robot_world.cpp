#include "robot_world.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <thread>

#include "fast_collision.hpp"
#include "vk_rrt.hpp"
#include "sphere_collision.hpp"
#include "vk_collision.hpp"
#include "vk_context.hpp"
#include "vk_repair.hpp"
#include "vk_validate.hpp"

namespace vkmotion {

namespace {

cutil::Vec3f transform_point(const cutil::Mat4f& t, const cutil::Vec3f& p) {
  return cutil::Vec3f(t(0, 0) * p[0] + t(0, 1) * p[1] + t(0, 2) * p[2] + t(0, 3), t(1, 0) * p[0] + t(1, 1) * p[1] + t(1, 2) * p[2] + t(1, 3),
                       t(2, 0) * p[0] + t(2, 1) * p[1] + t(2, 2) * p[2] + t(2, 3));
}

cutil::Mat4f link_world_matrix(const RobotConfig& robot, const JointState& js, const std::string& link_name) {
  std::vector<cutil::Mat4f> link_world = robot.forward_kinematics(js);
  const auto& names                     = robot.link_names();
  for(size_t i = 0; i < names.size(); i++)
    if(names[i] == link_name) return link_world[i];
  return cutil::Mat4f::identity();
}

// link_name付きの普通のCollisionSphereとして追加するだけで既存のsphere-vs-obstacle/self-collision除外ロジックに乗る(CPU・GPU(get_collision_distance系)ともシェーダ変更不要)。GPU常駐repairはロボット構造を構築時に1回だけアップロードする設計のため非対応。
std::vector<CollisionSphere> spheres_with_attached(const RobotConfig& robot, const WorldConfig& world, const JointState& js) {
  std::vector<CollisionSphere> spheres = robot.collision_spheres_world(js);
  for(const auto& att : world.attached_objects()) {
    cutil::Mat4f t = link_world_matrix(robot, js, att.link_name);
    spheres.push_back(CollisionSphere{att.link_name, transform_point(t, att.center), att.radius});
  }
  return spheres;
}

}  // namespace

RobotWorld::RobotWorld(const RobotWorldConfig& config) : config_(config) {
  for(const auto& joint : config_.robot_config->urdf_model().joints) {
    adjacent_links_.emplace_back(joint.parent_link, joint.child_link);
  }
}

RobotWorld::~RobotWorld() = default;

bool RobotWorld::is_adjacent(const std::string& a, const std::string& b) const {
  for(const auto& pair : adjacent_links_) {
    if((pair.first == a && pair.second == b) || (pair.first == b && pair.second == a)) return true;
  }
  return false;
}

VkCollisionPipeline& RobotWorld::vulkan_pipeline() const {
  if(!vk_ctx_) vk_ctx_ = std::make_unique<VkContext>();
  if(!vk_collision_) vk_collision_ = std::make_unique<VkCollisionPipeline>(*vk_ctx_);
  return *vk_collision_;
}

VkRepairPipeline& RobotWorld::vulkan_repair_pipeline() const {
  if(!vk_ctx_) vk_ctx_ = std::make_unique<VkContext>();
  if(!vk_repair_) vk_repair_ = std::make_unique<VkRepairPipeline>(*vk_ctx_, *config_.robot_config);
  return *vk_repair_;
}

std::vector<float> RobotWorld::get_collision_distance(const JointState& joint_state) const {
  std::vector<CollisionSphere> spheres = spheres_with_attached(*config_.robot_config, *config_.world_config, joint_state);
  const WorldConfig& world             = *config_.world_config;

  if(config_.use_vulkan) return vulkan_pipeline().compute(spheres, world);

  std::vector<float> distances;
  distances.reserve(spheres.size());
  for(const auto& s : spheres) {
    float min_dist = std::numeric_limits<float>::infinity();
    for(const auto& o : world.cuboid()) {
      if(config_.acm.is_allowed(s.link_name, o.name)) continue;
      min_dist = std::min(min_dist, distance_sphere_cuboid(s.center, s.radius, o));
    }
    for(const auto& o : world.sphere()) {
      if(config_.acm.is_allowed(s.link_name, o.name)) continue;
      min_dist = std::min(min_dist, distance_sphere_sphere(s.center, s.radius, o));
    }
    for(const auto& o : world.cylinder()) {
      if(config_.acm.is_allowed(s.link_name, o.name)) continue;
      min_dist = std::min(min_dist, distance_sphere_cylinder(s.center, s.radius, o));
    }
    for(const auto& o : world.capsule()) {
      if(config_.acm.is_allowed(s.link_name, o.name)) continue;
      min_dist = std::min(min_dist, distance_sphere_capsule(s.center, s.radius, o));
    }
    for(const auto& o : world.plane()) {
      if(config_.acm.is_allowed(s.link_name, o.name)) continue;
      min_dist = std::min(min_dist, distance_sphere_plane(s.center, s.radius, o));
    }
    for(const auto& o : world.convex_mesh()) {
      if(config_.acm.is_allowed(s.link_name, o.name)) continue;
      min_dist = std::min(min_dist, distance_sphere_convex(s.center, s.radius, o));
    }
    for(const auto& o : world.sdf_grid()) {
      if(config_.acm.is_allowed(s.link_name, o.name)) continue;
      min_dist = std::min(min_dist, distance_sphere_sdf(s.center, s.radius, o));
    }
    distances.push_back(min_dist);
  }
  return distances;
}

std::vector<float> RobotWorld::get_self_collision_distance(const JointState& joint_state) const {
  std::vector<CollisionSphere> spheres = spheres_with_attached(*config_.robot_config, *config_.world_config, joint_state);

  std::vector<float> distances;
  for(size_t i = 0; i < spheres.size(); i++) {
    for(size_t j = i + 1; j < spheres.size(); j++) {
      const auto& a = spheres[i];
      const auto& b = spheres[j];
      if(a.link_name == b.link_name) continue;
      if(is_adjacent(a.link_name, b.link_name)) continue;
      if(config_.acm.is_allowed(a.link_name, b.link_name)) continue;
      float d = static_cast<float>((a.center - b.center).length()) - a.radius - b.radius;
      distances.push_back(d);
    }
  }
  return distances;
}

void RobotWorld::warmup() const {
  const FastCollision* fc = fast_collision();
  if(fc && config_.use_vulkan && fc->gpu_compatible()) {
    // 閾値ちょうどのバッチを1回流してパイプライン作成と世界のアップロードまで済ませる
    std::vector<JointState> dummy(kGpuBatchThreshold, JointState{std::vector<float>(config_.robot_config->dof(), 0.0f), {}});
    is_collision_free_batch(dummy);
  }
  if(gpu_rrt_available()) {
    // GPU探索のパイプライン(ロボット専用シェーダのコンパイルを含む)と世界のアップロードまで済ませる。解けるかどうかは問わない。
    GpuRrtSettings gs;
    gs.max_submits = 1;
    gs.rounds_per_submit = 2;
    std::vector<std::vector<float>> path;
    const size_t dof = config_.robot_config->dof();
    gpu_rrt_connect(std::vector<float>(dof, 0.0f), {std::vector<float>(dof, 0.1f)}, gs, path);
  }
}

bool RobotWorld::gpu_rrt_batch(const std::vector<GpuRrtProblem>& problems, const GpuRrtSettings& settings, std::vector<GpuRrtResult>& results) const {
  if(!gpu_rrt_available()) return false;
  const FastCollision* fc = fast_collision();
  try {
    std::lock_guard<std::mutex> lock(fast_mutex_);
    if(!vk_rrt_) {
      GpuRobotData rd;
      if(!fc->pack_gpu_robot(rd)) throw std::runtime_error("robot exceeds GPU limits");
      if(!vk_ctx_) vk_ctx_ = std::make_unique<VkContext>();
      vk_rrt_         = std::make_unique<VkRrtPipeline>(*vk_ctx_, rd, fc->joint_reach(), config_.robot_config->joint_position_limits_lower(), config_.robot_config->joint_position_limits_upper());
      vk_rrt_version_ = ~uint64_t{0};
    }
    vk_rrt_->plan_batch(problems, settings, results);
    return true;
  } catch(const std::exception&) {
    vk_rrt_failed_ = true;
    return false;
  }
}

bool RobotWorld::gpu_rrt_available() const {
  if(!config_.use_vulkan || vk_rrt_failed_) return false;
  const FastCollision* fc = fast_collision();
  return fc && fc->gpu_compatible() && fc->dof() <= VkRrtPipeline::kMaxDof;
}

bool RobotWorld::gpu_rrt_connect(const std::vector<float>& start, const std::vector<std::vector<float>>& goals, const GpuRrtSettings& settings, std::vector<std::vector<float>>& path, float* margin_used) const {
  if(!gpu_rrt_available()) return false;
  const FastCollision* fc = fast_collision();
  try {
    std::lock_guard<std::mutex> lock(fast_mutex_);
    if(!vk_rrt_) {
      GpuRobotData rd;
      if(!fc->pack_gpu_robot(rd)) throw std::runtime_error("robot exceeds GPU limits");
      if(!vk_ctx_) vk_ctx_ = std::make_unique<VkContext>();
      vk_rrt_         = std::make_unique<VkRrtPipeline>(*vk_ctx_, rd, fc->joint_reach(), config_.robot_config->joint_position_limits_lower(), config_.robot_config->joint_position_limits_upper());
      vk_rrt_version_ = ~uint64_t{0};
    }
    if(vk_rrt_version_ != config_.world_config->version()) {
      GpuWorldData wd;
      fc->pack_gpu_world(wd);
      vk_rrt_->set_world(wd);
      vk_rrt_version_ = config_.world_config->version();
    }
    return vk_rrt_->plan(start, goals, settings, path, margin_used);
  } catch(const std::exception&) {
    vk_rrt_failed_ = true;
    return false;
  }
}

const FastCollision* RobotWorld::fast_collision() const {
  std::lock_guard<std::mutex> lock(fast_mutex_);
  const uint64_t version = config_.world_config->version();
  if(!fast_) fast_ = std::make_unique<FastCollision>(*config_.robot_config, config_.acm);
  if(fast_version_ != version) {
    fast_supported_ = fast_->set_world(*config_.world_config);
    fast_version_   = version;
  }
  return fast_supported_ ? fast_.get() : nullptr;
}

bool RobotWorld::is_collision_free(const JointState& joint_state) const {
  if(const FastCollision* fc = fast_collision()) {
    if(joint_state.position.size() >= fc->dof()) return fc->is_free(joint_state.position.data());
  }
  for(float d : get_collision_distance(joint_state))
    if(d <= 0.0f) return false;
  for(float d : get_self_collision_distance(joint_state))
    if(d <= 0.0f) return false;
  return true;
}

std::vector<float> RobotWorld::get_min_world_collision_distance_batch(const std::vector<JointState>& joint_states) const {
  std::vector<float> result(joint_states.size(), std::numeric_limits<float>::infinity());
  if(!config_.use_vulkan) {
    for(size_t i = 0; i < joint_states.size(); i++) {
      for(float d : get_collision_distance(joint_states[i])) result[i] = std::min(result[i], d);
    }
    return result;
  }

  std::vector<CollisionSphere> all_spheres;
  std::vector<size_t> spheres_per_state(joint_states.size());
  for(size_t i = 0; i < joint_states.size(); i++) {
    std::vector<CollisionSphere> spheres = spheres_with_attached(*config_.robot_config, *config_.world_config, joint_states[i]);
    spheres_per_state[i]                 = spheres.size();
    all_spheres.insert(all_spheres.end(), spheres.begin(), spheres.end());
  }
  std::vector<float> distances = vulkan_pipeline().compute(all_spheres, *config_.world_config);

  size_t offset = 0;
  for(size_t i = 0; i < joint_states.size(); i++) {
    for(size_t k = 0; k < spheres_per_state[i]; k++) result[i] = std::min(result[i], distances[offset + k]);
    offset += spheres_per_state[i];
  }
  return result;
}

std::vector<bool> RobotWorld::repair_waypoints(std::vector<std::vector<float>*>& qs, float repair_step, float repair_delta, int max_iterations) const {
  if(qs.empty()) return {};

  // GPU常駐repairはロボット構造を構築時に1回だけアップロードするため、ここではattached objectは未対応(WorldConfigの障害物のみ)。
  if(config_.use_vulkan) return vulkan_repair_pipeline().repair(qs, *config_.world_config, repair_step, repair_delta, static_cast<uint32_t>(max_iterations));

  std::vector<bool> solved(qs.size(), false);
  const auto& lower = config_.robot_config->joint_position_limits_lower();
  const auto& upper = config_.robot_config->joint_position_limits_upper();
  const size_t dof   = qs[0]->size();

  std::vector<size_t> pending;
  for(size_t i = 0; i < qs.size(); i++) pending.push_back(i);

  {
    std::vector<JointState> states;
    for(size_t idx : pending) states.push_back(JointState{*qs[idx], {}});
    std::vector<float> dists = get_min_world_collision_distance_batch(states);
    std::vector<size_t> next;
    for(size_t k = 0; k < pending.size(); k++) {
      if(dists[k] > 0.0f) solved[pending[k]] = true;
      else next.push_back(pending[k]);
    }
    pending = std::move(next);
  }

  for(int iter = 0; iter < max_iterations && !pending.empty(); iter++) {
    std::vector<JointState> perturbed;
    perturbed.reserve(pending.size() * dof * 2);
    for(size_t idx : pending) {
      const auto& q = *qs[idx];
      for(size_t j = 0; j < dof; j++) {
        std::vector<float> qp = q, qm = q;
        qp[j] += repair_delta;
        qm[j] -= repair_delta;
        perturbed.push_back(JointState{std::move(qp), {}});
        perturbed.push_back(JointState{std::move(qm), {}});
      }
    }
    std::vector<float> dists = get_min_world_collision_distance_batch(perturbed);

    std::vector<size_t> still_moving;
    size_t offset = 0;
    for(size_t idx : pending) {
      auto& q = *qs[idx];
      std::vector<float> grad(dof, 0.0f);
      float norm = 0.0f;
      for(size_t j = 0; j < dof; j++) {
        grad[j] = (dists[offset + 2 * j] - dists[offset + 2 * j + 1]) / (2.0f * repair_delta);
        norm += grad[j] * grad[j];
      }
      offset += dof * 2;
      norm = std::sqrt(norm);
      if(norm < 1e-8f) continue;  // 勾配消失、この反復では動かせない
      for(size_t j = 0; j < dof; j++) {
        q[j] += repair_step * grad[j] / norm;
        q[j] = std::max(lower[j], std::min(upper[j], q[j]));
      }
      still_moving.push_back(idx);
    }

    if(!still_moving.empty()) {
      std::vector<JointState> recheck;
      for(size_t idx : still_moving) recheck.push_back(JointState{*qs[idx], {}});
      std::vector<float> redist = get_min_world_collision_distance_batch(recheck);
      pending.clear();
      for(size_t k = 0; k < still_moving.size(); k++) {
        if(redist[k] > 0.0f) solved[still_moving[k]] = true;
        else pending.push_back(still_moving[k]);
      }
    } else {
      pending.clear();
    }
  }
  return solved;
}

std::vector<bool> RobotWorld::is_collision_free_batch(const std::vector<JointState>& joint_states) const {
  if(const FastCollision* fc = fast_collision()) {
    const size_t dof = fc->dof();
    std::vector<float> flat(joint_states.size() * dof);
    for(size_t i = 0; i < joint_states.size(); i++) std::copy(joint_states[i].position.begin(), joint_states[i].position.begin() + dof, flat.begin() + i * dof);
    std::vector<uint8_t> ok(joint_states.size());
    if(config_.use_vulkan && joint_states.size() >= kGpuBatchThreshold && fc->gpu_compatible() && !vk_validate_failed_) {
      try {
        std::lock_guard<std::mutex> lock(fast_mutex_);
        if(!vk_validate_) {
          GpuRobotData rd;
          if(!fc->pack_gpu_robot(rd)) throw std::runtime_error("robot exceeds GPU validate limits");
          if(!vk_ctx_) vk_ctx_ = std::make_unique<VkContext>();
          vk_validate_ = std::make_unique<VkValidatePipeline>(*vk_ctx_, rd);
          vk_validate_version_ = ~uint64_t{0};
        }
        if(vk_validate_version_ != config_.world_config->version()) {
          GpuWorldData wd;
          fc->pack_gpu_world(wd);
          vk_validate_->set_world(wd);
          vk_validate_version_ = config_.world_config->version();
        }
        vk_validate_->validate(flat.data(), joint_states.size(), 0.0f, ok.data());
        return std::vector<bool>(ok.begin(), ok.end());
      } catch(const std::exception&) {
        vk_validate_failed_ = true;  // Vulkanが使えない/ロボットが対応外 -> 以後CPUのみ
      }
    }
    int threads = config_.num_threads > 0 ? config_.num_threads : static_cast<int>(std::min(8u, std::max(1u, std::thread::hardware_concurrency())));
    fc->is_free_batch(flat.data(), joint_states.size(), ok.data(), threads);
    return std::vector<bool>(ok.begin(), ok.end());
  }
  if(!config_.use_vulkan) {
    std::vector<bool> results;
    results.reserve(joint_states.size());
    for(const auto& js : joint_states) results.push_back(is_collision_free(js));
    return results;
  }

  std::vector<CollisionSphere> all_spheres;
  std::vector<size_t> spheres_per_state(joint_states.size());
  for(size_t i = 0; i < joint_states.size(); i++) {
    std::vector<CollisionSphere> spheres = spheres_with_attached(*config_.robot_config, *config_.world_config, joint_states[i]);
    spheres_per_state[i]                 = spheres.size();
    all_spheres.insert(all_spheres.end(), spheres.begin(), spheres.end());
  }
  std::vector<float> distances = vulkan_pipeline().compute(all_spheres, *config_.world_config);

  std::vector<bool> results(joint_states.size());
  size_t offset = 0;
  for(size_t i = 0; i < joint_states.size(); i++) {
    bool free = true;
    for(size_t k = 0; k < spheres_per_state[i]; k++)
      if(distances[offset + k] <= 0.0f) free = false;
    offset += spheres_per_state[i];
    if(free)
      for(float d : get_self_collision_distance(joint_states[i]))
        if(d <= 0.0f) free = false;
    results[i] = free;
  }
  return results;
}

}  // namespace vkmotion
