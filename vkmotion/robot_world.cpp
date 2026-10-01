#include "robot_world.hpp"

#include <algorithm>
#include <limits>

#include "sphere_collision.hpp"
#include "vk_collision.hpp"
#include "vk_context.hpp"

namespace vkmotion {

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

std::vector<float> RobotWorld::get_collision_distance(const JointState& joint_state) const {
  std::vector<CollisionSphere> spheres = config_.robot_config->collision_spheres_world(joint_state);
  const WorldConfig& world             = *config_.world_config;

  if(config_.use_vulkan) return vulkan_pipeline().compute(spheres, world);

  std::vector<float> distances;
  distances.reserve(spheres.size());
  for(const auto& s : spheres) {
    float min_dist = std::numeric_limits<float>::infinity();
    for(const auto& o : world.cuboid()) min_dist = std::min(min_dist, distance_sphere_cuboid(s.center, s.radius, o));
    for(const auto& o : world.sphere()) min_dist = std::min(min_dist, distance_sphere_sphere(s.center, s.radius, o));
    for(const auto& o : world.cylinder()) min_dist = std::min(min_dist, distance_sphere_cylinder(s.center, s.radius, o));
    for(const auto& o : world.capsule()) min_dist = std::min(min_dist, distance_sphere_capsule(s.center, s.radius, o));
    distances.push_back(min_dist);
  }
  return distances;
}

std::vector<float> RobotWorld::get_self_collision_distance(const JointState& joint_state) const {
  std::vector<CollisionSphere> spheres = config_.robot_config->collision_spheres_world(joint_state);

  std::vector<float> distances;
  for(size_t i = 0; i < spheres.size(); i++) {
    for(size_t j = i + 1; j < spheres.size(); j++) {
      const auto& a = spheres[i];
      const auto& b = spheres[j];
      if(a.link_name == b.link_name) continue;
      if(is_adjacent(a.link_name, b.link_name)) continue;
      float d = static_cast<float>((a.center - b.center).length()) - a.radius - b.radius;
      distances.push_back(d);
    }
  }
  return distances;
}

bool RobotWorld::is_collision_free(const JointState& joint_state) const {
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
    std::vector<CollisionSphere> spheres = config_.robot_config->collision_spheres_world(joint_states[i]);
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

std::vector<bool> RobotWorld::is_collision_free_batch(const std::vector<JointState>& joint_states) const {
  if(!config_.use_vulkan) {
    std::vector<bool> results;
    results.reserve(joint_states.size());
    for(const auto& js : joint_states) results.push_back(is_collision_free(js));
    return results;
  }

  std::vector<CollisionSphere> all_spheres;
  std::vector<size_t> spheres_per_state(joint_states.size());
  for(size_t i = 0; i < joint_states.size(); i++) {
    std::vector<CollisionSphere> spheres = config_.robot_config->collision_spheres_world(joint_states[i]);
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
