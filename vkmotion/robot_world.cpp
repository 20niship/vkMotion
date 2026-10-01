#include "robot_world.hpp"

#include <algorithm>
#include <limits>

#include "sphere_collision.hpp"

namespace vkmotion {

RobotWorld::RobotWorld(const RobotWorldConfig& config) : config_(config) {
  for(const auto& joint : config_.robot_config->urdf_model().joints) {
    adjacent_links_.emplace_back(joint.parent_link, joint.child_link);
  }
}

bool RobotWorld::is_adjacent(const std::string& a, const std::string& b) const {
  for(const auto& pair : adjacent_links_) {
    if((pair.first == a && pair.second == b) || (pair.first == b && pair.second == a)) return true;
  }
  return false;
}

std::vector<float> RobotWorld::get_collision_distance(const JointState& joint_state) const {
  std::vector<CollisionSphere> spheres = config_.robot_config->collision_spheres_world(joint_state);
  const WorldConfig& world             = *config_.world_config;

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

std::vector<bool> RobotWorld::is_collision_free_batch(const std::vector<JointState>& joint_states) const {
  std::vector<bool> results;
  results.reserve(joint_states.size());
  for(const auto& js : joint_states) results.push_back(is_collision_free(js));
  return results;
}

}  // namespace vkmotion
