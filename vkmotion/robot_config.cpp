#include "robot_config.hpp"

#include <cutil/vec.hpp>
#include <fstream>
#include <regex>
#include <sstream>
#include <stdexcept>

#include "kinematics.hpp"

namespace vkmotion {

namespace {

// Reads only the limited schema documented in MANUAL.md (collision_spheres: <link>: [{center, radius}]).
std::vector<CollisionSphere> load_collision_spheres_yaml(const std::string& path) {
  std::ifstream file(path);
  if(!file) throw std::runtime_error("vkmotion: failed to open collision spheres YAML: " + path);

  static const std::regex link_re(R"(^\s{2}(\S+):\s*$)");
  static const std::regex center_re(R"(^\s*-\s*center:\s*\[\s*([^\]]*)\]\s*$)");
  static const std::regex radius_re(R"(^\s*radius:\s*([0-9eE+\-.]+)\s*$)");

  std::vector<CollisionSphere> spheres;
  std::string current_link;
  std::string line;
  CollisionSphere pending;
  bool has_pending_center = false;

  while(std::getline(file, line)) {
    std::smatch m;
    if(std::regex_match(line, m, link_re)) {
      current_link = m[1].str();
      continue;
    }
    if(std::regex_match(line, m, center_re)) {
      std::istringstream xyz(m[1].str());
      std::string component;
      float v[3] = {0.0f, 0.0f, 0.0f};
      for(int i = 0; i < 3 && std::getline(xyz, component, ','); i++) v[i] = std::stof(component);
      pending             = CollisionSphere{};
      pending.link_name   = current_link;
      pending.center      = cutil::Vec3f(v[0], v[1], v[2]);
      has_pending_center = true;
      continue;
    }
    if(has_pending_center && std::regex_match(line, m, radius_re)) {
      pending.radius = std::stof(m[1].str());
      spheres.push_back(pending);
      has_pending_center = false;
    }
  }
  return spheres;
}

}  // namespace

cutil::Ref<RobotConfig> RobotConfig::from_file(const std::string& urdf_path, const std::string& collision_spheres_yaml_path) {
  cutil::Ref<RobotConfig> cfg = cutil::make_ref<RobotConfig>();

  cfg->model_       = load_urdf(urdf_path);
  cfg->joint_names_ = active_joint_names(cfg->model_);
  for(const auto& name : cfg->joint_names_) {
    for(const auto& j : cfg->model_.joints) {
      if(j.name == name) {
        cfg->limits_lower_.push_back(j.limit_lower);
        cfg->limits_upper_.push_back(j.limit_upper);
        break;
      }
    }
  }
  for(const auto& link : cfg->model_.links) cfg->link_names_.push_back(link.name);

  cfg->collision_spheres_ = collision_spheres_yaml_path.empty() ? generate_collision_spheres_from_urdf(cfg->model_)
                                                                 : load_collision_spheres_yaml(collision_spheres_yaml_path);
  return cfg;
}

std::vector<cutil::Mat4f> RobotConfig::forward_kinematics(const JointState& joint_state) const {
  return vkmotion::forward_kinematics(model_, joint_state.position);
}

std::vector<CollisionSphere> RobotConfig::collision_spheres_world(const JointState& joint_state) const {
  std::vector<cutil::Mat4f> link_world = forward_kinematics(joint_state);
  std::vector<CollisionSphere> world_spheres;
  world_spheres.reserve(collision_spheres_.size());
  for(const auto& s : collision_spheres_) {
    int li = model_.find_link(s.link_name);
    cutil::Mat4f xform = (li >= 0) ? link_world[static_cast<size_t>(li)] : cutil::Mat4f::identity();
    CollisionSphere ws;
    ws.link_name = s.link_name;
    ws.radius    = s.radius;
    ws.center    = (xform * cutil::NVec<4, float>(s.center, 1.0f)).xyz();
    world_spheres.push_back(ws);
  }
  return world_spheres;
}

}  // namespace vkmotion
