#include "world_config.hpp"

#include <fstream>
#include <regex>
#include <sstream>
#include <stdexcept>

namespace vkmotion {

namespace {

std::vector<float> parse_float_list(const std::string& bracketed) {
  std::vector<float> values;
  std::istringstream ss(bracketed);
  std::string component;
  while(std::getline(ss, component, ',')) values.push_back(std::stof(component));
  return values;
}

Pose parse_pose(const std::vector<float>& v) {
  if(v.size() != 7) throw std::runtime_error("vkmotion: WorldConfig pose requires 7 values [x y z qw qx qy qz]");
  return Pose::from_list(v);
}

cutil::Vec3f parse_vec3(const std::vector<float>& v) {
  if(v.size() != 3) throw std::runtime_error("vkmotion: WorldConfig expected a 3-component field");
  return cutil::Vec3f(v[0], v[1], v[2]);
}

}  // namespace

// Minimal parser for the fixed schema documented in MANUAL.md (category -> name -> field).
cutil::Ref<WorldConfig> WorldConfig::from_file(const std::string& yaml_path) {
  std::ifstream file(yaml_path);
  if(!file) throw std::runtime_error("vkmotion: failed to open WorldConfig YAML: " + yaml_path);

  static const std::regex category_re(R"(^(\w+):\s*$)");
  static const std::regex name_re(R"(^\s{2}(\S+):\s*$)");
  static const std::regex list_field_re(R"(^\s{4}(\w+):\s*\[\s*([^\]]*)\]\s*$)");
  static const std::regex scalar_field_re(R"(^\s{4}(\w+):\s*([0-9eE+\-.]+)\s*$)");

  cutil::Ref<WorldConfig> world = cutil::make_ref<WorldConfig>();

  std::string category;
  std::string name;
  Pose pose;
  cutil::Vec3f dims{}, base{}, tip{};
  float radius = 0.0f, height = 0.0f;

  auto flush = [&]() {
    if(name.empty()) return;
    if(category == "cuboid") {
      Cuboid o;
      o.name = name;
      o.pose = pose;
      o.dims = dims;
      world->add_obstacle(o);
    } else if(category == "sphere") {
      Sphere o;
      o.name   = name;
      o.pose   = pose;
      o.radius = radius;
      world->add_obstacle(o);
    } else if(category == "cylinder") {
      Cylinder o;
      o.name   = name;
      o.pose   = pose;
      o.radius = radius;
      o.height = height;
      world->add_obstacle(o);
    } else if(category == "capsule") {
      Capsule o;
      o.name   = name;
      o.pose   = pose;
      o.radius = radius;
      o.base   = base;
      o.tip    = tip;
      world->add_obstacle(o);
    }
  };

  std::string line;
  while(std::getline(file, line)) {
    std::smatch m;
    if(std::regex_match(line, m, category_re)) {
      flush();
      category = m[1].str();
      name.clear();
      continue;
    }
    if(std::regex_match(line, m, name_re)) {
      flush();
      name   = m[1].str();
      pose   = Pose{};
      dims = base = tip = cutil::Vec3f{};
      radius = height = 0.0f;
      continue;
    }
    if(std::regex_match(line, m, list_field_re)) {
      std::vector<float> values = parse_float_list(m[2].str());
      const std::string& field  = m[1].str();
      if(field == "pose") pose = parse_pose(values);
      else if(field == "dims") dims = parse_vec3(values);
      else if(field == "base") base = parse_vec3(values);
      else if(field == "tip") tip = parse_vec3(values);
      continue;
    }
    if(std::regex_match(line, m, scalar_field_re)) {
      const std::string& field = m[1].str();
      float value               = std::stof(m[2].str());
      if(field == "radius") radius = value;
      else if(field == "height") height = value;
      continue;
    }
  }
  flush();
  return world;
}

}  // namespace vkmotion
