#pragma once

#include <cutil/ref.hpp>
#include <string>
#include <vector>

#include "types.hpp"

namespace vkmotion {

// Obstacle collection (cuRobo: curobo.geom.types.WorldConfig).
class WorldConfig {
 public:
  static cutil::Ref<WorldConfig> from_file(const std::string& yaml_path);

  void add_obstacle(const Cuboid& obstacle) { cuboid_.push_back(obstacle); }
  void add_obstacle(const Sphere& obstacle) { sphere_.push_back(obstacle); }
  void add_obstacle(const Cylinder& obstacle) { cylinder_.push_back(obstacle); }
  void add_obstacle(const Capsule& obstacle) { capsule_.push_back(obstacle); }

  const std::vector<Cuboid>& cuboid() const { return cuboid_; }
  const std::vector<Sphere>& sphere() const { return sphere_; }
  const std::vector<Cylinder>& cylinder() const { return cylinder_; }
  const std::vector<Capsule>& capsule() const { return capsule_; }

 private:
  std::vector<Cuboid> cuboid_;
  std::vector<Sphere> sphere_;
  std::vector<Cylinder> cylinder_;
  std::vector<Capsule> capsule_;
};

}  // namespace vkmotion
