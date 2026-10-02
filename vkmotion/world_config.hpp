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
  void add_obstacle(const Plane& obstacle) { plane_.push_back(obstacle); }
  void add_obstacle(const ConvexMesh& obstacle) { convex_mesh_.push_back(obstacle); }
  void add_obstacle(const SDFGrid& obstacle) { sdf_grid_.push_back(obstacle); }
  void clear() {
    cuboid_.clear();
    sphere_.clear();
    cylinder_.clear();
    capsule_.clear();
    plane_.clear();
    convex_mesh_.clear();
    sdf_grid_.clear();
  }

  const std::vector<Cuboid>& cuboid() const { return cuboid_; }
  const std::vector<Sphere>& sphere() const { return sphere_; }
  const std::vector<Cylinder>& cylinder() const { return cylinder_; }
  const std::vector<Capsule>& capsule() const { return capsule_; }
  const std::vector<Plane>& plane() const { return plane_; }
  const std::vector<ConvexMesh>& convex_mesh() const { return convex_mesh_; }
  const std::vector<SDFGrid>& sdf_grid() const { return sdf_grid_; }

 private:
  std::vector<Cuboid> cuboid_;
  std::vector<Sphere> sphere_;
  std::vector<Cylinder> cylinder_;
  std::vector<Capsule> capsule_;
  std::vector<Plane> plane_;
  std::vector<ConvexMesh> convex_mesh_;
  std::vector<SDFGrid> sdf_grid_;
};

}  // namespace vkmotion
