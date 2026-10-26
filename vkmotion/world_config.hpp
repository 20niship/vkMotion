#pragma once

#include <cutil/ref.hpp>
#include <cstdint>
#include <string>
#include <vector>

#include "types.hpp"

namespace vkmotion {

// Obstacle collection (cuRobo: curobo.geom.types.WorldConfig).
class WorldConfig {
 public:
  static cutil::Ref<WorldConfig> from_file(const std::string& yaml_path);

  void add_obstacle(const Cuboid& obstacle) {
    cuboid_.push_back(obstacle);
    version_++;
  }
  void add_obstacle(const Sphere& obstacle) {
    sphere_.push_back(obstacle);
    version_++;
  }
  void add_obstacle(const Cylinder& obstacle) {
    cylinder_.push_back(obstacle);
    version_++;
  }
  void add_obstacle(const Capsule& obstacle) {
    capsule_.push_back(obstacle);
    version_++;
  }
  void add_obstacle(const Plane& obstacle) {
    plane_.push_back(obstacle);
    version_++;
  }
  void add_obstacle(const ConvexMesh& obstacle) {
    convex_mesh_.push_back(obstacle);
    version_++;
  }
  void add_obstacle(const SDFGrid& obstacle) {
    sdf_grid_.push_back(obstacle);
    version_++;
  }
  void add_attached_object(const AttachedObject& obstacle) {
    attached_.push_back(obstacle);
    version_++;
  }
  // 内容が変わるたびに増える。コンパイル済みキャッシュ(RobotWorldのfast path)の無効化判定に使う。
  uint64_t version() const { return version_; }
  void clear() {
    version_++;
    cuboid_.clear();
    sphere_.clear();
    cylinder_.clear();
    capsule_.clear();
    plane_.clear();
    convex_mesh_.clear();
    sdf_grid_.clear();
    attached_.clear();
  }

  const std::vector<Cuboid>& cuboid() const { return cuboid_; }
  const std::vector<Sphere>& sphere() const { return sphere_; }
  const std::vector<Cylinder>& cylinder() const { return cylinder_; }
  const std::vector<Capsule>& capsule() const { return capsule_; }
  const std::vector<Plane>& plane() const { return plane_; }
  const std::vector<ConvexMesh>& convex_mesh() const { return convex_mesh_; }
  const std::vector<SDFGrid>& sdf_grid() const { return sdf_grid_; }
  const std::vector<AttachedObject>& attached_objects() const { return attached_; }

 private:
  uint64_t version_ = 0;
  std::vector<Cuboid> cuboid_;
  std::vector<Sphere> sphere_;
  std::vector<Cylinder> cylinder_;
  std::vector<Capsule> capsule_;
  std::vector<Plane> plane_;
  std::vector<ConvexMesh> convex_mesh_;
  std::vector<SDFGrid> sdf_grid_;
  std::vector<AttachedObject> attached_;
};

}  // namespace vkmotion
