#pragma once

#include <cutil/mat.hpp>
#include <cutil/quaternion.hpp>
#include <cutil/vec.hpp>
#include <string>
#include <vector>

namespace vkmotion {

// Robot joint configuration (cuRobo: curobo.types.state.JointState).
struct JointState {
  std::vector<float> position;
  std::vector<float> velocity;  // optional, empty if unused
};

// 6-DoF pose (cuRobo: curobo.types.math.Pose).
struct Pose {
  cutil::Vec3f position{};
  cutil::Quatf quaternion = cutil::Quatf::identity();

  static Pose from_list(const std::vector<float>& xyzwxyz);
  cutil::Mat4f to_matrix() const;
};

// Sphere approximating part of the robot itself (not an Obstacle); center is in the owning link's local frame.
struct CollisionSphere {
  std::string link_name;
  cutil::Vec3f center{};
  float radius = 0.0f;
};

// World obstacle base (cuRobo: curobo.geom.types.Obstacle). pose is world-frame.
struct Obstacle {
  std::string name;
  Pose pose;
};

struct Cuboid : Obstacle {
  cutil::Vec3f dims{};  // full extents [x, y, z]
};

struct Sphere : Obstacle {
  float radius = 0.0f;
};

struct Cylinder : Obstacle {
  float radius = 0.0f;
  float height = 0.0f;  // along local Z
};

struct Capsule : Obstacle {
  float radius = 0.0f;
  cutil::Vec3f base{};  // local-frame segment endpoints
  cutil::Vec3f tip{};
};

}  // namespace vkmotion
