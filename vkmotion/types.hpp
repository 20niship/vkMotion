#pragma once

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
};

// Sphere approximating part of the robot itself (not an Obstacle); center is in the owning link's local frame.
struct CollisionSphere {
  std::string link_name;
  cutil::Vec3f center{};
  float radius = 0.0f;
};

}  // namespace vkmotion
