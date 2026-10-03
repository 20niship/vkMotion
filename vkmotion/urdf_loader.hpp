#pragma once

#include <cutil/vec.hpp>
#include <string>
#include <vector>

#include "types.hpp"

namespace vkmotion {

enum class UrdfJointType { kFixed, kRevolute, kContinuous, kPrismatic };

struct UrdfJoint {
  std::string name;
  UrdfJointType type = UrdfJointType::kFixed;
  std::string parent_link;
  std::string child_link;
  cutil::Vec3f origin_xyz{};
  cutil::Vec3f origin_rpy{};
  cutil::Vec3f axis{1.0f, 0.0f, 0.0f};
  float limit_lower = 0.0f;
  float limit_upper = 0.0f;
  float limit_velocity = 1.0e6f;  // URDFに無い場合は実質無制限
};

enum class UrdfGeometryType { kBox, kCylinder, kSphere, kCapsule, kMesh };

struct UrdfCollisionGeometry {
  UrdfGeometryType type = UrdfGeometryType::kBox;
  cutil::Vec3f origin_xyz{};
  cutil::Vec3f origin_rpy{};
  cutil::Vec3f box_size{};
  float radius = 0.0f;  // kCylinder, kSphere, kCapsule
  float length = 0.0f;  // kCylinder, kCapsule
};

struct UrdfLink {
  std::string name;
  std::vector<UrdfCollisionGeometry> collisions;
};

struct UrdfModel {
  std::string name;
  std::vector<UrdfLink> links;
  std::vector<UrdfJoint> joints;

  int find_link(const std::string& link_name) const;
};

UrdfModel load_urdf(const std::string& path);

std::vector<std::string> active_joint_names(const UrdfModel& model);

std::vector<CollisionSphere> generate_collision_spheres_from_urdf(const UrdfModel& model);

}  // namespace vkmotion
