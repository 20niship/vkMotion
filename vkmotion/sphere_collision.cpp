#include "sphere_collision.hpp"

#include <algorithm>
#include <cmath>

namespace vkmotion {

namespace {

cutil::Vec3f to_local(const cutil::Vec3f& world_point, const Pose& pose) {
  cutil::Vec3f d = world_point - pose.position;
  cutil::Quatf inv = pose.quaternion.inverse();
  cutil::Mat<3, 3, float> r = inv.to_mat3();
  return cutil::Vec3f(r(0, 0) * d[0] + r(0, 1) * d[1] + r(0, 2) * d[2], r(1, 0) * d[0] + r(1, 1) * d[1] + r(1, 2) * d[2],
                       r(2, 0) * d[0] + r(2, 1) * d[1] + r(2, 2) * d[2]);
}

float closest_point_on_segment_distance(const cutil::Vec3f& p, const cutil::Vec3f& a, const cutil::Vec3f& b) {
  cutil::Vec3f ab = b - a;
  float len2 = static_cast<float>(ab.length_squared());
  float t = len2 > 1e-12f ? static_cast<float>((p - a).dot(ab)) / len2 : 0.0f;
  t = std::clamp(t, 0.0f, 1.0f);
  cutil::Vec3f closest = a + ab * t;
  return static_cast<float>((p - closest).length());
}

}  // namespace

float distance_sphere_sphere(const cutil::Vec3f& center, float radius, const Sphere& obstacle) {
  float d = static_cast<float>((center - obstacle.pose.position).length());
  return d - radius - obstacle.radius;
}

float distance_sphere_cuboid(const cutil::Vec3f& center, float radius, const Cuboid& obstacle) {
  cutil::Vec3f local = to_local(center, obstacle.pose);
  cutil::Vec3f half = obstacle.dims * 0.5f;
  cutil::Vec3f q(std::abs(local[0]) - half[0], std::abs(local[1]) - half[1], std::abs(local[2]) - half[2]);
  cutil::Vec3f q_pos(std::max(q[0], 0.0f), std::max(q[1], 0.0f), std::max(q[2], 0.0f));
  float outside = static_cast<float>(q_pos.length());
  float inside = std::min(std::max(q[0], std::max(q[1], q[2])), 0.0f);
  return outside + inside - radius;
}

float distance_sphere_cylinder(const cutil::Vec3f& center, float radius, const Cylinder& obstacle) {
  cutil::Vec3f local = to_local(center, obstacle.pose);
  float radial = std::sqrt(local[0] * local[0] + local[1] * local[1]) - obstacle.radius;
  float vertical = std::abs(local[2]) - obstacle.height * 0.5f;
  float outside = std::sqrt(std::max(radial, 0.0f) * std::max(radial, 0.0f) + std::max(vertical, 0.0f) * std::max(vertical, 0.0f));
  float inside = std::min(std::max(radial, vertical), 0.0f);
  return outside + inside - radius;
}

float distance_sphere_capsule(const cutil::Vec3f& center, float radius, const Capsule& obstacle) {
  cutil::Vec3f local = to_local(center, obstacle.pose);
  float d = closest_point_on_segment_distance(local, obstacle.base, obstacle.tip);
  return d - obstacle.radius - radius;
}

}  // namespace vkmotion
