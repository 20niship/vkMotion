#include "sphere_collision.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

#include "gjk.hpp"

namespace vkmotion {

namespace {

cutil::Vec3f rotate(const cutil::Quatf& q, const cutil::Vec3f& v) {
  cutil::Mat<3, 3, float> r = q.to_mat3();
  return cutil::Vec3f(r(0, 0) * v[0] + r(0, 1) * v[1] + r(0, 2) * v[2], r(1, 0) * v[0] + r(1, 1) * v[1] + r(1, 2) * v[2],
                       r(2, 0) * v[0] + r(2, 1) * v[1] + r(2, 2) * v[2]);
}

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

float distance_sphere_plane(const cutil::Vec3f& center, float radius, const Plane& obstacle) {
  cutil::Vec3f normal = rotate(obstacle.pose.quaternion, cutil::Vec3f(0.0f, 0.0f, 1.0f));
  float d = static_cast<float>((center - obstacle.pose.position).dot(normal));
  return d - radius;
}

float distance_sphere_convex(const cutil::Vec3f& center, float radius, const ConvexMesh& obstacle) {
  cutil::Vec3f local = to_local(center, obstacle.pose);
  bool inside         = false;
  float d             = gjk_point_convex_distance(local, obstacle.vertices, inside);
  float surf          = inside ? -d : d;  // no EPA: `d` is nearest-vertex distance when inside, not exact depth
  return surf - radius;
}

float distance_sphere_sdf(const cutil::Vec3f& center, float radius, const SDFGrid& obstacle) {
  cutil::Vec3f local = to_local(center, obstacle.pose);
  int rx = obstacle.resolution[0], ry = obstacle.resolution[1], rz = obstacle.resolution[2];
  if(rx < 2 || ry < 2 || rz < 2) return std::numeric_limits<float>::infinity();

  cutil::Vec3f cell(obstacle.dims[0] / static_cast<float>(rx - 1), obstacle.dims[1] / static_cast<float>(ry - 1),
                     obstacle.dims[2] / static_cast<float>(rz - 1));
  cutil::Vec3f g(local[0] / cell[0], local[1] / cell[1], local[2] / cell[2]);
  g[0] = std::clamp(g[0], 0.0f, static_cast<float>(rx - 1));
  g[1] = std::clamp(g[1], 0.0f, static_cast<float>(ry - 1));
  g[2] = std::clamp(g[2], 0.0f, static_cast<float>(rz - 1));

  int x0 = std::min(static_cast<int>(g[0]), rx - 2), y0 = std::min(static_cast<int>(g[1]), ry - 2), z0 = std::min(static_cast<int>(g[2]), rz - 2);
  float fx = g[0] - static_cast<float>(x0), fy = g[1] - static_cast<float>(y0), fz = g[2] - static_cast<float>(z0);

  auto at = [&](int x, int y, int z) { return obstacle.distances[static_cast<size_t>(x + y * rx + z * rx * ry)]; };
  float c00 = at(x0, y0, z0) * (1 - fx) + at(x0 + 1, y0, z0) * fx;
  float c10 = at(x0, y0 + 1, z0) * (1 - fx) + at(x0 + 1, y0 + 1, z0) * fx;
  float c01 = at(x0, y0, z0 + 1) * (1 - fx) + at(x0 + 1, y0, z0 + 1) * fx;
  float c11 = at(x0, y0 + 1, z0 + 1) * (1 - fx) + at(x0 + 1, y0 + 1, z0 + 1) * fx;
  float c0 = c00 * (1 - fy) + c10 * fy;
  float c1 = c01 * (1 - fy) + c11 * fy;
  float d   = c0 * (1 - fz) + c1 * fz;
  return d - radius;
}

}  // namespace vkmotion
