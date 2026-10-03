#pragma once

#include <cutil/vec.hpp>

#include "types.hpp"

namespace vkmotion {

// Signed distance from sphere surface to obstacle surface (negative = penetrating).

float distance_sphere_sphere(const cutil::Vec3f& center, float radius, const Sphere& obstacle);
float distance_sphere_cuboid(const cutil::Vec3f& center, float radius, const Cuboid& obstacle);
float distance_sphere_cylinder(const cutil::Vec3f& center, float radius, const Cylinder& obstacle);
float distance_sphere_capsule(const cutil::Vec3f& center, float radius, const Capsule& obstacle);
float distance_sphere_plane(const cutil::Vec3f& center, float radius, const Plane& obstacle);
float distance_sphere_convex(const cutil::Vec3f& center, float radius, const ConvexMesh& obstacle);
float distance_sphere_sdf(const cutil::Vec3f& center, float radius, const SDFGrid& obstacle);

}  // namespace vkmotion
