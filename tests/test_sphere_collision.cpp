#include <doctest/doctest.h>
#include <vkmotion/sphere_collision.hpp>

using namespace vkmotion;

TEST_CASE("distance_sphere_sphere: separated / touching / penetrating") {
  Sphere s;
  s.pose.position = cutil::Vec3f(0.0f, 0.0f, 0.0f);
  s.radius         = 1.0f;

  CHECK(distance_sphere_sphere(cutil::Vec3f(3.0f, 0.0f, 0.0f), 1.0f, s) == doctest::Approx(1.0f));
  CHECK(distance_sphere_sphere(cutil::Vec3f(2.0f, 0.0f, 0.0f), 1.0f, s) == doctest::Approx(0.0f));
  CHECK(distance_sphere_sphere(cutil::Vec3f(0.0f, 0.0f, 0.0f), 1.0f, s) == doctest::Approx(-2.0f));
}

TEST_CASE("distance_sphere_cuboid: outside / inside") {
  Cuboid c;
  c.pose.position = cutil::Vec3f(0.0f, 0.0f, 0.0f);
  c.dims           = cutil::Vec3f(2.0f, 2.0f, 2.0f);  // half-extent 1 on each axis

  CHECK(distance_sphere_cuboid(cutil::Vec3f(3.0f, 0.0f, 0.0f), 0.5f, c) == doctest::Approx(1.5f));
  CHECK(distance_sphere_cuboid(cutil::Vec3f(0.0f, 0.0f, 0.0f), 0.1f, c) == doctest::Approx(-1.1f));
}

TEST_CASE("distance_sphere_cylinder: outside radially and axially") {
  Cylinder cyl;
  cyl.pose.position = cutil::Vec3f(0.0f, 0.0f, 0.0f);
  cyl.radius         = 1.0f;
  cyl.height         = 2.0f;  // z in [-1, 1]

  CHECK(distance_sphere_cylinder(cutil::Vec3f(3.0f, 0.0f, 0.0f), 0.5f, cyl) == doctest::Approx(1.5f));
  CHECK(distance_sphere_cylinder(cutil::Vec3f(0.0f, 0.0f, 0.0f), 0.1f, cyl) == doctest::Approx(-1.1f));
}

TEST_CASE("distance_sphere_capsule: along segment and off the end") {
  Capsule cap;
  cap.pose.position = cutil::Vec3f(0.0f, 0.0f, 0.0f);
  cap.radius         = 0.5f;
  cap.base           = cutil::Vec3f(0.0f, 0.0f, 0.0f);
  cap.tip            = cutil::Vec3f(0.0f, 0.0f, 1.0f);

  CHECK(distance_sphere_capsule(cutil::Vec3f(1.5f, 0.0f, 0.5f), 0.5f, cap) == doctest::Approx(0.5f));
  CHECK(distance_sphere_capsule(cutil::Vec3f(0.0f, 0.0f, 2.0f), 0.0f, cap) == doctest::Approx(0.5f));
}

TEST_CASE("distance_sphere_plane: above / below / straddling") {
  Plane pl;
  pl.pose.position   = cutil::Vec3f(0.0f, 0.0f, 0.0f);
  pl.pose.quaternion = cutil::Quatf::identity();  // normal = +Z

  CHECK(distance_sphere_plane(cutil::Vec3f(0.0f, 0.0f, 3.0f), 0.5f, pl) == doctest::Approx(2.5f));
  CHECK(distance_sphere_plane(cutil::Vec3f(0.0f, 0.0f, -1.0f), 0.5f, pl) == doctest::Approx(-1.5f));
}

TEST_CASE("distance_sphere_convex: matches distance_sphere_cuboid for a box hull") {
  Cuboid c;
  c.pose.position = cutil::Vec3f(0.0f, 0.0f, 0.0f);
  c.dims           = cutil::Vec3f(2.0f, 2.0f, 2.0f);  // half-extent 1

  ConvexMesh hull;
  hull.pose.position = c.pose.position;
  for(float sx : {-1.0f, 1.0f})
    for(float sy : {-1.0f, 1.0f})
      for(float sz : {-1.0f, 1.0f}) hull.vertices.push_back(cutil::Vec3f(sx, sy, sz));

  CHECK(distance_sphere_convex(cutil::Vec3f(3.0f, 0.0f, 0.0f), 0.5f, hull) == doctest::Approx(distance_sphere_cuboid(cutil::Vec3f(3.0f, 0.0f, 0.0f), 0.5f, c)));
  CHECK(distance_sphere_convex(cutil::Vec3f(0.0f, 2.0f, 0.0f), 0.25f, hull) ==
        doctest::Approx(distance_sphere_cuboid(cutil::Vec3f(0.0f, 2.0f, 0.0f), 0.25f, c)));
  // inside: convex has no EPA so it only guarantees the sign (negative), not the exact depth.
  CHECK(distance_sphere_convex(cutil::Vec3f(0.0f, 0.0f, 0.0f), 0.1f, hull) < 0.0f);
}

TEST_CASE("distance_sphere_sdf: trilinear lookup on a 2x2x2 grid") {
  SDFGrid grid;
  grid.pose.position = cutil::Vec3f(-1.0f, -1.0f, -1.0f);  // min corner
  grid.dims            = cutil::Vec3f(2.0f, 2.0f, 2.0f);
  grid.resolution[0] = grid.resolution[1] = grid.resolution[2] = 2;
  // distances represent an implicit sphere of radius 1 centered at the grid center (world origin)
  grid.distances = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f};  // corners are all distance 1

  CHECK(distance_sphere_sdf(cutil::Vec3f(1.0f, 1.0f, 1.0f), 0.0f, grid) == doctest::Approx(1.0f));
  CHECK(distance_sphere_sdf(cutil::Vec3f(1.0f, 1.0f, 1.0f), 0.5f, grid) == doctest::Approx(0.5f));
}
