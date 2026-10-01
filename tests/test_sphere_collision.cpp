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
