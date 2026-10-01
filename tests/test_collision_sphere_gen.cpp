#include <cmath>
#include <doctest/doctest.h>
#include <vkmotion/urdf_loader.hpp>

using namespace vkmotion;

namespace {

UrdfModel make_single_geometry_model(const UrdfCollisionGeometry& geom) {
  UrdfModel model;
  UrdfLink link;
  link.name = "link";
  link.collisions.push_back(geom);
  model.links.push_back(link);
  return model;
}

}  // namespace

TEST_CASE("sphere geometry generates exactly one collision sphere matching its radius/center") {
  UrdfCollisionGeometry geom;
  geom.type       = UrdfGeometryType::kSphere;
  geom.radius     = 0.07f;
  geom.origin_xyz = cutil::Vec3f(1.0f, 2.0f, 3.0f);
  auto model      = make_single_geometry_model(geom);

  auto spheres = generate_collision_spheres_from_urdf(model);
  REQUIRE(spheres.size() == 1);
  CHECK(spheres[0].radius == doctest::Approx(0.07f));
  CHECK(spheres[0].center[0] == doctest::Approx(1.0f));
  CHECK(spheres[0].center[1] == doctest::Approx(2.0f));
  CHECK(spheres[0].center[2] == doctest::Approx(3.0f));
  CHECK(spheres[0].link_name == "link");
}

TEST_CASE("cylinder geometry is covered by multiple stacked spheres within its length") {
  UrdfCollisionGeometry geom;
  geom.type   = UrdfGeometryType::kCylinder;
  geom.radius = 0.02f;
  geom.length = 0.5f;
  auto model  = make_single_geometry_model(geom);

  auto spheres = generate_collision_spheres_from_urdf(model);
  CHECK(spheres.size() > 1);
  for(const auto& s : spheres) {
    CHECK(s.radius == doctest::Approx(0.02f));
    CHECK(std::abs(s.center[2]) <= geom.length * 0.5f + 1e-4f);
  }
}

TEST_CASE("box geometry spheres stay within the box bounds") {
  UrdfCollisionGeometry geom;
  geom.type     = UrdfGeometryType::kBox;
  geom.box_size = cutil::Vec3f(0.4f, 0.2f, 0.2f);
  auto model    = make_single_geometry_model(geom);

  auto spheres = generate_collision_spheres_from_urdf(model);
  CHECK(spheres.size() >= 1);
  for(const auto& s : spheres) {
    CHECK(std::abs(s.center[0]) <= geom.box_size[0] * 0.5f + 1e-4f);
    CHECK(std::abs(s.center[1]) <= geom.box_size[1] * 0.5f + 1e-4f);
    CHECK(std::abs(s.center[2]) <= geom.box_size[2] * 0.5f + 1e-4f);
  }
}
