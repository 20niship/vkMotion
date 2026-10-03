#include <doctest/doctest.h>
#include <vkmotion/world_config.hpp>

using namespace vkmotion;

TEST_CASE("WorldConfig::from_file parses all obstacle categories") {
  auto world = WorldConfig::from_file(std::string(VKMOTION_TEST_ASSET_DIR) + "/world/sample_world.yml");

  REQUIRE(world->cuboid().size() == 1);
  CHECK(world->cuboid()[0].name == "table");
  CHECK(world->cuboid()[0].dims[0] == doctest::Approx(1.0f));
  CHECK(world->cuboid()[0].pose.position[0] == doctest::Approx(0.5f));

  REQUIRE(world->sphere().size() == 1);
  CHECK(world->sphere()[0].radius == doctest::Approx(0.05f));

  REQUIRE(world->cylinder().size() == 1);
  CHECK(world->cylinder()[0].height == doctest::Approx(0.5f));

  REQUIRE(world->capsule().size() == 1);
  CHECK(world->capsule()[0].tip[2] == doctest::Approx(0.3f));
}
