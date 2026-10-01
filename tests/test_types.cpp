#include <doctest/doctest.h>
#include <vkmotion/types.hpp>

TEST_CASE("Pose::from_list roundtrips position and quaternion") {
  vkmotion::Pose p = vkmotion::Pose::from_list({1.0f, 2.0f, 3.0f, 1.0f, 0.0f, 0.0f, 0.0f});
  CHECK(p.position[0] == doctest::Approx(1.0f));
  CHECK(p.position[1] == doctest::Approx(2.0f));
  CHECK(p.position[2] == doctest::Approx(3.0f));
  CHECK(p.quaternion.w == doctest::Approx(1.0f));
}
