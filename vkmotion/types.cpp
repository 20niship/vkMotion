#include "types.hpp"

#include <cassert>

namespace vkmotion {

Pose Pose::from_list(const std::vector<float>& xyzwxyz) {
  assert(xyzwxyz.size() == 7 && "Pose::from_list expects [x y z qw qx qy qz]");
  Pose p;
  p.position    = cutil::Vec3f(xyzwxyz[0], xyzwxyz[1], xyzwxyz[2]);
  p.quaternion  = cutil::Quatf(xyzwxyz[4], xyzwxyz[5], xyzwxyz[6], xyzwxyz[3]);
  return p;
}

}  // namespace vkmotion
