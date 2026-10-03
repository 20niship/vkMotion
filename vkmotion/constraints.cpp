#include "constraints.hpp"

#include <algorithm>
#include <cmath>

namespace vkmotion {

float OrientationConstraint::violation(const JointState& js, const RobotConfig& robot) const {
  const cutil::Mat4f& t = robot.forward_kinematics(js).back();
  cutil::Mat3f basis;
  for(unsigned int r = 0; r < 3; r++)
    for(unsigned int c = 0; c < 3; c++) basis(r, c) = t(r, c);
  cutil::Quatf q(basis);

  float d     = std::clamp(std::abs(q.dot(target)), 0.0f, 1.0f);
  float angle = 2.0f * std::acos(d);
  return angle - tolerance_rad;
}

}  // namespace vkmotion
