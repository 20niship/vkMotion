#pragma once

#include <cutil/ref.hpp>
#include <string>
#include <vector>

#include "robot_config.hpp"
#include "types.hpp"
#include "world_config.hpp"

namespace vkmotion {

struct CartesianWaypoint {
  Pose pose;
};

struct CartesianPathConfig {
  float max_step         = 0.01f;  // max linear distance [m] between adjacent interpolated waypoints
  float max_angular_step = 0.05f;  // max rotation [rad] between adjacent interpolated waypoints
  float min_fraction      = 1.0f;  // result counts as success only if fraction_achieved >= this
};

struct CartesianPathResult {
  bool success             = false;
  float fraction_achieved  = 0.0f;  // cuRobo/MoveIt2 computeCartesianPath-style completion ratio, 0..1
  std::vector<JointState> trajectory;
  std::string status;
};

// Straight-line Cartesian path planning (MoveIt2: computeCartesianPath). Chains IK seeded by the previous solution along a lerp+slerp interpolation of the waypoints.
class CartesianPathPlanner {
 public:
  explicit CartesianPathPlanner(cutil::Ref<RobotConfig> robot_config, cutil::Ref<WorldConfig> world_config, bool use_vulkan = true);

  CartesianPathResult compute_cartesian_path(const JointState& start_state, const std::vector<CartesianWaypoint>& waypoints,
                                              const CartesianPathConfig& config = {});

 private:
  cutil::Ref<RobotConfig> robot_config_;
  cutil::Ref<WorldConfig> world_config_;
  bool use_vulkan_;
};

}  // namespace vkmotion
