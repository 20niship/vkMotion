#pragma once

#include <cutil/ref.hpp>
#include <string>
#include <vector>

#include "robot_config.hpp"
#include "types.hpp"
#include "world_config.hpp"

namespace vkmotion {

struct RrtConnectConfig {
  int max_iterations        = 2000;
  float step_size           = 0.1f;
  int max_samples_per_batch = 16;
};

struct RrtConnectResult {
  bool success = false;
  std::vector<JointState> trajectory;  // 粗いwaypoint列(呼び出し側でshortcut/time parameterizeする前提)
  std::string status;
};

// 双方向木のRRT-Connect。局所勾配降下のrepairだけでは迂回できない大域的な障害物配置向けのフォールバック。
class RrtConnectPlanner {
 public:
  explicit RrtConnectPlanner(cutil::Ref<RobotConfig> robot_config, cutil::Ref<WorldConfig> world_config, bool use_vulkan = true);
  RrtConnectResult plan(const JointState& start, const JointState& goal, const RrtConnectConfig& config = {});

 private:
  cutil::Ref<RobotConfig> robot_config_;
  cutil::Ref<WorldConfig> world_config_;
  bool use_vulkan_;
};

}  // namespace vkmotion
