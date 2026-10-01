#pragma once

#include <cutil/ref.hpp>
#include <vector>

#include "robot_config.hpp"
#include "types.hpp"
#include "world_config.hpp"

namespace vkmotion {

struct RobotWorldConfig {
  cutil::Ref<RobotConfig> robot_config;
  cutil::Ref<WorldConfig> world_config;
  bool use_vulkan = true;  // TODO: フェーズ3でVulkan compute実装。現状は無視し常にCPUで計算する。
};

// Robot-world and self collision queries (cuRobo: curobo.wrap.model.robot_world.RobotWorld).
class RobotWorld {
 public:
  explicit RobotWorld(const RobotWorldConfig& config);

  std::vector<float> get_collision_distance(const JointState& joint_state) const;
  std::vector<float> get_self_collision_distance(const JointState& joint_state) const;
  bool is_collision_free(const JointState& joint_state) const;
  std::vector<bool> is_collision_free_batch(const std::vector<JointState>& joint_states) const;

 private:
  RobotWorldConfig config_;
  // Unordered pairs of link names that share a joint; excluded from self-collision checks.
  std::vector<std::pair<std::string, std::string>> adjacent_links_;

  bool is_adjacent(const std::string& a, const std::string& b) const;
};

}  // namespace vkmotion
