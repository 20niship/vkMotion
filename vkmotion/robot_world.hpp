#pragma once

#include <cutil/ref.hpp>
#include <memory>
#include <vector>

#include "robot_config.hpp"
#include "types.hpp"
#include "world_config.hpp"

namespace vkmotion {

class VkContext;
class VkCollisionPipeline;

struct RobotWorldConfig {
  cutil::Ref<RobotConfig> robot_config;
  cutil::Ref<WorldConfig> world_config;
  bool use_vulkan = true;
};

// Robot-world and self collision queries (cuRobo: curobo.wrap.model.robot_world.RobotWorld).
class RobotWorld {
 public:
  explicit RobotWorld(const RobotWorldConfig& config);
  ~RobotWorld();

  std::vector<float> get_collision_distance(const JointState& joint_state) const;
  std::vector<float> get_self_collision_distance(const JointState& joint_state) const;
  bool is_collision_free(const JointState& joint_state) const;
  std::vector<bool> is_collision_free_batch(const std::vector<JointState>& joint_states) const;
  // 世界障害物との最小符号付き距離(自己衝突は含まない)を状態ごとに1回のdispatchでまとめて返す。
  std::vector<float> get_min_world_collision_distance_batch(const std::vector<JointState>& joint_states) const;

 private:
  RobotWorldConfig config_;
  // Unordered pairs of link names that share a joint; excluded from self-collision checks.
  std::vector<std::pair<std::string, std::string>> adjacent_links_;
  mutable std::unique_ptr<VkContext> vk_ctx_;
  mutable std::unique_ptr<VkCollisionPipeline> vk_collision_;

  bool is_adjacent(const std::string& a, const std::string& b) const;
  VkCollisionPipeline& vulkan_pipeline() const;
};

}  // namespace vkmotion
