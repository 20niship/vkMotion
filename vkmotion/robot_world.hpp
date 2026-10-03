#pragma once

#include <cutil/ref.hpp>
#include <memory>
#include <vector>

#include "allowed_collision_matrix.hpp"
#include "robot_config.hpp"
#include "types.hpp"
#include "world_config.hpp"

namespace vkmotion {

class VkContext;
class VkCollisionPipeline;
class VkRepairPipeline;

struct RobotWorldConfig {
  cutil::Ref<RobotConfig> robot_config;
  cutil::Ref<WorldConfig> world_config;
  bool use_vulkan = true;
  AllowedCollisionMatrix acm;  // self-collision(CPU)とCPU側world障害物チェックに適用。GPU経路のworld障害物チェックは未対応(下記参照)。
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

  // 衝突しているwaypointを同時並行でrepair。use_vulkan=trueならFK+反復ループ全体を1 dispatchでGPU常駐実行(直列チェーンのみ)、falseならCPU逐次反復。qsはその場で更新、戻り値は同順の収束成否。
  std::vector<bool> repair_waypoints(std::vector<std::vector<float>*>& qs, float repair_step, float repair_delta, int max_iterations) const;

 private:
  RobotWorldConfig config_;
  // Unordered pairs of link names that share a joint; excluded from self-collision checks.
  std::vector<std::pair<std::string, std::string>> adjacent_links_;
  mutable std::unique_ptr<VkContext> vk_ctx_;
  mutable std::unique_ptr<VkCollisionPipeline> vk_collision_;
  mutable std::unique_ptr<VkRepairPipeline> vk_repair_;

  bool is_adjacent(const std::string& a, const std::string& b) const;
  VkCollisionPipeline& vulkan_pipeline() const;
  VkRepairPipeline& vulkan_repair_pipeline() const;
};

}  // namespace vkmotion
