#pragma once

#include <cutil/ref.hpp>
#include <memory>
#include <mutex>
#include <vector>

#include "allowed_collision_matrix.hpp"
#include "robot_config.hpp"
#include "types.hpp"
#include "world_config.hpp"

namespace vkmotion {

class FastCollision;
class VkContext;
class VkCollisionPipeline;
class VkRepairPipeline;
class VkValidatePipeline;
class VkRrtPipeline;
struct GpuRrtSettings;

struct RobotWorldConfig {
  cutil::Ref<RobotConfig> robot_config;
  cutil::Ref<WorldConfig> world_config;
  bool use_vulkan = true;
  AllowedCollisionMatrix acm;  // self-collision(CPU)とCPU側world障害物チェックに適用。GPU経路のworld障害物チェックは未対応(下記参照)。
  int num_threads = 0;         // バッチ判定のCPUスレッド数。0=自動(hardware_concurrency、最大8)
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
  // コンパイル済みの高速衝突判定(障害物がcuboid/sphere/cylinder/capsule/planeのみでattached objectが無いとき)。未対応ならnullptr。
  // 世界を書き換えた後に呼べば自動で作り直す。返したポインタは次にworldが変わるまで有効。
  const FastCollision* fast_collision() const;

  // GPU上で双方向RRT-Connectを複数本並列に走らせ、最初に解けた経路を返す(shaders/rrt_gpu.comp)。
  // GPUが使えない/ロボットが非対応(dof>7, 球/直方体以外の障害物など)ならfalseを返し、以後も常にfalse。解けなかった場合もfalse。
  bool gpu_rrt_connect(const std::vector<float>& start, const std::vector<std::vector<float>>& goals, const GpuRrtSettings& settings, std::vector<std::vector<float>>& path, float* margin_used = nullptr) const;
  bool gpu_rrt_available() const;

  // コンパイル済み衝突判定とGPUパイプライン(use_vulkan時)を先に作る。初回呼び出しのレイテンシ(Vulkan初期化・シェーダ読込)を計測/制御ループから外すため。
  void warmup() const;

  // use_vulkan=trueかつこの件数以上のバッチはGPU(robot_validate.comp)で判定する。それ未満はCPU SIMDの方が速い(dispatch固定費の分)。
  static constexpr size_t kGpuBatchThreshold = 4096;

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
  mutable std::unique_ptr<VkValidatePipeline> vk_validate_;
  mutable uint64_t vk_validate_version_ = ~uint64_t{0};
  mutable bool vk_validate_failed_      = false;
  mutable std::unique_ptr<VkRrtPipeline> vk_rrt_;
  mutable uint64_t vk_rrt_version_ = ~uint64_t{0};
  mutable bool vk_rrt_failed_      = false;
  mutable std::mutex fast_mutex_;
  mutable std::unique_ptr<FastCollision> fast_;
  mutable uint64_t fast_version_ = ~uint64_t{0};
  mutable bool fast_supported_   = false;

  bool is_adjacent(const std::string& a, const std::string& b) const;
  VkCollisionPipeline& vulkan_pipeline() const;
  VkRepairPipeline& vulkan_repair_pipeline() const;
};

}  // namespace vkmotion
