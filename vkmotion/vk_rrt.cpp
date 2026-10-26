#include "vk_rrt.hpp"

#include <algorithm>
#include <cstring>
#include <stdexcept>
#include <vkmotion/vkmotion_config.hpp>

#include "gpu_codegen.hpp"
#include "vk_buffer.hpp"
#include "vk_context.hpp"
#include "vk_pipeline.hpp"

namespace vkmotion {

namespace {

struct PushConstants {
  uint32_t unused0;
  uint32_t dof;
  uint32_t num_steps;
  uint32_t num_spheres;
  uint32_t num_clusters;
  uint32_t num_groups;
  uint32_t num_pairs;
  uint32_t num_boxes;
  uint32_t num_balls;
  uint32_t num_links;
  float margin;
  float range;
  float motion_step;
  uint32_t num_goals;
  uint32_t seed;
  uint32_t gen;
  uint32_t g0;
  uint32_t gmax;
  uint32_t grow;
  uint32_t iters;
};

// shaders/include/rrt_common.glsl と同じ値
constexpr uint32_t kMaxPoints = 128;   // MAXP
constexpr uint32_t kTreeCap   = 16384;  // CAP
constexpr uint32_t kCtlSolved = 4;
constexpr uint32_t kCtlArgs   = 8;
constexpr uint32_t kCtlWords  = 16;

template <class T>
std::unique_ptr<VkStorageBuffer> make_buffer(VkContext& ctx, const std::vector<T>& v) {
  auto buf = std::make_unique<VkStorageBuffer>(ctx, std::max<size_t>(v.size() * sizeof(T), 16), true);
  if(!v.empty()) buf->upload(v.data(), v.size() * sizeof(T));
  return buf;
}

}  // namespace

VkRrtPipeline::VkRrtPipeline(VkContext& ctx, const GpuRobotData& robot, const std::vector<float>& joint_reach, const std::vector<float>& lower, const std::vector<float>& upper)
    : ctx_(ctx), robot_(robot), reach_(joint_reach), lower_(lower), upper_(upper) {
  if(robot_.dof > static_cast<int>(kMaxDof)) throw std::runtime_error("GPU RRT supports dof <= 7");
  {
    // シェーダはサブグループ=32レーンを仮定し、サブグループ演算(vote/arithmetic/shuffle)を使う
    VkPhysicalDeviceSubgroupProperties sg{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES};
    VkPhysicalDeviceProperties2 props{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
    props.pNext = &sg;
    vkGetPhysicalDeviceProperties2(ctx_.physical_device(), &props);
    constexpr VkSubgroupFeatureFlags need = VK_SUBGROUP_FEATURE_BASIC_BIT | VK_SUBGROUP_FEATURE_VOTE_BIT | VK_SUBGROUP_FEATURE_ARITHMETIC_BIT | VK_SUBGROUP_FEATURE_SHUFFLE_BIT;
    if(sg.subgroupSize != 32 || (sg.supportedOperations & need) != need || !(sg.supportedStages & VK_SHADER_STAGE_COMPUTE_BIT))
      throw std::runtime_error("GPU RRT needs 32-wide subgroups with vote/arithmetic/shuffle");
  }
  const std::string dir = VKMOTION_SHADER_DIR;
  // ロボット専用に生成したFK+衝突判定でコンパイルしたシェーダ(無ければ汎用版)
  std::string round_spv = compile_specialized_shader("rrt_gpu.comp", robot_);
  specialized_          = !round_spv.empty();
  if(!specialized_) round_spv = dir + "/rrt_gpu.comp.spv";
  seq_          = std::make_unique<VkComputeSequence>(ctx_, std::vector<std::string>{dir + "/rrt_advance.comp.spv", round_spv}, 14, sizeof(PushConstants));
  params_       = std::make_unique<VkStorageBuffer>(ctx_, (32 + 8 * kMaxGoals) * sizeof(float), true);
  result_       = std::make_unique<VkStorageBuffer>(ctx_, (4 + kMaxPoints * 8) * sizeof(uint32_t));
  ctl_          = std::make_unique<VkStorageBuffer>(ctx_, kCtlWords * sizeof(uint32_t), true);
  tree_         = std::make_unique<VkStorageBuffer>(ctx_, 2 * kTreeCap * 16);
  steps_meta_   = make_buffer(ctx_, robot_.step_meta);
  steps_data_   = make_buffer(ctx_, robot_.step_data);
  spheres_      = make_buffer(ctx_, robot_.spheres);
  cluster_meta_ = make_buffer(ctx_, robot_.cluster_meta);
  cluster_bs_   = make_buffer(ctx_, robot_.cluster_bs);
  group_meta_   = make_buffer(ctx_, robot_.group_meta);
  group_bs_     = make_buffer(ctx_, robot_.group_bs);
  pairs_        = make_buffer(ctx_, robot_.pairs);
  set_world(GpuWorldData{});
}

VkRrtPipeline::~VkRrtPipeline() = default;

void VkRrtPipeline::set_world(const GpuWorldData& world) {
  boxes_     = make_buffer(ctx_, world.boxes);
  balls_     = make_buffer(ctx_, world.balls);
  num_boxes_ = world.num_boxes;
  num_balls_ = world.num_balls;
}

bool VkRrtPipeline::plan(const std::vector<float>& start, const std::vector<std::vector<float>>& goals, const GpuRrtSettings& s, std::vector<std::vector<float>>& path, float* margin_used) {
  const size_t dof = static_cast<size_t>(robot_.dof);
  if(goals.empty() || goals.size() > kMaxGoals || start.size() < dof) return false;

  std::vector<float> prm(32 + 8 * kMaxGoals, 0.0f);
  for(size_t j = 0; j < dof; j++) {
    prm[j]      = lower_[j];
    prm[8 + j]  = upper_[j];
    prm[16 + j] = reach_[j];
    prm[24 + j] = start[j];
  }
  for(size_t g = 0; g < goals.size(); g++)
    for(size_t j = 0; j < dof; j++) prm[32 + g * 8 + j] = goals[g][j];
  params_->upload(prm.data(), prm.size() * sizeof(float));
  const uint32_t zeros[kCtlWords] = {};
  ctl_->upload(zeros, sizeof(zeros));  // ラウンド番号を0に戻す(木のカウンタ・根はrrt_advance.compが初期化する)

  seq_->update_bindings({params_->descriptor_info(), steps_meta_->descriptor_info(), steps_data_->descriptor_info(), spheres_->descriptor_info(), cluster_meta_->descriptor_info(),
                         cluster_bs_->descriptor_info(), group_meta_->descriptor_info(), group_bs_->descriptor_info(), pairs_->descriptor_info(), boxes_->descriptor_info(),
                         balls_->descriptor_info(), result_->descriptor_info(), ctl_->descriptor_info(), tree_->descriptor_info()});

  PushConstants pc{};
  pc.dof          = static_cast<uint32_t>(robot_.dof);
  pc.num_steps    = static_cast<uint32_t>(robot_.num_steps);
  pc.num_spheres  = static_cast<uint32_t>(robot_.num_spheres);
  pc.num_clusters = static_cast<uint32_t>(robot_.num_clusters);
  pc.num_groups   = static_cast<uint32_t>(robot_.num_groups);
  pc.num_pairs    = static_cast<uint32_t>(robot_.num_pairs);
  pc.num_boxes    = static_cast<uint32_t>(num_boxes_);
  pc.num_balls    = static_cast<uint32_t>(num_balls_);
  pc.num_links    = static_cast<uint32_t>(robot_.num_links);
  pc.margin       = s.margin;
  pc.range        = s.range;
  pc.motion_step  = s.motion_step;
  pc.num_goals    = static_cast<uint32_t>(goals.size());
  pc.seed         = s.seed;
  pc.gen          = ++gen_;
  pc.g0           = std::max<uint32_t>(s.first_warps, 1);
  pc.gmax         = std::max<uint32_t>(s.max_warps, pc.g0);
  pc.grow         = 1;
  pc.iters        = std::max<uint32_t>(s.iters, 1);

  for(uint32_t submit = 0; submit < s.max_submits; submit++) {
    seq_->begin(&pc, sizeof(pc));
    for(uint32_t r = 0; r < s.rounds_per_submit; r++) {
      seq_->dispatch(0, 1);
      seq_->barrier();
      seq_->dispatch_indirect(1, ctl_->handle(), kCtlArgs * sizeof(uint32_t));
      seq_->barrier();
    }
    seq_->submit_and_wait();

    uint32_t ctl[kCtlWords];
    ctl_->download(ctl, sizeof(ctl));
    if(ctl[kCtlSolved] < gen_) continue;
    if(margin_used) std::memcpy(margin_used, &ctl[7], sizeof(float));
    std::vector<uint32_t> r(4 + kMaxPoints * 8);
    result_->download(r.data(), r.size() * sizeof(uint32_t));
    const uint32_t n = r[0];
    if(n < 2 || n > kMaxPoints) return false;
    path.assign(n, std::vector<float>(dof));
    for(uint32_t k = 0; k < n; k++)
      for(size_t j = 0; j < dof; j++) std::memcpy(&path[k][j], &r[4 + k * 8 + j], sizeof(float));
    return true;
  }
  return false;
}

}  // namespace vkmotion
