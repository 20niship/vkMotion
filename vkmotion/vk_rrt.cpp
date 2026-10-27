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
  uint32_t tree_cap;
};

// shaders/include/rrt_common.glsl と同じ値
constexpr uint32_t kMaxPoints = 128;   // MAXP
constexpr uint32_t kTreeBudgetNodes = 2u << 20;  // 全問題の木(2本ずつ)のノード数の予算(16B/ノード -> 32MB)
constexpr uint32_t kCtlSolved = 4;
constexpr uint32_t kCtlMargin = 7;
constexpr uint32_t kCtlWords  = 16;   // CTL_STRIDE
constexpr uint32_t kPrmStride = 104;  // PRM_STRIDE
constexpr uint32_t kResStride = 4 + kMaxPoints * 8;

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
  steps_meta_   = make_buffer(ctx_, robot_.step_meta);
  steps_data_   = make_buffer(ctx_, robot_.step_data);
  spheres_      = make_buffer(ctx_, robot_.spheres);
  cluster_meta_ = make_buffer(ctx_, robot_.cluster_meta);
  cluster_bs_   = make_buffer(ctx_, robot_.cluster_bs);
  group_meta_   = make_buffer(ctx_, robot_.group_meta);
  group_bs_     = make_buffer(ctx_, robot_.group_bs);
  pairs_        = make_buffer(ctx_, robot_.pairs);
  tree_cap_ = 4096;
  ensure_capacity(1);
}

VkRrtPipeline::~VkRrtPipeline() = default;

void VkRrtPipeline::ensure_capacity(size_t problems) {
  if(problems <= capacity_) return;
  capacity_ = problems <= 1 ? 1 : ((problems + 7) / 8) * 8;
  params_   = std::make_unique<VkStorageBuffer>(ctx_, capacity_ * kPrmStride * sizeof(float), true);
  result_   = std::make_unique<VkStorageBuffer>(ctx_, capacity_ * kResStride * sizeof(uint32_t));
  ctl_      = std::make_unique<VkStorageBuffer>(ctx_, capacity_ * kCtlWords * sizeof(uint32_t), true);
  tree_     = std::make_unique<VkStorageBuffer>(ctx_, capacity_ * 2 * tree_cap_ * 16, true);
  boxes_    = std::make_unique<VkStorageBuffer>(ctx_, capacity_ * kMaxBoxes * 24 * sizeof(float), true);
  balls_    = std::make_unique<VkStorageBuffer>(ctx_, capacity_ * kMaxBalls * 8 * sizeof(float), true);
}

void VkRrtPipeline::set_world(const GpuWorldData& world) { world_ = world; }

bool VkRrtPipeline::plan(const std::vector<float>& start, const std::vector<std::vector<float>>& goals, const GpuRrtSettings& s, std::vector<std::vector<float>>& path, float* margin_used) {
  GpuRrtProblem prob;
  prob.start = start;
  prob.goals = goals;
  prob.world = &world_;
  std::vector<GpuRrtResult> res;
  plan_batch({prob}, s, res);
  if(!res[0].solved) return false;
  path = std::move(res[0].path);
  if(margin_used) *margin_used = res[0].margin;
  return true;
}

// 問題のアップロードとpush constantの用意(まだsubmitしない)。有効な問題が1つも無ければfalse。
bool VkRrtPipeline::prepare(const std::vector<GpuRrtProblem>& problems, const GpuRrtSettings& s, std::vector<uint8_t>& valid) {
  const size_t dof = static_cast<size_t>(robot_.dof);
  const size_t n   = problems.size();
  // 木の上限ノード数: 問題数が少ないほど大きく(最大4096。これより大きいと最近傍探索が重くなり、リスタートで木を小さく保つ方が速い)、多いと小さく(最小1024)して全体を予算に収める
  uint32_t cap = 4096;
  while(cap > 1024 && static_cast<uint64_t>(n) * 2 * cap > kTreeBudgetNodes) cap /= 2;
  if(cap != tree_cap_) {
    tree_cap_ = cap;
    capacity_ = 0;  // 木のバッファを作り直す
  }
  ensure_capacity(n);

  // 問題ごとのパラメータ・世界。入力が不正(ゴール0個/多すぎ、箱が多すぎ)な問題は最初から解けなかったことにする(そのctlは解けた扱いにして走らせない)
  std::vector<float> prm(n * kPrmStride, 0.0f), boxes(n * kMaxBoxes * 24, 0.0f), balls(n * kMaxBalls * 8, 0.0f);
  std::vector<uint32_t> ctl0(n * kCtlWords, 0);
  valid.assign(n, 0);
  auto put_bits = [](float& f, uint32_t v) { std::memcpy(&f, &v, sizeof(v)); };
  const uint32_t gen = ++gen_;
  size_t active      = 0;
  for(size_t i = 0; i < n; i++) {
    const GpuRrtProblem& pr = problems[i];
    const bool ok = !pr.goals.empty() && pr.goals.size() <= kMaxGoals && pr.start.size() >= dof && pr.world && static_cast<size_t>(pr.world->num_boxes) <= kMaxBoxes &&
                    static_cast<size_t>(pr.world->num_balls) <= kMaxBalls;
    if(!ok) {
      ctl0[i * kCtlWords + kCtlSolved] = 0xFFFFFFFFu;  // 解けた扱い(走らせない)。結果のsolvedはfalseのまま
      continue;
    }
    valid[i] = 1;
    active++;
    float* P = &prm[i * kPrmStride];
    for(size_t j = 0; j < dof; j++) {
      P[j]      = lower_[j];
      P[8 + j]  = upper_[j];
      P[16 + j] = reach_[j];
      P[24 + j] = pr.start[j];
    }
    for(size_t g = 0; g < pr.goals.size(); g++)
      for(size_t j = 0; j < dof; j++) P[32 + g * 8 + j] = pr.goals[g][j];
    put_bits(P[96], static_cast<uint32_t>(pr.goals.size()));
    put_bits(P[97], static_cast<uint32_t>(pr.world->num_boxes));
    put_bits(P[98], static_cast<uint32_t>(pr.world->num_balls));
    put_bits(P[99], static_cast<uint32_t>(i * 2654435761u));
    std::copy(pr.world->boxes.begin(), pr.world->boxes.end(), boxes.begin() + static_cast<std::ptrdiff_t>(i * kMaxBoxes * 24));
    std::copy(pr.world->balls.begin(), pr.world->balls.end(), balls.begin() + static_cast<std::ptrdiff_t>(i * kMaxBalls * 8));
  }
  if(active == 0) return false;
  params_->upload(prm.data(), prm.size() * sizeof(float));
  boxes_->upload(boxes.data(), boxes.size() * sizeof(float));
  balls_->upload(balls.data(), balls.size() * sizeof(float));
  ctl_->upload(ctl0.data(), ctl0.size() * sizeof(uint32_t));  // ラウンド番号を0に戻す(木のカウンタ・根はrrt_advance.compが初期化する)

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
  pc.num_links    = static_cast<uint32_t>(robot_.num_links);
  pc.margin       = s.margin;
  pc.range        = s.range;
  pc.motion_step  = s.motion_step;
  pc.seed         = s.seed;
  pc.gen          = gen;
  pc.g0           = std::max<uint32_t>(s.first_warps, 1);
  pc.gmax         = std::max<uint32_t>(s.max_warps, pc.g0);
  pc.grow         = 1;
  pc.iters        = std::max<uint32_t>(s.iters, 1);
  pc.tree_cap     = tree_cap_;
  job_pc_.assign(reinterpret_cast<const uint8_t*>(&pc), reinterpret_cast<const uint8_t*>(&pc) + sizeof(pc));
  job_gen_        = gen;
  job_chunks_     = 0;
  job_max_chunks_ = s.max_submits;
  job_rounds_     = s.rounds_per_submit;
  return true;
}

// 区切り(rounds_per_submit ラウンド)を1つ積んで待たずにsubmitする。n は直近のprepareの問題数(ctl_の先頭から数える)
void VkRrtPipeline::submit_chunk() {
  PushConstants pc;
  std::memcpy(&pc, job_pc_.data(), sizeof(pc));
  seq_->begin(&pc, sizeof(pc));
  for(uint32_t r = 0; r < job_rounds_; r++) {
    seq_->dispatch(0, job_n_);  // 問題ごとにラウンド間処理
    seq_->barrier();
    seq_->dispatch(1, pc.gmax, job_n_);  // (ワープ, 問題)
    seq_->barrier();
  }
  seq_->submit();
  job_active_ = true;
  job_chunks_++;
}

void VkRrtPipeline::wait_job() {
  if(job_active_) {
    seq_->wait();
    job_active_ = false;
  }
}

void VkRrtPipeline::plan_batch(const std::vector<GpuRrtProblem>& problems, const GpuRrtSettings& s, std::vector<GpuRrtResult>& results) {
  const size_t dof = static_cast<size_t>(robot_.dof);
  const size_t n   = problems.size();
  results.assign(n, GpuRrtResult{});
  if(n == 0 || n > kMaxBatch) return;
  wait_job();
  job_open_ = false;
  std::vector<uint8_t> valid;
  if(!prepare(problems, s, valid)) return;
  job_n_ = static_cast<uint32_t>(n);
  const uint32_t gen = job_gen_;

  std::vector<uint32_t> ctl(n * kCtlWords);
  while(job_chunks_ < job_max_chunks_) {
    submit_chunk();
    wait_job();
    ctl_->download(ctl.data(), ctl.size() * sizeof(uint32_t));
    size_t unsolved = 0;
    for(size_t i = 0; i < n; i++)
      if(valid[i] && ctl[i * kCtlWords + kCtlSolved] < gen) unsolved++;
    if(unsolved == 0) break;
  }

  std::vector<uint32_t> res(n * kResStride);
  result_->download(res.data(), res.size() * sizeof(uint32_t));
  for(size_t i = 0; i < n; i++) {
    if(!valid[i] || ctl[i * kCtlWords + kCtlSolved] < gen) continue;
    const uint32_t* r = &res[i * kResStride];
    const uint32_t k  = r[0];
    if(k < 2 || k > kMaxPoints) continue;
    GpuRrtResult& out = results[i];
    out.path.assign(k, std::vector<float>(dof));
    for(uint32_t a = 0; a < k; a++)
      for(size_t j = 0; j < dof; j++) std::memcpy(&out.path[a][j], &r[4 + a * 8 + j], sizeof(float));
    std::memcpy(&out.margin, &ctl[i * kCtlWords + kCtlMargin], sizeof(float));
    out.solved = true;
  }
}

bool VkRrtPipeline::async_begin(const GpuRrtProblem& problem, const GpuRrtSettings& s) {
  wait_job();
  job_open_ = false;
  std::vector<uint8_t> valid;
  if(!prepare({problem}, s, valid)) return false;
  job_n_    = 1;
  job_open_ = true;
  submit_chunk();
  return true;
}

VkRrtPipeline::Poll VkRrtPipeline::async_poll() {
  if(!job_open_) return Poll::Failed;
  if(job_active_) {
    if(!seq_->fence_ready()) return Poll::Running;
    job_active_ = false;
  }
  uint32_t ctl[kCtlWords];
  ctl_->download(ctl, sizeof(ctl));
  if(ctl[kCtlSolved] >= job_gen_) {
    job_open_ = false;
    return Poll::Solved;
  }
  if(job_chunks_ >= job_max_chunks_) {
    job_open_ = false;
    return Poll::Failed;
  }
  submit_chunk();  // 未解決なら続きを積む
  return Poll::Running;
}

bool VkRrtPipeline::async_take(GpuRrtResult& out) {
  out = GpuRrtResult{};
  wait_job();
  const size_t dof = static_cast<size_t>(robot_.dof);
  uint32_t ctl[kCtlWords];
  ctl_->download(ctl, sizeof(ctl));
  if(ctl[kCtlSolved] != job_gen_) return false;  // cancel(0xFFFFFFFF)された・他の世代
  std::vector<uint32_t> r(kResStride);
  result_->download(r.data(), r.size() * sizeof(uint32_t));
  const uint32_t k = r[0];
  if(k < 2 || k > kMaxPoints) return false;
  out.path.assign(k, std::vector<float>(dof));
  for(uint32_t a = 0; a < k; a++)
    for(size_t j = 0; j < dof; j++) std::memcpy(&out.path[a][j], &r[4 + a * 8 + j], sizeof(float));
  std::memcpy(&out.margin, &ctl[kCtlMargin], sizeof(float));
  out.solved = true;
  return true;
}

void VkRrtPipeline::async_cancel() {
  if(!job_open_) return;
  job_open_ = false;
  const uint32_t all = 0xFFFFFFFFu;
  ctl_->upload(&all, sizeof(all), kCtlSolved * sizeof(uint32_t));  // 以降のラウンドはすぐ終わる
}

}  // namespace vkmotion
