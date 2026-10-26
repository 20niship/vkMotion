#include "vk_validate.hpp"

#include <algorithm>
#include <cstring>
#include <vkmotion/vkmotion_config.hpp>

#include "gpu_codegen.hpp"
#include "vk_buffer.hpp"
#include "vk_context.hpp"
#include "vk_pipeline.hpp"

namespace vkmotion {

namespace {

struct PushConstants {
  uint32_t num_configs;
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
  uint32_t pad0;
};

constexpr uint32_t kLocalSize = 64;

template <class T>
std::unique_ptr<VkStorageBuffer> make_buffer(VkContext& ctx, const std::vector<T>& v) {
  auto buf = std::make_unique<VkStorageBuffer>(ctx, std::max<size_t>(v.size() * sizeof(T), 16));
  if(!v.empty()) buf->upload(v.data(), v.size() * sizeof(T));
  return buf;
}

void ensure(VkContext& ctx, std::unique_ptr<VkStorageBuffer>& buf, size_t bytes) {
  if(!buf || buf->size() < bytes) buf = std::make_unique<VkStorageBuffer>(ctx, std::max<size_t>(bytes, 16));
}

}  // namespace

VkValidatePipeline::VkValidatePipeline(VkContext& ctx, const GpuRobotData& robot) : ctx_(ctx), robot_(robot) {
  std::string spv = compile_specialized_shader("robot_validate.comp", robot_);
  if(spv.empty()) spv = std::string(VKMOTION_SHADER_DIR) + "/robot_validate.comp.spv";
  pipeline_      = std::make_unique<VkComputePipeline>(ctx_, spv, 12, sizeof(PushConstants));
  steps_meta_    = make_buffer(ctx_, robot_.step_meta);
  steps_data_    = make_buffer(ctx_, robot_.step_data);
  spheres_       = make_buffer(ctx_, robot_.spheres);
  cluster_meta_  = make_buffer(ctx_, robot_.cluster_meta);
  cluster_bs_    = make_buffer(ctx_, robot_.cluster_bs);
  group_meta_    = make_buffer(ctx_, robot_.group_meta);
  group_bs_      = make_buffer(ctx_, robot_.group_bs);
  pairs_         = make_buffer(ctx_, robot_.pairs);
  set_world(GpuWorldData{});
}

VkValidatePipeline::~VkValidatePipeline() = default;

void VkValidatePipeline::set_world(const GpuWorldData& world) {
  boxes_      = make_buffer(ctx_, world.boxes);
  balls_      = make_buffer(ctx_, world.balls);
  num_boxes_  = world.num_boxes;
  num_balls_  = world.num_balls;
}

void VkValidatePipeline::validate(const float* q, size_t n, float margin, uint8_t* out) {
  if(n == 0) return;
  ensure(ctx_, cfg_, n * static_cast<size_t>(robot_.dof) * sizeof(float));
  ensure(ctx_, out_, n * sizeof(uint32_t));
  cfg_->upload(q, n * static_cast<size_t>(robot_.dof) * sizeof(float));

  pipeline_->update_bindings({cfg_->descriptor_info(), steps_meta_->descriptor_info(), steps_data_->descriptor_info(), spheres_->descriptor_info(),
                              cluster_meta_->descriptor_info(), cluster_bs_->descriptor_info(), group_meta_->descriptor_info(), group_bs_->descriptor_info(),
                              pairs_->descriptor_info(), boxes_->descriptor_info(), balls_->descriptor_info(), out_->descriptor_info()});

  PushConstants pc{};
  pc.num_configs  = static_cast<uint32_t>(n);
  pc.dof          = static_cast<uint32_t>(robot_.dof);
  pc.num_steps    = static_cast<uint32_t>(robot_.num_steps);
  pc.num_spheres  = static_cast<uint32_t>(robot_.num_spheres);
  pc.num_clusters = static_cast<uint32_t>(robot_.num_clusters);
  pc.num_groups   = static_cast<uint32_t>(robot_.num_groups);
  pc.num_pairs    = static_cast<uint32_t>(robot_.num_pairs);
  pc.num_boxes    = static_cast<uint32_t>(num_boxes_);
  pc.num_balls    = static_cast<uint32_t>(num_balls_);
  pc.num_links    = static_cast<uint32_t>(robot_.num_links);
  pc.margin       = margin;
  pipeline_->dispatch(static_cast<uint32_t>((n + kLocalSize - 1) / kLocalSize), &pc, sizeof(pc));

  std::vector<uint32_t> flags(n);
  out_->download(flags.data(), n * sizeof(uint32_t));
  for(size_t i = 0; i < n; i++) out[i] = static_cast<uint8_t>(flags[i]);
}

}  // namespace vkmotion
