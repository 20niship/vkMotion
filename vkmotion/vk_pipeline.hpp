#pragma once

#include <vulkan/vulkan.h>

#include <cstdint>
#include <string>
#include <vector>

namespace vkmotion {

class VkContext;

// One compute shader, one descriptor set of storage buffers, one push-constant block.
class VkComputePipeline {
 public:
  VkComputePipeline(VkContext& ctx, const std::string& spv_path, uint32_t num_storage_bindings, uint32_t push_constant_bytes);
  ~VkComputePipeline();
  VkComputePipeline(const VkComputePipeline&)            = delete;
  VkComputePipeline& operator=(const VkComputePipeline&) = delete;

  void update_bindings(const std::vector<VkDescriptorBufferInfo>& buffers);
  void dispatch(uint32_t group_count_x, const void* push_constants, uint32_t push_constant_bytes);

 private:
  VkContext& ctx_;
  VkDescriptorSetLayout set_layout_ = VK_NULL_HANDLE;
  VkPipelineLayout pipeline_layout_ = VK_NULL_HANDLE;
  VkShaderModule shader_module_     = VK_NULL_HANDLE;
  VkPipeline pipeline_              = VK_NULL_HANDLE;
  VkDescriptorPool descriptor_pool_ = VK_NULL_HANDLE;
  VkDescriptorSet descriptor_set_   = VK_NULL_HANDLE;
  // 都度alloc/free・create/destroyするとドライバ呼び出しがdispatchの固定コストを支配するため使い回す。
  VkCommandBuffer cmd_ = VK_NULL_HANDLE;
  VkFence fence_       = VK_NULL_HANDLE;
};

// 同じデスクリプタセット/push constantを共有する複数のcompute shaderを、1つのコマンドバッファに並べて1回のsubmitで流す。
// 反復アルゴリズムの各ラウンドをdispatchにして、ラウンド間のCPU往復を無くすため(dispatch間はGPU上のbarrierだけ)。
class VkComputeSequence {
 public:
  VkComputeSequence(VkContext& ctx, const std::vector<std::string>& spv_paths, uint32_t num_storage_bindings, uint32_t push_constant_bytes);
  ~VkComputeSequence();
  VkComputeSequence(const VkComputeSequence&)            = delete;
  VkComputeSequence& operator=(const VkComputeSequence&) = delete;

  void update_bindings(const std::vector<VkDescriptorBufferInfo>& buffers);
  // コマンドバッファを開き、デスクリプタセットとpush constantを設定する(以降のdispatchすべてに効く)
  void begin(const void* push_constants, uint32_t push_constant_bytes);
  void dispatch(size_t pipeline_index, uint32_t group_count_x);
  void dispatch_indirect(size_t pipeline_index, VkBuffer args, VkDeviceSize offset);
  void dispatch(size_t pipeline_index, uint32_t group_count_x, uint32_t group_count_y);
  // 直前までのshader書き込みを、次のdispatch(間接引数の読み出しを含む)から見えるようにする
  void barrier();
  void submit_and_wait();
  // 待たずにsubmitする。完了は fence_ready() で調べるか wait() で待つ(次のbegin()の前には必ず完了させること)
  void submit();
  bool fence_ready() const;
  void wait();

 private:
  VkContext& ctx_;
  VkDescriptorSetLayout set_layout_ = VK_NULL_HANDLE;
  VkPipelineLayout pipeline_layout_ = VK_NULL_HANDLE;
  std::vector<VkShaderModule> modules_;
  std::vector<VkPipeline> pipelines_;
  VkDescriptorPool descriptor_pool_ = VK_NULL_HANDLE;
  VkDescriptorSet descriptor_set_   = VK_NULL_HANDLE;
  VkCommandBuffer cmd_              = VK_NULL_HANDLE;
  VkFence fence_                    = VK_NULL_HANDLE;
};

}  // namespace vkmotion
