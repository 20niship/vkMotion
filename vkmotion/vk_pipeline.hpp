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
};

}  // namespace vkmotion
