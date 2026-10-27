#include "vk_pipeline.hpp"

#include <chrono>
#include <fstream>
#include <stdexcept>
#include <vector>

#include "vk_context.hpp"

namespace vkmotion {

namespace {

std::vector<char> read_file(const std::string& path) {
  std::ifstream file(path, std::ios::ate | std::ios::binary);
  if(!file.is_open()) throw std::runtime_error("failed to open SPIR-V file: " + path);
  size_t size = static_cast<size_t>(file.tellg());
  std::vector<char> buffer(size);
  file.seekg(0);
  file.read(buffer.data(), static_cast<std::streamsize>(size));
  return buffer;
}

}  // namespace

VkComputePipeline::VkComputePipeline(VkContext& ctx, const std::string& spv_path, uint32_t num_storage_bindings, uint32_t push_constant_bytes)
    : ctx_(ctx) {
  std::vector<char> code = read_file(spv_path);

  VkShaderModuleCreateInfo module_info{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
  module_info.codeSize = code.size();
  module_info.pCode    = reinterpret_cast<const uint32_t*>(code.data());
  if(vkCreateShaderModule(ctx_.device(), &module_info, nullptr, &shader_module_) != VK_SUCCESS)
    throw std::runtime_error("vkCreateShaderModule failed");

  std::vector<VkDescriptorSetLayoutBinding> bindings(num_storage_bindings);
  for(uint32_t i = 0; i < num_storage_bindings; i++) {
    bindings[i].binding         = i;
    bindings[i].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    bindings[i].descriptorCount = 1;
    bindings[i].stageFlags      = VK_SHADER_STAGE_COMPUTE_BIT;
  }
  VkDescriptorSetLayoutCreateInfo set_layout_info{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
  set_layout_info.bindingCount = static_cast<uint32_t>(bindings.size());
  set_layout_info.pBindings    = bindings.data();
  if(vkCreateDescriptorSetLayout(ctx_.device(), &set_layout_info, nullptr, &set_layout_) != VK_SUCCESS)
    throw std::runtime_error("vkCreateDescriptorSetLayout failed");

  VkPushConstantRange push_range{};
  push_range.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
  push_range.size       = push_constant_bytes;

  VkPipelineLayoutCreateInfo layout_info{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
  layout_info.setLayoutCount         = 1;
  layout_info.pSetLayouts            = &set_layout_;
  layout_info.pushConstantRangeCount = push_constant_bytes > 0 ? 1 : 0;
  layout_info.pPushConstantRanges    = push_constant_bytes > 0 ? &push_range : nullptr;
  if(vkCreatePipelineLayout(ctx_.device(), &layout_info, nullptr, &pipeline_layout_) != VK_SUCCESS)
    throw std::runtime_error("vkCreatePipelineLayout failed");

  VkPipelineShaderStageCreateInfo stage_info{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
  stage_info.stage  = VK_SHADER_STAGE_COMPUTE_BIT;
  stage_info.module = shader_module_;
  stage_info.pName  = "main";

  VkComputePipelineCreateInfo pipeline_info{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
  pipeline_info.stage  = stage_info;
  pipeline_info.layout = pipeline_layout_;
  if(vkCreateComputePipelines(ctx_.device(), VK_NULL_HANDLE, 1, &pipeline_info, nullptr, &pipeline_) != VK_SUCCESS)
    throw std::runtime_error("vkCreateComputePipelines failed");

  VkDescriptorPoolSize pool_size{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, num_storage_bindings};
  VkDescriptorPoolCreateInfo pool_info{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
  pool_info.maxSets       = 1;
  pool_info.poolSizeCount = 1;
  pool_info.pPoolSizes    = &pool_size;
  if(vkCreateDescriptorPool(ctx_.device(), &pool_info, nullptr, &descriptor_pool_) != VK_SUCCESS)
    throw std::runtime_error("vkCreateDescriptorPool failed");

  VkDescriptorSetAllocateInfo set_alloc_info{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
  set_alloc_info.descriptorPool     = descriptor_pool_;
  set_alloc_info.descriptorSetCount = 1;
  set_alloc_info.pSetLayouts        = &set_layout_;
  if(vkAllocateDescriptorSets(ctx_.device(), &set_alloc_info, &descriptor_set_) != VK_SUCCESS)
    throw std::runtime_error("vkAllocateDescriptorSets failed");

  VkCommandBufferAllocateInfo cmd_alloc_info{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
  cmd_alloc_info.commandPool        = ctx_.command_pool();
  cmd_alloc_info.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  cmd_alloc_info.commandBufferCount = 1;
  if(vkAllocateCommandBuffers(ctx_.device(), &cmd_alloc_info, &cmd_) != VK_SUCCESS) throw std::runtime_error("vkAllocateCommandBuffers failed");

  VkFenceCreateInfo fence_info{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
  if(vkCreateFence(ctx_.device(), &fence_info, nullptr, &fence_) != VK_SUCCESS) throw std::runtime_error("vkCreateFence failed");
}

VkComputePipeline::~VkComputePipeline() {
  if(fence_) vkDestroyFence(ctx_.device(), fence_, nullptr);
  if(cmd_) vkFreeCommandBuffers(ctx_.device(), ctx_.command_pool(), 1, &cmd_);
  if(descriptor_pool_) vkDestroyDescriptorPool(ctx_.device(), descriptor_pool_, nullptr);
  if(pipeline_) vkDestroyPipeline(ctx_.device(), pipeline_, nullptr);
  if(pipeline_layout_) vkDestroyPipelineLayout(ctx_.device(), pipeline_layout_, nullptr);
  if(set_layout_) vkDestroyDescriptorSetLayout(ctx_.device(), set_layout_, nullptr);
  if(shader_module_) vkDestroyShaderModule(ctx_.device(), shader_module_, nullptr);
}

void VkComputePipeline::update_bindings(const std::vector<VkDescriptorBufferInfo>& buffers) {
  std::vector<VkWriteDescriptorSet> writes(buffers.size());
  for(size_t i = 0; i < buffers.size(); i++) {
    writes[i].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[i].dstSet          = descriptor_set_;
    writes[i].dstBinding      = static_cast<uint32_t>(i);
    writes[i].descriptorCount = 1;
    writes[i].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    writes[i].pBufferInfo     = &buffers[i];
  }
  vkUpdateDescriptorSets(ctx_.device(), static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);
}

void VkComputePipeline::dispatch(uint32_t group_count_x, const void* push_constants, uint32_t push_constant_bytes) {
  vkResetCommandBuffer(cmd_, 0);

  VkCommandBufferBeginInfo begin_info{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
  begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
  vkBeginCommandBuffer(cmd_, &begin_info);

  vkCmdBindPipeline(cmd_, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_);
  vkCmdBindDescriptorSets(cmd_, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_layout_, 0, 1, &descriptor_set_, 0, nullptr);
  if(push_constant_bytes > 0) vkCmdPushConstants(cmd_, pipeline_layout_, VK_SHADER_STAGE_COMPUTE_BIT, 0, push_constant_bytes, push_constants);
  vkCmdDispatch(cmd_, group_count_x, 1, 1);

  vkEndCommandBuffer(cmd_);

  vkResetFences(ctx_.device(), 1, &fence_);

  VkSubmitInfo submit_info{VK_STRUCTURE_TYPE_SUBMIT_INFO};
  submit_info.commandBufferCount = 1;
  submit_info.pCommandBuffers    = &cmd_;
  vkQueueSubmit(ctx_.compute_queue(), 1, &submit_info, fence_);
  vkWaitForFences(ctx_.device(), 1, &fence_, VK_TRUE, UINT64_MAX);
}

VkComputeSequence::VkComputeSequence(VkContext& ctx, const std::vector<std::string>& spv_paths, uint32_t num_storage_bindings, uint32_t push_constant_bytes) : ctx_(ctx) {
  std::vector<VkDescriptorSetLayoutBinding> bindings(num_storage_bindings);
  for(uint32_t i = 0; i < num_storage_bindings; i++) {
    bindings[i].binding         = i;
    bindings[i].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    bindings[i].descriptorCount = 1;
    bindings[i].stageFlags      = VK_SHADER_STAGE_COMPUTE_BIT;
  }
  VkDescriptorSetLayoutCreateInfo set_layout_info{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
  set_layout_info.bindingCount = static_cast<uint32_t>(bindings.size());
  set_layout_info.pBindings    = bindings.data();
  if(vkCreateDescriptorSetLayout(ctx_.device(), &set_layout_info, nullptr, &set_layout_) != VK_SUCCESS) throw std::runtime_error("vkCreateDescriptorSetLayout failed");

  VkPushConstantRange push_range{};
  push_range.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
  push_range.size       = push_constant_bytes;
  VkPipelineLayoutCreateInfo layout_info{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
  layout_info.setLayoutCount         = 1;
  layout_info.pSetLayouts            = &set_layout_;
  layout_info.pushConstantRangeCount = push_constant_bytes > 0 ? 1 : 0;
  layout_info.pPushConstantRanges    = push_constant_bytes > 0 ? &push_range : nullptr;
  if(vkCreatePipelineLayout(ctx_.device(), &layout_info, nullptr, &pipeline_layout_) != VK_SUCCESS) throw std::runtime_error("vkCreatePipelineLayout failed");

  for(const auto& path : spv_paths) {
    std::vector<char> code = read_file(path);
    VkShaderModuleCreateInfo module_info{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    module_info.codeSize = code.size();
    module_info.pCode    = reinterpret_cast<const uint32_t*>(code.data());
    VkShaderModule module = VK_NULL_HANDLE;
    if(vkCreateShaderModule(ctx_.device(), &module_info, nullptr, &module) != VK_SUCCESS) throw std::runtime_error("vkCreateShaderModule failed");
    modules_.push_back(module);
    VkPipelineShaderStageCreateInfo stage_info{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
    stage_info.stage  = VK_SHADER_STAGE_COMPUTE_BIT;
    stage_info.module = module;
    stage_info.pName  = "main";
    VkComputePipelineCreateInfo pipeline_info{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    pipeline_info.stage  = stage_info;
    pipeline_info.layout = pipeline_layout_;
    VkPipeline pipeline  = VK_NULL_HANDLE;
    if(vkCreateComputePipelines(ctx_.device(), VK_NULL_HANDLE, 1, &pipeline_info, nullptr, &pipeline) != VK_SUCCESS) throw std::runtime_error("vkCreateComputePipelines failed");
    pipelines_.push_back(pipeline);
  }

  VkDescriptorPoolSize pool_size{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, num_storage_bindings};
  VkDescriptorPoolCreateInfo pool_info{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
  pool_info.maxSets       = 1;
  pool_info.poolSizeCount = 1;
  pool_info.pPoolSizes    = &pool_size;
  if(vkCreateDescriptorPool(ctx_.device(), &pool_info, nullptr, &descriptor_pool_) != VK_SUCCESS) throw std::runtime_error("vkCreateDescriptorPool failed");
  VkDescriptorSetAllocateInfo set_alloc_info{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
  set_alloc_info.descriptorPool     = descriptor_pool_;
  set_alloc_info.descriptorSetCount = 1;
  set_alloc_info.pSetLayouts        = &set_layout_;
  if(vkAllocateDescriptorSets(ctx_.device(), &set_alloc_info, &descriptor_set_) != VK_SUCCESS) throw std::runtime_error("vkAllocateDescriptorSets failed");

  VkCommandBufferAllocateInfo cmd_alloc_info{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
  cmd_alloc_info.commandPool        = ctx_.command_pool();
  cmd_alloc_info.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  cmd_alloc_info.commandBufferCount = 1;
  if(vkAllocateCommandBuffers(ctx_.device(), &cmd_alloc_info, &cmd_) != VK_SUCCESS) throw std::runtime_error("vkAllocateCommandBuffers failed");
  VkFenceCreateInfo fence_info{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
  if(vkCreateFence(ctx_.device(), &fence_info, nullptr, &fence_) != VK_SUCCESS) throw std::runtime_error("vkCreateFence failed");
}

VkComputeSequence::~VkComputeSequence() {
  if(fence_) vkDestroyFence(ctx_.device(), fence_, nullptr);
  if(cmd_) vkFreeCommandBuffers(ctx_.device(), ctx_.command_pool(), 1, &cmd_);
  if(descriptor_pool_) vkDestroyDescriptorPool(ctx_.device(), descriptor_pool_, nullptr);
  for(VkPipeline p : pipelines_) vkDestroyPipeline(ctx_.device(), p, nullptr);
  for(VkShaderModule m : modules_) vkDestroyShaderModule(ctx_.device(), m, nullptr);
  if(pipeline_layout_) vkDestroyPipelineLayout(ctx_.device(), pipeline_layout_, nullptr);
  if(set_layout_) vkDestroyDescriptorSetLayout(ctx_.device(), set_layout_, nullptr);
}

void VkComputeSequence::update_bindings(const std::vector<VkDescriptorBufferInfo>& buffers) {
  std::vector<VkWriteDescriptorSet> writes(buffers.size());
  for(size_t i = 0; i < buffers.size(); i++) {
    writes[i].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[i].dstSet          = descriptor_set_;
    writes[i].dstBinding      = static_cast<uint32_t>(i);
    writes[i].descriptorCount = 1;
    writes[i].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    writes[i].pBufferInfo     = &buffers[i];
  }
  vkUpdateDescriptorSets(ctx_.device(), static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);
}

void VkComputeSequence::begin(const void* push_constants, uint32_t push_constant_bytes) {
  vkResetCommandBuffer(cmd_, 0);
  VkCommandBufferBeginInfo begin_info{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
  begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
  vkBeginCommandBuffer(cmd_, &begin_info);
  vkCmdBindDescriptorSets(cmd_, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_layout_, 0, 1, &descriptor_set_, 0, nullptr);
  if(push_constant_bytes > 0) vkCmdPushConstants(cmd_, pipeline_layout_, VK_SHADER_STAGE_COMPUTE_BIT, 0, push_constant_bytes, push_constants);
}

void VkComputeSequence::dispatch(size_t pipeline_index, uint32_t group_count_x) {
  vkCmdBindPipeline(cmd_, VK_PIPELINE_BIND_POINT_COMPUTE, pipelines_.at(pipeline_index));
  vkCmdDispatch(cmd_, group_count_x, 1, 1);
}

void VkComputeSequence::dispatch(size_t pipeline_index, uint32_t group_count_x, uint32_t group_count_y) {
  vkCmdBindPipeline(cmd_, VK_PIPELINE_BIND_POINT_COMPUTE, pipelines_.at(pipeline_index));
  vkCmdDispatch(cmd_, group_count_x, group_count_y, 1);
}

void VkComputeSequence::dispatch_indirect(size_t pipeline_index, VkBuffer args, VkDeviceSize offset) {
  vkCmdBindPipeline(cmd_, VK_PIPELINE_BIND_POINT_COMPUTE, pipelines_.at(pipeline_index));
  vkCmdDispatchIndirect(cmd_, args, offset);
}

void VkComputeSequence::barrier() {
  VkMemoryBarrier mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
  mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
  mb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_INDIRECT_COMMAND_READ_BIT;
  vkCmdPipelineBarrier(cmd_, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT, 0, 1, &mb, 0, nullptr, 0, nullptr);
}

void VkComputeSequence::submit_and_wait() {
  vkEndCommandBuffer(cmd_);
  vkResetFences(ctx_.device(), 1, &fence_);
  VkSubmitInfo submit_info{VK_STRUCTURE_TYPE_SUBMIT_INFO};
  submit_info.commandBufferCount = 1;
  submit_info.pCommandBuffers    = &cmd_;
  vkQueueSubmit(ctx_.compute_queue(), 1, &submit_info, fence_);
  // 数msで終わる処理なので、まず短時間スピンして完了を拾う(ブロッキング待ちはスレッド起床に数十〜百µsかかる)。長引けば通常の待ちに切り替える。
  const auto spin_until = std::chrono::steady_clock::now() + std::chrono::milliseconds(3);
  while(vkGetFenceStatus(ctx_.device(), fence_) == VK_NOT_READY && std::chrono::steady_clock::now() < spin_until) {
  }
  vkWaitForFences(ctx_.device(), 1, &fence_, VK_TRUE, UINT64_MAX);
}

}  // namespace vkmotion
