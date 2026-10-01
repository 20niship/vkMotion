#pragma once

#include <vk_mem_alloc.h>
#include <vulkan/vulkan.h>

#include <cstddef>

namespace vkmotion {

class VkContext;

// Host-visible + host-coherent storage buffer (VMA-backed, persistently mapped).
class VkStorageBuffer {
 public:
  VkStorageBuffer(VkContext& ctx, size_t size_bytes);
  ~VkStorageBuffer();
  VkStorageBuffer(const VkStorageBuffer&)            = delete;
  VkStorageBuffer& operator=(const VkStorageBuffer&) = delete;

  void upload(const void* data, size_t size_bytes);
  void download(void* out, size_t size_bytes) const;

  VkBuffer handle() const { return buffer_; }
  VkDescriptorBufferInfo descriptor_info() const;
  size_t size() const { return size_; }

 private:
  VkContext& ctx_;
  VkBuffer buffer_          = VK_NULL_HANDLE;
  VmaAllocation allocation_ = VK_NULL_HANDLE;
  void* mapped_             = nullptr;
  size_t size_              = 0;
};

}  // namespace vkmotion
