#pragma once

#include <vk_mem_alloc.h>
#include <vulkan/vulkan.h>

#include <cstddef>

namespace vkmotion {

class VkContext;

// Host-visible + host-coherent storage buffer (VMA-backed, persistently mapped).
class VkStorageBuffer {
 public:
  // prefer_device=true: 可能ならGPUのローカルメモリ(BAR経由でホストから書ける領域)に置く。GPUが頻繁にアトミック/読み書きする小さなバッファ用。
  VkStorageBuffer(VkContext& ctx, size_t size_bytes, bool prefer_device = false);
  ~VkStorageBuffer();
  VkStorageBuffer(const VkStorageBuffer&)            = delete;
  VkStorageBuffer& operator=(const VkStorageBuffer&) = delete;

  void upload(const void* data, size_t size_bytes, size_t offset_bytes = 0);
  void download(void* out, size_t size_bytes, size_t offset_bytes = 0) const;

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
