#include "vk_buffer.hpp"

#include <cstring>
#include <stdexcept>

#include "vk_context.hpp"

namespace vkmotion {

VkStorageBuffer::VkStorageBuffer(VkContext& ctx, size_t size_bytes) : ctx_(ctx), size_(size_bytes) {
  VkBufferCreateInfo buffer_info{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
  buffer_info.size  = size_bytes;
  buffer_info.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;

  VmaAllocationCreateInfo alloc_info{};
  alloc_info.usage = VMA_MEMORY_USAGE_AUTO;
  alloc_info.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT;

  VmaAllocationInfo out_info{};
  if(vmaCreateBuffer(ctx_.allocator(), &buffer_info, &alloc_info, &buffer_, &allocation_, &out_info) != VK_SUCCESS)
    throw std::runtime_error("vmaCreateBuffer failed");
  mapped_ = out_info.pMappedData;
}

VkStorageBuffer::~VkStorageBuffer() {
  if(buffer_) vmaDestroyBuffer(ctx_.allocator(), buffer_, allocation_);
}

void VkStorageBuffer::upload(const void* data, size_t size_bytes) { std::memcpy(mapped_, data, size_bytes); }

void VkStorageBuffer::download(void* out, size_t size_bytes) const { std::memcpy(out, mapped_, size_bytes); }

VkDescriptorBufferInfo VkStorageBuffer::descriptor_info() const {
  VkDescriptorBufferInfo info{};
  info.buffer = buffer_;
  info.offset = 0;
  info.range  = size_;
  return info;
}

}  // namespace vkmotion
