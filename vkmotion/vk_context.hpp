#pragma once

#include <vk_mem_alloc.h>
#include <vulkan/vulkan.h>

#include <cstdint>

namespace vkmotion {

// Headless compute-only Vulkan context: no swapchain/graphics queue.
class VkContext {
 public:
  VkContext();
  ~VkContext();
  VkContext(const VkContext&)            = delete;
  VkContext& operator=(const VkContext&) = delete;

  VkInstance instance() const { return instance_; }
  VkPhysicalDevice physical_device() const { return physical_device_; }
  VkDevice device() const { return device_; }
  VkQueue compute_queue() const { return compute_queue_; }
  uint32_t compute_queue_family() const { return compute_queue_family_; }
  VkCommandPool command_pool() const { return command_pool_; }
  VmaAllocator allocator() const { return allocator_; }

 private:
  VkInstance instance_                     = VK_NULL_HANDLE;
  VkDebugUtilsMessengerEXT debug_messenger_ = VK_NULL_HANDLE;
  VkPhysicalDevice physical_device_         = VK_NULL_HANDLE;
  VkDevice device_                          = VK_NULL_HANDLE;
  VkQueue compute_queue_                    = VK_NULL_HANDLE;
  uint32_t compute_queue_family_            = 0;
  VkCommandPool command_pool_               = VK_NULL_HANDLE;
  VmaAllocator allocator_                   = VK_NULL_HANDLE;
};

}  // namespace vkmotion
