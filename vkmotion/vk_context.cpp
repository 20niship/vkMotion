#define VMA_IMPLEMENTATION
#include "vk_context.hpp"

#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace vkmotion {

namespace {

bool has_validation_layer() {
  uint32_t count = 0;
  vkEnumerateInstanceLayerProperties(&count, nullptr);
  std::vector<VkLayerProperties> layers(count);
  vkEnumerateInstanceLayerProperties(&count, layers.data());
  for(const auto& l : layers)
    if(std::strcmp(l.layerName, "VK_LAYER_KHRONOS_validation") == 0) return true;
  return false;
}

VKAPI_ATTR VkBool32 VKAPI_CALL debug_callback(VkDebugUtilsMessageSeverityFlagBitsEXT severity, VkDebugUtilsMessageTypeFlagsEXT,
                                               const VkDebugUtilsMessengerCallbackDataEXT* data, void*) {
  if(severity >= VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT) std::fprintf(stderr, "[vulkan] %s\n", data->pMessage);
  return VK_FALSE;
}

}  // namespace

VkContext::VkContext() {
  bool validation = false;
#ifdef VKMOTION_ENABLE_VALIDATION
  validation = has_validation_layer();
#endif

  VkApplicationInfo app_info{VK_STRUCTURE_TYPE_APPLICATION_INFO};
  app_info.pApplicationName = "vkMotion";
  app_info.apiVersion       = VK_API_VERSION_1_1;

  std::vector<const char*> layers;
  if(validation) layers.push_back("VK_LAYER_KHRONOS_validation");
  std::vector<const char*> extensions;
  if(validation) extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);

  VkInstanceCreateInfo instance_info{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
  instance_info.pApplicationInfo        = &app_info;
  instance_info.enabledLayerCount       = static_cast<uint32_t>(layers.size());
  instance_info.ppEnabledLayerNames     = layers.data();
  instance_info.enabledExtensionCount   = static_cast<uint32_t>(extensions.size());
  instance_info.ppEnabledExtensionNames = extensions.data();

  if(vkCreateInstance(&instance_info, nullptr, &instance_) != VK_SUCCESS) throw std::runtime_error("vkCreateInstance failed");

  if(validation) {
    VkDebugUtilsMessengerCreateInfoEXT dbg_info{VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT};
    dbg_info.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
    dbg_info.messageType     = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
                            VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
    dbg_info.pfnUserCallback = debug_callback;

    auto create_fn = reinterpret_cast<PFN_vkCreateDebugUtilsMessengerEXT>(vkGetInstanceProcAddr(instance_, "vkCreateDebugUtilsMessengerEXT"));
    if(create_fn) create_fn(instance_, &dbg_info, nullptr, &debug_messenger_);
  }

  uint32_t device_count = 0;
  vkEnumeratePhysicalDevices(instance_, &device_count, nullptr);
  if(device_count == 0) throw std::runtime_error("no Vulkan physical devices found");
  std::vector<VkPhysicalDevice> devices(device_count);
  vkEnumeratePhysicalDevices(instance_, &device_count, devices.data());

  for(VkPhysicalDevice candidate : devices) {
    uint32_t family_count = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(candidate, &family_count, nullptr);
    std::vector<VkQueueFamilyProperties> families(family_count);
    vkGetPhysicalDeviceQueueFamilyProperties(candidate, &family_count, families.data());
    for(uint32_t i = 0; i < family_count; i++) {
      if(families[i].queueFlags & VK_QUEUE_COMPUTE_BIT) {
        physical_device_       = candidate;
        compute_queue_family_  = i;
        break;
      }
    }
    if(physical_device_ != VK_NULL_HANDLE) break;
  }
  if(physical_device_ == VK_NULL_HANDLE) throw std::runtime_error("no Vulkan device with a compute queue found");

  float queue_priority = 1.0f;
  VkDeviceQueueCreateInfo queue_info{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
  queue_info.queueFamilyIndex = compute_queue_family_;
  queue_info.queueCount       = 1;
  queue_info.pQueuePriorities = &queue_priority;

  VkDeviceCreateInfo device_info{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
  device_info.queueCreateInfoCount = 1;
  device_info.pQueueCreateInfos    = &queue_info;

  if(vkCreateDevice(physical_device_, &device_info, nullptr, &device_) != VK_SUCCESS) throw std::runtime_error("vkCreateDevice failed");
  vkGetDeviceQueue(device_, compute_queue_family_, 0, &compute_queue_);

  VkCommandPoolCreateInfo pool_info{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
  pool_info.flags            = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
  pool_info.queueFamilyIndex = compute_queue_family_;
  if(vkCreateCommandPool(device_, &pool_info, nullptr, &command_pool_) != VK_SUCCESS) throw std::runtime_error("vkCreateCommandPool failed");

  VmaAllocatorCreateInfo alloc_info{};
  alloc_info.physicalDevice   = physical_device_;
  alloc_info.device           = device_;
  alloc_info.instance         = instance_;
  alloc_info.vulkanApiVersion = VK_API_VERSION_1_1;
  if(vmaCreateAllocator(&alloc_info, &allocator_) != VK_SUCCESS) throw std::runtime_error("vmaCreateAllocator failed");
}

VkContext::~VkContext() {
  if(allocator_) vmaDestroyAllocator(allocator_);
  if(command_pool_) vkDestroyCommandPool(device_, command_pool_, nullptr);
  if(device_) vkDestroyDevice(device_, nullptr);
  if(debug_messenger_) {
    auto destroy_fn = reinterpret_cast<PFN_vkDestroyDebugUtilsMessengerEXT>(vkGetInstanceProcAddr(instance_, "vkDestroyDebugUtilsMessengerEXT"));
    if(destroy_fn) destroy_fn(instance_, debug_messenger_, nullptr);
  }
  if(instance_) vkDestroyInstance(instance_, nullptr);
}

}  // namespace vkmotion
