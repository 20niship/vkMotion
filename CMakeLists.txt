cmake_minimum_required(VERSION 3.20)
project(vkMotion LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(VKMOTION_BUILD_TESTS "Build doctest-based unit tests" OFF)
option(VKMOTION_BUILD_EXAMPLES "Build example tools (stress-test runner, etc.)" OFF)
option(VKMOTION_ENABLE_VALIDATION "Enable Vulkan validation layers by default" ON)

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

include(cmake/Dependencies.cmake)

find_package(Vulkan REQUIRED)

# ---- shader compilation (glslc .comp -> .spv) --------------------------------
find_program(GLSLC_EXECUTABLE glslc REQUIRED)

file(GLOB VKMOTION_SHADER_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/shaders/*.comp)
set(VKMOTION_SHADER_BINARY_DIR ${CMAKE_BINARY_DIR}/shaders)
file(MAKE_DIRECTORY ${VKMOTION_SHADER_BINARY_DIR})

set(VKMOTION_SPV_FILES "")
foreach(shader_src ${VKMOTION_SHADER_SOURCES})
  get_filename_component(shader_name ${shader_src} NAME)
  set(spv_out ${VKMOTION_SHADER_BINARY_DIR}/${shader_name}.spv)
  add_custom_command(
    OUTPUT ${spv_out}
    COMMAND ${GLSLC_EXECUTABLE} ${shader_src} -o ${spv_out}
    DEPENDS ${shader_src}
    COMMENT "Compiling shader ${shader_name}"
    VERBATIM)
  list(APPEND VKMOTION_SPV_FILES ${spv_out})
endforeach()
add_custom_target(vkmotion_shaders DEPENDS ${VKMOTION_SPV_FILES})

configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/cmake/vkmotion_config.hpp.in
  ${CMAKE_BINARY_DIR}/generated/vkmotion/vkmotion_config.hpp
  @ONLY)

# ---- vkmotion library ---------------------------------------------------------
file(GLOB VKMOTION_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/vkmotion/*.cpp)
file(GLOB VKMOTION_HEADERS ${CMAKE_CURRENT_SOURCE_DIR}/vkmotion/*.hpp)

add_library(vkmotion STATIC ${VKMOTION_SOURCES})
add_dependencies(vkmotion vkmotion_shaders)

target_include_directories(vkmotion
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_BINARY_DIR}/generated)

target_link_libraries(vkmotion
  PUBLIC
    cutil::cutil
    Eigen3::Eigen
    Vulkan::Vulkan
    GPUOpen::VulkanMemoryAllocator
    tinyxml2::tinyxml2)

target_compile_definitions(vkmotion PUBLIC
  $<$<BOOL:${VKMOTION_ENABLE_VALIDATION}>:VKMOTION_ENABLE_VALIDATION=1>)

if(VKMOTION_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()

if(VKMOTION_BUILD_EXAMPLES)
  add_subdirectory(examples)
endif()
