cmake_minimum_required(VERSION 3.20)
project(vkMotion LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(VKMOTION_BUILD_TESTS "Build doctest-based unit tests" OFF)
option(VKMOTION_BUILD_EXAMPLES "Build example tools (stress-test runner, etc.)" OFF)
option(VKMOTION_BUILD_BENCH "Build bench/ (comparison against vamp / cuRobo)" OFF)
option(VKMOTION_NATIVE_ARCH "Compile the library with -march=native (CPU SIMD for collision/FK hot paths; build-from-source use only)" ON)
option(VKMOTION_ENABLE_VALIDATION "Enable Vulkan validation layers by default" ON)

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

include(cmake/Dependencies.cmake)

find_package(Vulkan REQUIRED)

# ---- shader compilation (glslc .comp -> .spv) --------------------------------
find_program(GLSLC_EXECUTABLE glslc REQUIRED)

file(GLOB VKMOTION_SHADER_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/shaders/*.comp)
file(GLOB VKMOTION_SHADER_INCLUDES ${CMAKE_CURRENT_SOURCE_DIR}/shaders/include/*.glsl)
set(VKMOTION_SHADER_BINARY_DIR ${CMAKE_BINARY_DIR}/shaders)
file(MAKE_DIRECTORY ${VKMOTION_SHADER_BINARY_DIR})

set(VKMOTION_SPV_FILES "")
foreach(shader_src ${VKMOTION_SHADER_SOURCES})
  get_filename_component(shader_name ${shader_src} NAME)
  set(spv_out ${VKMOTION_SHADER_BINARY_DIR}/${shader_name}.spv)
  add_custom_command(
    OUTPUT ${spv_out}
    COMMAND ${GLSLC_EXECUTABLE} --target-env=vulkan1.1 ${shader_src} -o ${spv_out}
    DEPENDS ${shader_src} ${VKMOTION_SHADER_INCLUDES}
    COMMENT "Compiling shader ${shader_name}"
    VERBATIM)
  list(APPEND VKMOTION_SPV_FILES ${spv_out})
endforeach()
add_custom_target(vkmotion_shaders DEPENDS ${VKMOTION_SPV_FILES})

# 実行時に生成するCPUカーネルのコンパイル用フラグ(ライブラリ本体と同じ命令セットにする)
if(VKMOTION_NATIVE_ARCH)
  set(VKMOTION_JIT_FLAGS "-O2 -march=native -fno-math-errno -std=c++20 -shared -fPIC")
else()
  set(VKMOTION_JIT_FLAGS "-O2 -fno-math-errno -std=c++20 -shared -fPIC")
endif()
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

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  # sqrtfがerrnoを設定しないことでループがSIMD化できる
  target_compile_options(vkmotion PRIVATE -fno-math-errno)
  if(VKMOTION_NATIVE_ARCH)
    target_compile_options(vkmotion PRIVATE -march=native)
  endif()
endif()
find_package(Threads REQUIRED)
target_link_libraries(vkmotion PUBLIC Threads::Threads ${CMAKE_DL_LIBS})

target_compile_definitions(vkmotion PUBLIC
  $<$<BOOL:${VKMOTION_ENABLE_VALIDATION}>:VKMOTION_ENABLE_VALIDATION=1>)

if(VKMOTION_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()

if(VKMOTION_BUILD_EXAMPLES)
  add_subdirectory(examples)
endif()

if(VKMOTION_BUILD_BENCH)
  add_subdirectory(bench)
endif()
