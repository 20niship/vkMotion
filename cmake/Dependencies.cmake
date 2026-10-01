include(FetchContent)
if(POLICY CMP0169)
  cmake_policy(SET CMP0169 OLD)
endif()

# ---- cutil: header-only math / Ref<T> library ---------------------------------
# cutil's own CMakeLists.txt builds nanobind Python bindings, which we don't need
# here, so we vendor the source and expose it as a plain INTERFACE target instead
# of add_subdirectory()-ing it.
FetchContent_Declare(
  cutil
  GIT_REPOSITORY https://github.com/20niship/cutil.git
  GIT_TAG master)
FetchContent_Populate(cutil)

add_library(cutil INTERFACE)
target_include_directories(cutil INTERFACE ${cutil_SOURCE_DIR})
add_library(cutil::cutil ALIAS cutil)

# ---- Eigen: header-only linear algebra (used by IKSolver) ---------------------
# Eigen's own CMakeLists.txt builds its full test/doc/blas infra (Fortran, Qt,
# SuiteSparse probing, ...); we only need the headers, so populate the source
# and expose it as a plain INTERFACE target instead of add_subdirectory()-ing it.
FetchContent_Declare(
  eigen
  GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
  GIT_TAG 3.4.0)
FetchContent_Populate(eigen)

add_library(Eigen3 INTERFACE)
target_include_directories(Eigen3 INTERFACE ${eigen_SOURCE_DIR})
add_library(Eigen3::Eigen ALIAS Eigen3)

# ---- VulkanMemoryAllocator ------------------------------------------------------
FetchContent_Declare(
  vma
  GIT_REPOSITORY https://github.com/GPUOpen-LibrariesAndSDKs/VulkanMemoryAllocator.git
  GIT_TAG v3.1.0)
FetchContent_MakeAvailable(vma)

# ---- tinyxml2: URDF parsing, vendored as a git submodule -----------------------
set(tinyxml2_BUILD_TESTING OFF CACHE BOOL "" FORCE)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/third_party/tinyxml2 EXCLUDE_FROM_ALL)
