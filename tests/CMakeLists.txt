include(FetchContent)
FetchContent_Declare(
  doctest
  GIT_REPOSITORY https://github.com/doctest/doctest.git
  GIT_TAG v2.4.11
  GIT_SHALLOW TRUE)
set(CMAKE_POLICY_VERSION_MINIMUM 3.5)
FetchContent_MakeAvailable(doctest)
unset(CMAKE_POLICY_VERSION_MINIMUM)

file(GLOB VKMOTION_TEST_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)

add_executable(vkmotion_tests ${VKMOTION_TEST_SOURCES})
target_link_libraries(vkmotion_tests PRIVATE vkmotion doctest::doctest)
target_include_directories(vkmotion_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_definitions(vkmotion_tests PRIVATE
  VKMOTION_TEST_ASSET_DIR="${CMAKE_SOURCE_DIR}/test_assets")

include(${doctest_SOURCE_DIR}/scripts/cmake/doctest.cmake)
doctest_discover_tests(vkmotion_tests)
