include(FetchContent)
FetchContent_Declare(
  json
  GIT_REPOSITORY https://github.com/nlohmann/json.git
  GIT_TAG v3.11.3
  GIT_SHALLOW TRUE)
set(JSON_BuildTests OFF CACHE INTERNAL "")
FetchContent_MakeAvailable(json)

add_executable(run_stress_test run_stress_test.cpp)
target_link_libraries(run_stress_test PRIVATE vkmotion nlohmann_json::nlohmann_json)

add_executable(run_batch_benchmark run_batch_benchmark.cpp)
target_link_libraries(run_batch_benchmark PRIVATE vkmotion nlohmann_json::nlohmann_json)
