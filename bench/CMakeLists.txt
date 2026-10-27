include(FetchContent)
if(NOT TARGET nlohmann_json::nlohmann_json)
  FetchContent_Declare(
    json
    GIT_REPOSITORY https://github.com/nlohmann/json.git
    GIT_TAG v3.11.3
    GIT_SHALLOW TRUE)
  set(JSON_BuildTests OFF CACHE INTERNAL "")
  FetchContent_MakeAvailable(json)
endif()

add_executable(bench_vkmotion bench_vkmotion.cpp)
target_link_libraries(bench_vkmotion PRIVATE vkmotion nlohmann_json::nlohmann_json)
