option(BENCHMARK_PROFILE_CUDA "turn on cuda profiling")

add_executable(benchmark benchmark.cpp)
yakl_process_target(benchmark)
target_link_libraries(benchmark omega)
if (BENCHMARK_PROFILE_CUDA)
  target_compile_definitions(benchmark PUBLIC BENCHMARK_PROFILE_CUDA)
endif()
