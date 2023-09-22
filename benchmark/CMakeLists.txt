add_executable(benchmark benchmark.cpp)
yakl_process_target(benchmark)
target_link_libraries(benchmark omega)
