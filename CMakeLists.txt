cmake_minimum_required(VERSION 3.20)

project(omega_playground CXX)

set(CMAKE_CXX_STANDARD 17)

set(OMEGA_VECTOR_LENGTH 1 CACHE STRING "omega vector length")

include(FetchContent)
set(FETCHCONTENT_UPDATES_DISCONNECTED ON)
FetchContent_Declare(Kokkos
  GIT_REPOSITORY https://github.com/kokkos/kokkos.git
  GIT_TAG 4.3.00
)
FetchContent_MakeAvailable(Kokkos)

add_library(omega)
target_link_libraries(omega PUBLIC Kokkos::kokkos)
target_compile_definitions(omega PUBLIC OMEGA_VECTOR_LENGTH=${OMEGA_VECTOR_LENGTH})

target_include_directories(omega PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/src
)

add_subdirectory(src)

add_subdirectory(benchmark)

enable_testing()
add_subdirectory(test)
