cmake_minimum_required(VERSION 3.20)

project(omega_playground CXX)

set(CMAKE_CXX_STANDARD 17)

if ("${YAKL_ARCH}" STREQUAL "CUDA")
  enable_language(CUDA)
endif()

include(FetchContent)
FetchContent_Declare(yakl
  GIT_REPOSITORY https://github.com/mrnorman/YAKL.git
  GIT_TAG main
)

FetchContent_MakeAvailable(yakl)

add_library(omega
  src/planar_hexagonal_mesh.cpp
  src/shallow_water.cpp
  src/lsrk.cpp
  src/rk4.cpp
)
yakl_process_target(omega)
target_include_directories(omega PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/src
)

enable_testing()
add_subdirectory(test)
