cmake_minimum_required(VERSION 3.20)

project(omega_playground CXX)

set(CMAKE_CXX_STANDARD 17)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/")

if ("${YAKL_ARCH}" STREQUAL "CUDA")
  enable_language(CUDA)
endif()

include(FetchContent)

set(FETCHCONTENT_UPDATES_DISCONNECTED ON)

FetchContent_Declare(yakl
  GIT_REPOSITORY https://github.com/mrnorman/YAKL.git
  GIT_TAG main
)

find_package(NetCDF REQUIRED)

FetchContent_MakeAvailable(yakl)

add_library(omega
  src/mpas_mesh.cpp
  src/planar_hexagonal_mesh.cpp
  src/file_mesh.cpp
  src/shallow_water.cpp
  src/lsrk.cpp
  src/rk4.cpp
)
yakl_process_target(omega)
target_include_directories(omega PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/src
)
target_link_libraries(omega NetCDF::NetCDF ${OMEGA_LINK_FLAGS})

add_subdirectory(benchmark)
add_subdirectory(experiment)

enable_testing()
add_subdirectory(test)
