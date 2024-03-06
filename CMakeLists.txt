cmake_minimum_required(VERSION 3.20)

project(omega_playground CXX)

set(OMEGA_VECTOR_LENGTH 1 CACHE STRING "omega vector length")
set(OMEGA_USE_CXX_SIMD OFF CACHE BOOL "use std::simd for vectorization")
set(OMEGA_BUILD_EXPERIMENTS OFF CACHE BOOL "build experiments")
set(OMEGA_USE_HIERARCHICAL OFF CACHE BOOL "use hierarchical parallelism")

if (${OMEGA_USE_CXX_SIMD})
  set(CMAKE_CXX_STANDARD 23)
else()
  set(CMAKE_CXX_STANDARD 17)
endif()


if ("${YAKL_ARCH}" STREQUAL "CUDA")
  enable_language(CUDA)
endif()

include(FetchContent)

set(FETCHCONTENT_UPDATES_DISCONNECTED ON)

FetchContent_Declare(yakl
  GIT_REPOSITORY https://github.com/mrnorman/YAKL.git
  GIT_TAG main
)

if (${OMEGA_BUILD_EXPERIMENTS})
  set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/")
  find_package(NetCDF REQUIRED)
endif()

FetchContent_MakeAvailable(yakl)


set(OMEGA_SOURCES
  src/mpas_mesh.cpp
  src/planar_hexagonal_mesh.cpp
  src/lsrk.cpp
  src/rk4.cpp
)

if (${OMEGA_USE_HIERARCHICAL})
    list(APPEND OMEGA_SOURCES src/shallow_water_hierarchical.cpp)
else()
  if (${OMEGA_VECTOR_LENGTH} EQUAL 1)
    list(APPEND OMEGA_SOURCES src/shallow_water.cpp)
  else()
    if (${OMEGA_USE_CXX_SIMD})
      list(APPEND OMEGA_SOURCES src/shallow_water_vectorized_cxx_simd.cpp)
    else()
      list(APPEND OMEGA_SOURCES src/shallow_water_vectorized_yakl_packs.cpp)
    endif()
  endif()
endif()

if (${OMEGA_BUILD_EXPERIMENTS})
  list(APPEND OMEGA_SOURCES src/file_mesh.cpp)
endif()

add_library(omega ${OMEGA_SOURCES})
yakl_process_target(omega)
target_include_directories(omega PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/src
)
target_link_libraries(omega ${OMEGA_LINK_FLAGS})
target_compile_definitions(omega PUBLIC OMEGA_VECTOR_LENGTH=${OMEGA_VECTOR_LENGTH})
if (${OMEGA_BUILD_EXPERIMENTS})
  target_link_libraries(omega NetCDF::NetCDF)
  target_compile_definitions(omega PUBLIC OMEGA_FILE_MESH)
endif()

add_subdirectory(benchmark)
if (${OMEGA_BUILD_EXPERIMENTS})
  add_subdirectory(experiment)
endif()

enable_testing()
add_subdirectory(test)
