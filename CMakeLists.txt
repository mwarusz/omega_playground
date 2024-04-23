cmake_minimum_required(VERSION 3.20)

project(omega_playground CXX)

set(CMAKE_CXX_STANDARD 17)

set(OMEGA_VECTOR_LENGTH 1 CACHE STRING "omega vector length")

find_package(Kokkos REQUIRED PATHS ${Kokkos_PATH} NO_DEFAULT_PATH)

set(OMEGA_SOURCES
  src/mpas_mesh.cpp
  src/planar_hexagonal_mesh.cpp
  src/lsrk.cpp
  src/rk4.cpp
)

if (${OMEGA_USE_HIERARCHICAL})
  list(APPEND OMEGA_SOURCES src/shallow_water_hierarchical.cpp)
else()
  list(APPEND OMEGA_SOURCES src/shallow_water.cpp)
endif()

add_library(omega ${OMEGA_SOURCES})
target_include_directories(omega PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/src
)
target_link_libraries(omega Kokkos::kokkos)
target_compile_definitions(omega PUBLIC OMEGA_VECTOR_LENGTH=${OMEGA_VECTOR_LENGTH})

add_subdirectory(benchmark)

enable_testing()
add_subdirectory(test)
