cmake_minimum_required(VERSION 3.20)

project(omega_playground CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_EXTENSIONS OFF)

set(OMEGA_KOKKOS_REPO "https://github.com/kokkos/kokkos.git" CACHE STRING "kokkos repo")
set(OMEGA_KOKKOS_TAG "4.3.00" CACHE STRING "kokkos tag")

set(OMEGA_VECTOR_LENGTH 1 CACHE STRING "omega vector length")
set(OMEGA_USE_CALIPER OFF CACHE BOOL "caliper integration")

include(FetchContent)
set(FETCHCONTENT_UPDATES_DISCONNECTED ON)
FetchContent_Declare(Kokkos
  GIT_REPOSITORY ${OMEGA_KOKKOS_REPO}
  GIT_TAG ${OMEGA_KOKKOS_TAG}
)
FetchContent_MakeAvailable(Kokkos)

add_library(omega)

target_link_libraries(omega PUBLIC Kokkos::kokkos)
if (${OMEGA_USE_CALIPER})
  find_package(caliper REQUIRED)
  target_link_libraries(omega PUBLIC caliper)
  target_compile_definitions(omega PUBLIC OMEGA_USE_CALIPER=1)
endif()
if (${OMEGA_NO_SIMD})
  target_compile_definitions(omega PUBLIC OMEGA_NO_SIMD=1)
endif()
if (${OMEGA_KOKKOS_SIMD})
  target_compile_definitions(omega PUBLIC OMEGA_KOKKOS_SIMD=1)
endif()
if (${OMEGA_NO_INNER_IF})
  target_compile_definitions(omega PUBLIC OMEGA_NO_INNER_IF=1)
endif()

find_package(netCDFCxx)
if (${netCDFCxx_FOUND})
  find_package(HDF5 REQUIRED COMPONENTS C HL)
  target_link_libraries(omega PUBLIC HDF5::HDF5 netCDF::netcdf-cxx4)
endif()

target_include_directories(omega PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/src
)

add_subdirectory(src)

add_subdirectory(benchmark)
if (${netCDFCxx_FOUND})
  add_subdirectory(exp)
endif()

enable_testing()
add_subdirectory(test)
