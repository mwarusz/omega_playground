set(CMAKE_CXX_COMPILER "amdclang++" CACHE STRING "")
set(CMAKE_BUILD_TYPE "Release" CACHE STRING "")
set(CMAKE_CXX_EXTENSIONS OFF CACHE BOOL "")

set(Kokkos_ENABLE_SERIAL ON CACHE BOOL "")
set(Kokkos_ENABLE_OPENMP OFF CACHE BOOL "")
set(Kokkos_ENABLE_HIP ON CACHE BOOL "")
set(Kokkos_ARCH_NATIVE ON CACHE BOOL "")
set(Kokkos_ARCH_GFX90A ON CACHE BOOL "")
