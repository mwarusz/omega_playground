set(CMAKE_CXX_COMPILER "clang++" CACHE STRING "")
set(CMAKE_BUILD_TYPE "Release" CACHE STRING "")
set(CMAKE_CXX_FLAGS "-mprefer-vector-width=512" CACHE STRING "")
set(CMAKE_CXX_EXTENSIONS OFF CACHE BOOL "")

set(Kokkos_ENABLE_SERIAL ON CACHE BOOL "")
set(Kokkos_ENABLE_OPENMP ON CACHE BOOL "")
set(Kokkos_ARCH_SPR ON CACHE BOOL "")