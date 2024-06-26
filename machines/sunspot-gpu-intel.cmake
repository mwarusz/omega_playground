set(CMAKE_CXX_COMPILER "icpx" CACHE STRING "")
set(CMAKE_BUILD_TYPE "Release" CACHE STRING "")
set(CMAKE_CXX_FLAGS "-Xclang -mlong-double-64" CACHE STRING "")

set(Kokkos_ENABLE_SERIAL ON CACHE BOOL "")
set(Kokkos_ENABLE_OPENMP OFF CACHE BOOL "")
set(Kokkos_ENABLE_SYCL ON CACHE BOOL "")
set(Kokkos_ARCH_NATIVE ON CACHE BOOL "")
set(Kokkos_ARCH_INTEL_GEN ON CACHE BOOL "")
set(Kokkos_ENABLE_ONEDPL OFF CACHE BOOL "")

set(OMEGA_USE_CALIPER ON CACHE BOOL "")
set(caliper_DIR $ENV{HOME}/installs/caliper/share/cmake/caliper CACHE STRING "")
