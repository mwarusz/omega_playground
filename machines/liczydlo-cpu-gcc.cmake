set(CMAKE_CXX_COMPILER "g++" CACHE STRING "")
set(CMAKE_BUILD_TYPE "Release" CACHE STRING "")
set(CMAKE_CXX_FLAGS "-mprefer-vector-width=512" CACHE STRING "")

set(Kokkos_ENABLE_SERIAL ON CACHE BOOL "")
set(Kokkos_ENABLE_OPENMP OFF CACHE BOOL "")
set(Kokkos_ARCH_NATIVE ON CACHE BOOL "")

set(OMEGA_USE_CALIPER ON CACHE BOOL "")
set(caliper_DIR "/home/mwarusz/installs/caliper/share/cmake/caliper" CACHE STRING "")
