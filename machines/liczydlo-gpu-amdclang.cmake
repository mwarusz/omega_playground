set(CMAKE_CXX_COMPILER "/opt/rocm/llvm/bin/clang++" CACHE STRING "")
set(CMAKE_BUILD_TYPE "Release" CACHE STRING "")

set(OMEGA_KOKKOS_REPO "https://github.com/mwarusz/kokkos.git" CACHE STRING "")
set(OMEGA_KOKKOS_TAG "e8ce3aeba39b7ae6056de9d1e354ca23ef487a0b" CACHE STRING "")

set(Kokkos_ENABLE_SERIAL ON CACHE BOOL "")
set(Kokkos_ENABLE_OPENMP OFF CACHE BOOL "")
set(Kokkos_ENABLE_HIP ON CACHE BOOL "")
set(Kokkos_ARCH_NATIVE ON CACHE BOOL "")

set(OMEGA_USE_CALIPER ON CACHE BOOL "")
set(caliper_DIR "/home/mwarusz/installs/caliper/share/cmake/caliper" CACHE STRING "")
