set(CMAKE_CXX_COMPILER "g++-14" CACHE STRING "")
set(CMAKE_BUILD_TYPE "Release" CACHE STRING "")

set(Kokkos_ENABLE_SERIAL ON CACHE BOOL "")
set(Kokkos_ENABLE_OPENMP OFF CACHE BOOL "")
set(Kokkos_ARCH_NATIVE ON CACHE BOOL "")

set(OMEGA_USE_CALIPER ON CACHE BOOL "")
set(caliper_DIR "/Users/mwarusz/installs/caliper/share/cmake/caliper" CACHE STRING "")

set(NETCDF_WORKAROUND ON CACHE BOOL "")
set(netCDFCxx_DIR $ENV{HOME}/installs/netcdf-cxx/lib/cmake/netCDF/ CACHE STRING "")
