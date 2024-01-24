set(CMAKE_CXX_COMPILER "amdclang++" CACHE STRING "")
set(CMAKE_C_COMPILER "amdclang" CACHE STRING "")
set(CMAKE_Fortran_COMPILER "amdflang" CACHE STRING "")

set(YAKL_ARCH "HIP" CACHE STRING "")
set(YAKL_HIP_FLAGS "-O3 -std=c++17 -D__HIP_ROCclr__ -D__HIP_ARCH_GFX90A__=1 --rocm-path=$ENV{ROCM_PATH} --offload-arch=gfx90a -x hip -Wno-unused-result" CACHE STRING "")

set(OMEGA_LINK_FLAGS "--rocm-path=$ENV{ROCM_PATH} -L$ENV{ROCM_PATH}/lib -lamdhip64" CACHE STRING "")
