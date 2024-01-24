set(CMAKE_CXX_COMPILER "icpx" CACHE STRING "")
set(CMAKE_C_COMPILER "icx" CACHE STRING "")
set(CMAKE_Fortran_COMPILER "ifx" CACHE STRING "")

set(YAKL_ARCH "SYCL" CACHE STRING "")
set(YAKL_SYCL_FLAGS "-O3 -fsycl -sycl-std=2020 -fsycl-unnamed-lambda -fsycl-device-code-split=per_kernel -fsycl-targets=spir64 -mlong-double-64 -Xclang -mlong-double-64" CACHE STRING "")
