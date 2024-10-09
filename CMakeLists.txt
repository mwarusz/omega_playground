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

add_library(omega_split)
target_compile_definitions(omega_split PUBLIC OMEGA_NO_INNER_IF=1)

add_library(omega_fused)

add_library(omega_fused_accum)
target_compile_definitions(omega_fused_accum PUBLIC OMEGA_ACCUM=1)

add_library(omega_split_osimd)
target_compile_definitions(omega_split_osimd PUBLIC OMEGA_NO_INNER_IF=1)
target_compile_options(omega_split_osimd PUBLIC -fopenmp-simd)

add_library(omega_fused_osimd)
target_compile_options(omega_fused_osimd PUBLIC -fopenmp-simd)

add_library(omega_fused_accum_osimd)
target_compile_definitions(omega_fused_accum_osimd PUBLIC OMEGA_ACCUM=1)
target_compile_options(omega_fused_accum_osimd PUBLIC -fopenmp-simd)

add_library(omega_split_ksimd)
target_compile_definitions(omega_split_ksimd PUBLIC OMEGA_NO_INNER_IF=1)
target_compile_definitions(omega_split_ksimd PUBLIC OMEGA_KOKKOS_SIMD=1)

add_library(omega_fused_ksimd)
target_compile_definitions(omega_fused_ksimd PUBLIC OMEGA_KOKKOS_SIMD=1)

add_library(omega_fused_accum_ksimd)
target_compile_definitions(omega_fused_accum_ksimd PUBLIC OMEGA_KOKKOS_SIMD=1)
target_compile_definitions(omega_fused_accum_ksimd PUBLIC OMEGA_ACCUM=1)

set(OMEGA_LIBS omega_split omega_fused omega_fused_accum
               omega_split_osimd omega_fused_osimd omega_fused_accum_osimd
               omega_split_ksimd omega_fused_ksimd omega_fused_accum_ksimd)

foreach(omega_lib IN LISTS OMEGA_LIBS)
  target_link_libraries(${omega_lib} PUBLIC Kokkos::kokkos)
endforeach()

if (${OMEGA_USE_CALIPER})
  find_package(caliper REQUIRED)
  foreach(omega_lib IN LISTS OMEGA_LIBS)
    target_link_libraries(${omega_lib} PUBLIC caliper)
    target_compile_definitions(${omega_lib} PUBLIC OMEGA_USE_CALIPER=1)
  endforeach()
endif()

set(NETCDF_WORKAROUND OFF CACHE BOOL "netcdf/hdf5 workaround")

find_package(netCDFCxx)
if (${netCDFCxx_FOUND})
  foreach(omega_lib IN LISTS OMEGA_LIBS)
    target_link_libraries(${omega_lib} PUBLIC netCDF::netcdf-cxx4)
  endforeach()
  # for some reason I need to explictly look for hdf5 on my mac laptop
  if (${NETCDF_WORKAROUND})
    find_package(HDF5 REQUIRED COMPONENTS C HL)
    foreach(omega_lib IN LISTS OMEGA_LIBS)
      target_link_libraries(${omega_lib} PUBLIC HDF5::HDF5)
    endforeach()
  endif()
endif()

foreach(omega_lib IN LISTS OMEGA_LIBS)
  target_include_directories(${omega_lib} PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src
  )
endforeach()

add_subdirectory(src)

#add_subdirectory(benchmark)
if (${netCDFCxx_FOUND})
  add_subdirectory(exp)
endif()
#
#enable_testing()
#add_subdirectory(test)
