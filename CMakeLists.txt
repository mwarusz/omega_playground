cmake_minimum_required(VERSION 3.20)

project(omega_playground CXX)

set(CMAKE_CXX_STANDARD 17)

include(FetchContent)
FetchContent_Declare(yakl
  GIT_REPOSITORY https://github.com/mrnorman/YAKL.git
  GIT_TAG main
)

FetchContent_MakeAvailable(yakl)

add_library(lib INTERFACE)
target_include_directories(lib INTERFACE
  ${CMAKE_CURRENT_SOURCE_DIR}/src
)

#find_package(VTK COMPONENTS 
#  CommonCore
#  CommonDataModel
#  IOXML
#)

add_executable(driver src/driver.cpp)
#target_link_libraries(driver lib ${VTK_LIBRARIES})
target_link_libraries(driver lib)
yakl_process_target(driver)

add_executable(linear_swe src/linear_swe.cpp)
#target_link_libraries(driver lib ${VTK_LIBRARIES})
target_link_libraries(linear_swe lib)
yakl_process_target(linear_swe)
