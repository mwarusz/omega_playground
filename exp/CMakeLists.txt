foreach(omega_lib IN LISTS OMEGA_LIBS)
  add_executable(steady_zonal_${omega_lib} steady_zonal.cpp)
  target_link_libraries(steady_zonal_${omega_lib} ${omega_lib})
endforeach()
