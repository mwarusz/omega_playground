foreach(omega_lib IN LISTS OMEGA_LIBS)
  target_sources(${omega_lib} PRIVATE
    lsrk.cpp
    rk4.cpp
  )
endforeach()
