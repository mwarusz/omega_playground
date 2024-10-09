foreach(omega_lib IN LISTS OMEGA_LIBS)
  target_sources(${omega_lib} PRIVATE
    shallow_water_state.cpp
    shallow_water_auxstate.cpp
    shallow_water_model.cpp
    shallow_water_diagnostics.cpp
  )
endforeach()
