target_sources(omega PRIVATE
  shallow_water_state.cpp
  shallow_water_auxstate.cpp
  shallow_water_model.cpp
  shallow_water_diagnostics.cpp
)

add_subdirectory(auxiliary_vars)
