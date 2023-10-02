macro(add_omega_experiment expname)
  add_executable(${expname} ${expname}.cpp)
  yakl_process_target(${expname})
  target_link_libraries(${expname} omega)
endmacro(add_omega_experiment)

add_omega_experiment(cosine_bell)
add_omega_experiment(steady_zonal)
