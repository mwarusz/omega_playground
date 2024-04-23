target_sources(omega PRIVATE
  vel_div_cell.cpp
  vel_del2_div_cell.cpp
  kinetic_energy_cell.cpp
  norm_tr_cell.cpp
  tracer_del2_cell.cpp
  flux_thickness_edge.cpp
  mean_thickness_edge.cpp
  drag_thickness_edge.cpp
  norm_coriolis_edge.cpp
  norm_rvort_edge.cpp
  vel_del2_edge.cpp
  rvort_vertex.cpp
  vel_del2_rvort_vertex.cpp
  norm_rvort_vertex.cpp
  norm_coriolis_vertex.cpp
)
