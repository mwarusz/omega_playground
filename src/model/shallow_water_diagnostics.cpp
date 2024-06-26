#include "shallow_water_diagnostics.hpp"

namespace omega {

Real energy_integral(const ShallowWaterState &state,
                     const ShallowWaterModel &model) {
  OMEGA_SCOPE(h_cell, state.m_h_cell);
  OMEGA_SCOPE(vn_edge, state.m_vn_edge);
  OMEGA_SCOPE(mesh, model.m_mesh);
  OMEGA_SCOPE(grav, model.m_params.m_grav);

  OMEGA_SCOPE(nedges_on_cell, mesh->m_nedges_on_cell);
  OMEGA_SCOPE(edges_on_cell, mesh->m_edges_on_cell);
  OMEGA_SCOPE(dv_edge, mesh->m_dv_edge);
  OMEGA_SCOPE(dc_edge, mesh->m_dc_edge);
  OMEGA_SCOPE(area_cell, mesh->m_area_cell);
  OMEGA_SCOPE(max_level_cell, mesh->m_max_level_cell);

  Real total_energy;
  omega_parallel_reduce(
      "compute_column_energy", {mesh->m_ncells},
      KOKKOS_LAMBDA(Int icell, Real & column_energy) {
        for (Int k = 0; k < max_level_cell(icell); ++k) {
          Real K = 0;
          for (Int j = 0; j < nedges_on_cell(icell); ++j) {
            Int jedge = edges_on_cell(icell, j);
            Real area_edge = dv_edge(jedge) * dc_edge(jedge);
            K += area_edge * vn_edge(jedge, k) * vn_edge(jedge, k) / 4;
          }
          K /= area_cell(icell);
          column_energy += area_cell(icell) *
                           (grav * h_cell(icell, k) * h_cell(icell, k) / 2 +
                            h_cell(icell, k) * K);
        }
      },
      total_energy);
  return total_energy;
}

Real mass_integral(const ShallowWaterState &state,
                   const ShallowWaterModel &model) {
  OMEGA_SCOPE(h_cell, state.m_h_cell);
  OMEGA_SCOPE(mesh, model.m_mesh);

  OMEGA_SCOPE(area_cell, mesh->m_area_cell);
  OMEGA_SCOPE(max_level_cell, mesh->m_max_level_cell);

  Real total_mass;
  omega_parallel_reduce(
      "compute_column_mass", {mesh->m_ncells},
      KOKKOS_LAMBDA(Int icell, Real & column_mass) {
        for (Int k = 0; k < max_level_cell(icell); ++k) {
          column_mass += area_cell(icell) * h_cell(icell, k);
        }
      },
      total_mass);
  return total_mass;
}

Real circulation_integral(const ShallowWaterState &state,
                          const ShallowWaterModel &model) {
  OMEGA_SCOPE(vn_edge, state.m_vn_edge);
  OMEGA_SCOPE(mesh, model.m_mesh);

  OMEGA_SCOPE(dc_edge, mesh->m_dc_edge);
  OMEGA_SCOPE(edges_on_vertex, mesh->m_edges_on_vertex);
  OMEGA_SCOPE(edge_sign_on_vertex, mesh->m_edge_sign_on_vertex);
  OMEGA_SCOPE(area_triangle, mesh->m_area_triangle);
  OMEGA_SCOPE(f_vertex, model.m_f_vertex);
  OMEGA_SCOPE(max_level_vertex_bot, mesh->m_max_level_vertex_bot);

  Real total_circulation;
  omega_parallel_reduce(
      "compute_column_circulation", {mesh->m_nvertices},
      KOKKOS_LAMBDA(Int ivertex, Real & column_circulation) {
        for (Int k = 0; k < max_level_vertex_bot(ivertex); ++k) {
          Real cir_i = -0;
          for (Int j = 0; j < 3; ++j) {
            Int jedge = edges_on_vertex(ivertex, j);
            cir_i += dc_edge(jedge) * edge_sign_on_vertex(ivertex, j) *
                     vn_edge(jedge, k);
          }
          column_circulation +=
              cir_i + f_vertex(ivertex) * area_triangle(ivertex);
        }
      },
      total_circulation);
  return total_circulation;
}

} // namespace omega
