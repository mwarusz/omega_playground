#include "shallow_water_auxstate.hpp"

namespace omega {

ShallowWaterAuxiliaryState::ShallowWaterAuxiliaryState(const MPASMesh *mesh)
    : m_ke_cell(mesh), m_vel_div_cell(mesh), m_vel_del2_div_cell(mesh),
      m_norm_tr_cell(mesh), m_tr_del2_cell(mesh), m_h_flux_edge(mesh),
      m_h_mean_edge(mesh), m_h_drag_edge(mesh), m_norm_rvort_edge(mesh),
      m_norm_f_edge(mesh), m_vel_del2_edge(mesh), m_rvort_vertex(mesh),
      m_vel_del2_rvort_vertex(mesh), m_norm_rvort_vertex(mesh),
      m_norm_f_vertex(mesh) {}

void ShallowWaterAuxiliaryState::allocate(const MPASMesh *mesh, Int ntracers) {
  m_ke_cell.allocate(mesh);
  m_vel_div_cell.allocate(mesh);
  m_vel_del2_div_cell.allocate(mesh);
  m_norm_tr_cell.allocate(mesh, ntracers);
  m_tr_del2_cell.allocate(mesh, ntracers);

  m_h_flux_edge.allocate(mesh);
  m_h_mean_edge.allocate(mesh);
  m_h_drag_edge.allocate(mesh);
  m_norm_rvort_edge.allocate(mesh);
  m_norm_f_edge.allocate(mesh);
  m_vel_del2_edge.allocate(mesh);

  m_rvort_vertex.allocate(mesh);
  m_vel_del2_rvort_vertex.allocate(mesh);
  m_norm_rvort_vertex.allocate(mesh);
  m_norm_f_vertex.allocate(mesh);
}

void ShallowWaterAuxiliaryState::compute(RealConst2d h_cell,
                                         RealConst2d vn_edge,
                                         RealConst3d tr_cell,
                                         RealConst1d f_vertex,
                                         const MPASMesh *mesh) const {

  OMEGA_SCOPE(rvort_vertex, m_rvort_vertex);
  OMEGA_SCOPE(norm_rvort_vertex, m_norm_rvort_vertex);
  OMEGA_SCOPE(norm_f_vertex, m_norm_f_vertex);
  OMEGA_SCOPE(rvort_vertex_const_arr, rvort_vertex.const_array());

  omega_parallel_for(
      "compute_vertex_auxiliarys_phase1", {mesh->m_nvertices, mesh->m_nlayers},
      KOKKOS_LAMBDA(Int ivertex, Int k) {
        if (rvort_vertex.m_enabled) {
          rvort_vertex.m_array(ivertex, k) = rvort_vertex(ivertex, k, vn_edge);
        }

        if (norm_rvort_vertex.m_enabled) {
          norm_rvort_vertex.m_array(ivertex, k) =
              norm_rvort_vertex(ivertex, k, rvort_vertex_const_arr, h_cell);
        }

        if (norm_f_vertex.m_enabled) {
          norm_f_vertex.m_array(ivertex, k) =
              norm_f_vertex(ivertex, k, f_vertex, h_cell);
        }
      });

  OMEGA_SCOPE(ke_cell, m_ke_cell);
  OMEGA_SCOPE(vel_div_cell, m_vel_div_cell);
  OMEGA_SCOPE(norm_tr_cell, m_norm_tr_cell);
  const Int ntracers = tr_cell.extent(0);

  omega_parallel_for(
      "compute_cell_auxiliarys_phase1", {mesh->m_ncells, mesh->m_nlayers},
      KOKKOS_LAMBDA(Int icell, Int k) {
        if (ke_cell.m_enabled) {
          ke_cell.m_array(icell, k) = ke_cell(icell, k, vn_edge);
        }

        if (vel_div_cell.m_enabled) {
          vel_div_cell.m_array(icell, k) = vel_div_cell(icell, k, vn_edge);
        }

        if (norm_tr_cell.m_enabled) {
          for (Int l = 0; l < ntracers; ++l) {
            norm_tr_cell.m_array(l, icell, k) =
                norm_tr_cell(l, icell, k, tr_cell, h_cell);
          }
        }
      });

  OMEGA_SCOPE(h_flux_edge, m_h_flux_edge);
  OMEGA_SCOPE(h_mean_edge, m_h_mean_edge);
  OMEGA_SCOPE(h_drag_edge, m_h_drag_edge);
  OMEGA_SCOPE(norm_rvort_edge, m_norm_rvort_edge);
  OMEGA_SCOPE(norm_f_edge, m_norm_f_edge);
  OMEGA_SCOPE(vel_del2_edge, m_vel_del2_edge);
  OMEGA_SCOPE(norm_rvort_vertex_const_arr, norm_rvort_vertex.const_array());
  OMEGA_SCOPE(norm_f_vertex_const_arr, norm_f_vertex.const_array());
  OMEGA_SCOPE(vel_div_cell_const_arr, vel_div_cell.const_array());

  omega_parallel_for(
      "compute_edge_auxiliarys_phase1", {mesh->m_nedges, mesh->m_nlayers},
      KOKKOS_LAMBDA(Int iedge, Int k) {
        if (h_flux_edge.m_enabled) {
          h_flux_edge.m_array(iedge, k) = h_flux_edge(iedge, k, h_cell);
        }

        if (h_mean_edge.m_enabled) {
          h_mean_edge.m_array(iedge, k) = h_mean_edge(iedge, k, h_cell);
        }

        if (h_drag_edge.m_enabled) {
          h_drag_edge.m_array(iedge, k) = h_drag_edge(iedge, k, h_cell);
        }

        if (norm_rvort_edge.m_enabled) {
          norm_rvort_edge.m_array(iedge, k) =
              norm_rvort_edge(iedge, k, norm_rvort_vertex_const_arr);
        }

        if (norm_f_edge.m_enabled) {
          norm_f_edge.m_array(iedge, k) =
              norm_f_edge(iedge, k, norm_f_vertex_const_arr);
        }

        if (vel_del2_edge.m_enabled) {
          vel_del2_edge.m_array(iedge, k) = vel_del2_edge(
              iedge, k, vel_div_cell_const_arr, rvort_vertex_const_arr);
        }
      });

  // phase2

  OMEGA_SCOPE(tr_del2_cell, m_tr_del2_cell);
  OMEGA_SCOPE(vel_del2_div_cell, m_vel_del2_div_cell);
  OMEGA_SCOPE(norm_tr_cell_const_arr, norm_tr_cell.const_array());
  OMEGA_SCOPE(h_mean_edge_const_arr, h_mean_edge.const_array());
  OMEGA_SCOPE(vel_del2_edge_const_arr, vel_del2_edge.const_array());
  omega_parallel_for(
      "compute_cell_auxiliarys_phase2", {mesh->m_ncells, mesh->m_nlayers},
      KOKKOS_LAMBDA(Int icell, Int k) {
        if (tr_del2_cell.m_enabled) {
          for (Int l = 0; l < ntracers; ++l) {
            tr_del2_cell.m_array(l, icell, k) = tr_del2_cell(
                l, icell, k, norm_tr_cell_const_arr, h_mean_edge_const_arr);
          }
        }

        if (vel_del2_div_cell.m_enabled) {
          vel_del2_div_cell.m_array(icell, k) =
              vel_del2_div_cell(icell, k, vel_del2_edge_const_arr);
        }
      });

  OMEGA_SCOPE(vel_del2_rvort_vertex, m_vel_del2_rvort_vertex);
  omega_parallel_for(
      "compute_vertex_auxiliarys_phase2", {mesh->m_nvertices, mesh->m_nlayers},
      KOKKOS_LAMBDA(Int ivertex, Int k) {
        if (vel_del2_rvort_vertex.m_enabled) {
          vel_del2_rvort_vertex.m_array(ivertex, k) =
              vel_del2_rvort_vertex(ivertex, k, vel_del2_edge_const_arr);
        }
      });
}
} // namespace omega
