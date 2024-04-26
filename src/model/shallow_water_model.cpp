#include "shallow_water_model.hpp"

namespace omega {

ShallowWaterModel::ShallowWaterModel(MPASMesh *mesh,
                                     const ShallowWaterParams &params)
    : m_params(params), m_mesh(mesh), m_aux_state(mesh),
      m_thickness_hadv_cell(mesh), m_pv_flux_edge(mesh), m_ke_grad_edge(mesh),
      m_ssh_grad_edge(mesh, params.m_grav),
      m_vel_diff_edge(mesh, params.m_visc_del2),
      m_vel_hyperdiff_edge(mesh, params.m_visc_del4), m_tracer_hadv_cell(mesh),
      m_tracer_diff_cell(mesh, params.m_eddy_diff2),
      m_tracer_hyperdiff_cell(mesh, params.m_eddy_diff4),
      m_f_vertex("f_vertex", mesh->m_nvertices),
      m_f_edge("f_edge", mesh->m_nedges) {

  m_thickness_hadv_cell.enable(m_aux_state);

  m_pv_flux_edge.enable(m_aux_state);
  m_ke_grad_edge.enable(m_aux_state);
  m_ssh_grad_edge.enable(m_aux_state);
  if (params.m_visc_del2 > 0) {
    m_vel_diff_edge.enable(m_aux_state);
  }
  if (params.m_visc_del4 > 0) {
    m_vel_hyperdiff_edge.enable(m_aux_state);
  }

  m_tracer_hadv_cell.enable(m_aux_state);
  if (params.m_eddy_diff2 > 0) {
    m_tracer_diff_cell.enable(m_aux_state);
  }
  if (params.m_eddy_diff4 > 0) {
    m_tracer_hyperdiff_cell.enable(m_aux_state);
  }

  m_aux_state.allocate(m_mesh, m_params.m_ntracers);

  deep_copy(m_f_vertex, m_params.m_f0);
  deep_copy(m_f_edge, m_params.m_f0);
}

void ShallowWaterModel::compute_h_tendency(Real2d h_tend_cell,
                                           RealConst2d h_cell,
                                           RealConst2d vn_edge,
                                           AddMode add_mode) const {
  OMEGA_SCOPE(h_flux_edge, m_aux_state.m_h_flux_edge.const_array());
  OMEGA_SCOPE(thickness_hadv_cell, m_thickness_hadv_cell);

  omega_parallel_for(
      "compute_h_tend", {m_mesh->m_ncells, m_mesh->m_nlayers},
      KOKKOS_LAMBDA(Int icell, Int k) {
        Real h_tend_cell_accum =
            add_mode == AddMode::increment ? h_tend_cell(icell, k) : 0;

        if (thickness_hadv_cell.m_enabled) {
          h_tend_cell_accum -=
              thickness_hadv_cell(icell, k, vn_edge, h_flux_edge);
        }

        h_tend_cell(icell, k) = h_tend_cell_accum;
      });
}

void ShallowWaterModel::compute_vn_tendency(Real2d vn_tend_edge,
                                            RealConst2d h_cell,
                                            RealConst2d vn_edge,
                                            AddMode add_mode) const {

  OMEGA_SCOPE(h_flux_edge, m_aux_state.m_h_flux_edge.const_array());
  OMEGA_SCOPE(ke_cell, m_aux_state.m_ke_cell.const_array());
  OMEGA_SCOPE(norm_rvort_edge, m_aux_state.m_norm_rvort_edge.const_array());
  OMEGA_SCOPE(norm_f_edge, m_aux_state.m_norm_f_edge.const_array());
  OMEGA_SCOPE(rvort_vertex, m_aux_state.m_rvort_vertex.const_array());
  OMEGA_SCOPE(vel_div_cell, m_aux_state.m_vel_div_cell.const_array());
  OMEGA_SCOPE(vel_del2_rvort_vertex,
              m_aux_state.m_vel_del2_rvort_vertex.const_array());
  OMEGA_SCOPE(vel_del2_div_cell, m_aux_state.m_vel_del2_div_cell.const_array());

  OMEGA_SCOPE(pv_flux_edge, m_pv_flux_edge);
  OMEGA_SCOPE(ke_grad_edge, m_ke_grad_edge);
  OMEGA_SCOPE(ssh_grad_edge, m_ssh_grad_edge);
  OMEGA_SCOPE(vel_diff_edge, m_vel_diff_edge);
  OMEGA_SCOPE(vel_hyperdiff_edge, m_vel_hyperdiff_edge);

  omega_parallel_for(
      "compute_vtend", {m_mesh->m_nedges, m_mesh->m_nlayers},
      KOKKOS_LAMBDA(Int iedge, Int k) {
        Real vn_tend_edge_accum =
            add_mode == AddMode::increment ? vn_tend_edge(iedge, k) : 0;

        if (pv_flux_edge.m_enabled) {
          vn_tend_edge_accum += pv_flux_edge(iedge, k, norm_rvort_edge,
                                             norm_f_edge, h_flux_edge, vn_edge);
        }

        if (ke_grad_edge.m_enabled) {
          vn_tend_edge_accum -= ke_grad_edge(iedge, k, ke_cell);
        }

        if (ssh_grad_edge.m_enabled) {
          vn_tend_edge_accum -= ssh_grad_edge(iedge, k, h_cell);
        }

        if (vel_diff_edge.m_enabled) {
          vn_tend_edge_accum +=
              vel_diff_edge(iedge, k, vel_div_cell, rvort_vertex);
        }

        if (vel_hyperdiff_edge.m_enabled) {
          vn_tend_edge_accum += vel_hyperdiff_edge(iedge, k, vel_del2_div_cell,
                                                   vel_del2_rvort_vertex);
        }

        vn_tend_edge(iedge, k) = vn_tend_edge_accum;
      });
}

void ShallowWaterModel::compute_tr_tendency(Real3d tr_tend_cell,
                                            RealConst3d tr_cell,
                                            RealConst2d vn_edge,
                                            AddMode add_mode) const {
  OMEGA_SCOPE(h_flux_edge, m_aux_state.m_h_flux_edge.const_array());
  OMEGA_SCOPE(h_mean_edge, m_aux_state.m_h_mean_edge.const_array());
  OMEGA_SCOPE(norm_tr_cell, m_aux_state.m_norm_tr_cell.const_array());
  OMEGA_SCOPE(tr_del2_cell, m_aux_state.m_tr_del2_cell.const_array());
  OMEGA_SCOPE(tracer_hadv_cell, m_tracer_hadv_cell);
  OMEGA_SCOPE(tracer_diff_cell, m_tracer_diff_cell);
  OMEGA_SCOPE(tracer_hyperdiff_cell, m_tracer_hyperdiff_cell);

  omega_parallel_for(
      "compute_tr_tend",
      {m_params.m_ntracers, m_mesh->m_ncells, m_mesh->m_nlayers},
      KOKKOS_LAMBDA(Int l, Int icell, Int k) {
        Real tr_tend_cell_accum =
            add_mode == AddMode::increment ? tr_tend_cell(l, icell, k) : 0;

        if (tracer_hadv_cell.m_enabled) {
          tr_tend_cell_accum -=
              tracer_hadv_cell(l, icell, k, vn_edge, norm_tr_cell, h_flux_edge);
        }

        if (tracer_diff_cell.m_enabled) {
          tr_tend_cell_accum +=
              tracer_diff_cell(l, icell, k, norm_tr_cell, h_mean_edge);
        }

        if (tracer_hyperdiff_cell.m_enabled) {
          tr_tend_cell_accum +=
              tracer_hyperdiff_cell(l, icell, k, tr_del2_cell);
        }

        tr_tend_cell(l, icell, k) = tr_tend_cell_accum;
      });
}

void ShallowWaterModel::compute_tendency(const ShallowWaterState &tend,
                                         const ShallowWaterState &state, Real t,
                                         AddMode add_mode) const {

  m_aux_state.compute(state.m_h_cell, state.m_vn_edge, state.m_tr_cell,
                      m_f_vertex, m_mesh);

  if (!m_params.m_disable_h_tendency) {
    compute_h_tendency(tend.m_h_cell, state.m_h_cell, state.m_vn_edge,
                       add_mode);
  }

  if (!m_params.m_disable_vn_tendency) {
    compute_vn_tendency(tend.m_vn_edge, state.m_h_cell, state.m_vn_edge,
                        add_mode);
  }

  if (m_params.m_ntracers > 0) {
    compute_tr_tendency(tend.m_tr_cell, state.m_tr_cell, state.m_vn_edge,
                        add_mode);
  }

  additional_tendency(tend.m_h_cell, tend.m_vn_edge, state.m_h_cell,
                      state.m_vn_edge, t);
}

} // namespace omega