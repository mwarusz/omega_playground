#include "shallow_water_model.hpp"

namespace omega {

ShallowWaterModel::ShallowWaterModel(MPASMesh *mesh,
                                     const ShallowWaterParams &params)
    : m_params(params), m_mesh(mesh), m_aux_state(mesh, params.m_ntracers),
      m_thickness_hadv_cell(mesh), m_pv_flux_edge(mesh), m_ke_grad_edge(mesh),
      m_ssh_grad_edge(mesh, params.m_grav),
      m_vel_diff_edge(mesh, params.m_visc_del2),
      m_vel_hyperdiff_edge(mesh, params.m_visc_del4), m_tracer_hadv_cell(mesh),
      m_tracer_diff_cell(mesh, params.m_eddy_diff2),
      m_tracer_hyperdiff_cell(mesh, params.m_ntracers, params.m_eddy_diff4),
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
  const auto flux_h_edge =
      const_view(m_aux_state.m_thickness_aux.m_flux_h_edge);
  OMEGA_SCOPE(thickness_hadv_cell, m_thickness_hadv_cell);

#ifdef OMEGA_NO_INNER_IF
  if (add_mode == AddMode::replace) {
    omega_parallel_for(
        "compute_h_tend", {m_mesh->m_ncells, m_mesh->m_nlayers_vec},
        KOKKOS_LAMBDA(Int icell, Int kchunk) {
          for (Int kvec = 0; kvec < vector_length; ++kvec) {
            const Int k = kchunk * vector_length + kvec;
            h_tend_cell(icell, k) = 0;
          }
        });
  }

  if (thickness_hadv_cell.m_enabled) {
    omega_parallel_for(
        "compute_h_tend", {m_mesh->m_ncells, m_mesh->m_nlayers_vec},
        KOKKOS_LAMBDA(Int icell, Int kchunk) {
          thickness_hadv_cell(h_tend_cell, icell, kchunk, vn_edge, flux_h_edge);
        });
  }
#else
  omega_parallel_for(
      "compute_h_tend", {m_mesh->m_ncells, m_mesh->m_nlayers_vec},
      KOKKOS_LAMBDA(Int icell, Int kchunk) {
        if (add_mode == AddMode::replace) {
          for (Int kvec = 0; kvec < vector_length; ++kvec) {
            const Int k = kchunk * vector_length + kvec;
            h_tend_cell(icell, k) = 0;
          }
        }

        if (thickness_hadv_cell.m_enabled) {
          thickness_hadv_cell(h_tend_cell, icell, kchunk, vn_edge, flux_h_edge);
        }
      });
#endif
}

void ShallowWaterModel::compute_vn_tendency(Real2d vn_tend_edge,
                                            RealConst2d h_cell,
                                            RealConst2d vn_edge,
                                            AddMode add_mode) const {

  const auto flux_h_edge =
      const_view(m_aux_state.m_thickness_aux.m_flux_h_edge);
  const auto rvort_vertex =
      const_view(m_aux_state.m_vorticity_aux.m_rvort_vertex);
  const auto norm_rvort_edge =
      const_view(m_aux_state.m_vorticity_aux.m_norm_rvort_edge);
  const auto norm_pvort_edge =
      const_view(m_aux_state.m_vorticity_aux.m_norm_pvort_edge);
  const auto ke_cell = const_view(m_aux_state.m_kinetic_aux.m_ke_cell);
  const auto vel_div_cell =
      const_view(m_aux_state.m_kinetic_aux.m_vel_div_cell);

  OMEGA_SCOPE(pv_flux_edge, m_pv_flux_edge);
  OMEGA_SCOPE(ke_grad_edge, m_ke_grad_edge);
  OMEGA_SCOPE(ssh_grad_edge, m_ssh_grad_edge);
  OMEGA_SCOPE(vel_diff_edge, m_vel_diff_edge);
  OMEGA_SCOPE(vel_hyperdiff_edge, m_vel_hyperdiff_edge);

  if (vel_hyperdiff_edge.m_enabled) {
    omega_parallel_for(
        "compute_vel_del2", {m_mesh->m_nedges, m_mesh->m_nlayers_vec},
        KOKKOS_LAMBDA(Int iedge, Int kchunk) {
          vel_hyperdiff_edge.compute_vel_del2(iedge, kchunk, vel_div_cell,
                                              rvort_vertex);
        });

    omega_parallel_for(
        "compute_vel_del2_rvort", {m_mesh->m_nvertices, m_mesh->m_nlayers_vec},
        KOKKOS_LAMBDA(Int ivertex, Int kchunk) {
          vel_hyperdiff_edge.compute_vel_del2_rvort(ivertex, kchunk);
        });

    omega_parallel_for(
        "compute_vel_del2_div", {m_mesh->m_ncells, m_mesh->m_nlayers_vec},
        KOKKOS_LAMBDA(Int icell, Int kchunk) {
          vel_hyperdiff_edge.compute_vel_del2_div(icell, kchunk);
        });
  }

#ifdef OMEGA_NO_INNER_IF
  if (add_mode == AddMode::replace) {
    omega_parallel_for(
        "compute_vtend1", {m_mesh->m_nedges, m_mesh->m_nlayers_vec},
        KOKKOS_LAMBDA(Int iedge, Int kchunk) {
          for (Int kvec = 0; kvec < vector_length; ++kvec) {
            const Int k = kchunk * vector_length + kvec;
            vn_tend_edge(iedge, k) = 0;
          }
        });
  }

  if (pv_flux_edge.m_enabled) {
    omega_parallel_for(
        "compute_vtend2", {m_mesh->m_nedges, m_mesh->m_nlayers_vec},
        KOKKOS_LAMBDA(Int iedge, Int kchunk) {
          pv_flux_edge(vn_tend_edge, iedge, kchunk, norm_rvort_edge,
                       norm_pvort_edge, flux_h_edge, vn_edge);
        });
  }

  if (ke_grad_edge.m_enabled) {
    omega_parallel_for(
        "compute_vtend3", {m_mesh->m_nedges, m_mesh->m_nlayers_vec},
        KOKKOS_LAMBDA(Int iedge, Int kchunk) {
          ke_grad_edge(vn_tend_edge, iedge, kchunk, ke_cell);
        });
  }

  if (ssh_grad_edge.m_enabled) {
    omega_parallel_for(
        "compute_vtend4", {m_mesh->m_nedges, m_mesh->m_nlayers_vec},
        KOKKOS_LAMBDA(Int iedge, Int kchunk) {
          ssh_grad_edge(vn_tend_edge, iedge, kchunk, h_cell);
        });
  }

  if (vel_diff_edge.m_enabled) {
    omega_parallel_for(
        "compute_vtend5", {m_mesh->m_nedges, m_mesh->m_nlayers_vec},
        KOKKOS_LAMBDA(Int iedge, Int kchunk) {
          vel_diff_edge(vn_tend_edge, iedge, kchunk, vel_div_cell,
                        rvort_vertex);
        });
  }

  if (vel_hyperdiff_edge.m_enabled) {
    omega_parallel_for(
        "compute_vtend6", {m_mesh->m_nedges, m_mesh->m_nlayers_vec},
        KOKKOS_LAMBDA(Int iedge, Int kchunk) {
          vel_hyperdiff_edge(vn_tend_edge, iedge, kchunk);
        });
  }
#else
  omega_parallel_for(
      "compute_vtend", {m_mesh->m_nedges, m_mesh->m_nlayers_vec},
      KOKKOS_LAMBDA(Int iedge, Int kchunk) {
        if (add_mode == AddMode::replace) {
          for (Int kvec = 0; kvec < vector_length; ++kvec) {
            const Int k = kchunk * vector_length + kvec;
            vn_tend_edge(iedge, k) = 0;
          }
        }

        if (pv_flux_edge.m_enabled) {
          pv_flux_edge(vn_tend_edge, iedge, kchunk, norm_rvort_edge,
                       norm_pvort_edge, flux_h_edge, vn_edge);
        }

        if (ke_grad_edge.m_enabled) {
          ke_grad_edge(vn_tend_edge, iedge, kchunk, ke_cell);
        }

        if (ssh_grad_edge.m_enabled) {
          ssh_grad_edge(vn_tend_edge, iedge, kchunk, h_cell);
        }

        if (vel_diff_edge.m_enabled) {
          vel_diff_edge(vn_tend_edge, iedge, kchunk, vel_div_cell,
                        rvort_vertex);
        }

        if (vel_hyperdiff_edge.m_enabled) {
          vel_hyperdiff_edge(vn_tend_edge, iedge, kchunk);
        }
      });
#endif
}

void ShallowWaterModel::compute_tr_tendency(Real3d tr_tend_cell,
                                            RealConst3d tr_cell,
                                            RealConst2d vn_edge,
                                            AddMode add_mode) const {
  const auto flux_h_edge =
      const_view(m_aux_state.m_thickness_aux.m_mean_h_edge);
  const auto mean_h_edge =
      const_view(m_aux_state.m_thickness_aux.m_flux_h_edge);
  const auto norm_tr_cell = const_view(m_aux_state.m_tracer_aux.m_norm_tr_cell);
  OMEGA_SCOPE(tracer_hadv_cell, m_tracer_hadv_cell);
  OMEGA_SCOPE(tracer_diff_cell, m_tracer_diff_cell);
  OMEGA_SCOPE(tracer_hyperdiff_cell, m_tracer_hyperdiff_cell);

  if (tracer_hyperdiff_cell.m_enabled) {
    omega_parallel_for(
        "compute_tr_del2",
        {m_params.m_ntracers, m_mesh->m_ncells, m_mesh->m_nlayers_vec},
        KOKKOS_LAMBDA(Int l, Int icell, Int kchunk) {
          tracer_hyperdiff_cell.compute_tracer_del2(l, icell, kchunk,
                                                    norm_tr_cell, mean_h_edge);
        });
  }

  const auto tr_del2_cell =
      const_view(tracer_hyperdiff_cell.m_tracer_del2_cell);
#ifdef OMEGA_NO_INNER_IF
  if (add_mode == AddMode::replace) {
    omega_parallel_for(
        "compute_tr_tend",
        {m_params.m_ntracers, m_mesh->m_ncells, m_mesh->m_nlayers_vec},
        KOKKOS_LAMBDA(Int l, Int icell, Int kchunk) {
          for (Int kvec = 0; kvec < vector_length; ++kvec) {
            const Int k = kchunk * vector_length + kvec;
            tr_tend_cell(l, icell, k) = 0;
          }
        });
  }
  if (tracer_hadv_cell.m_enabled) {
    omega_parallel_for(
        "compute_tr_tend",
        {m_params.m_ntracers, m_mesh->m_ncells, m_mesh->m_nlayers_vec},
        KOKKOS_LAMBDA(Int l, Int icell, Int kchunk) {
          tracer_hadv_cell(tr_tend_cell, l, icell, kchunk, vn_edge,
                           norm_tr_cell, flux_h_edge);
        });
  }
  if (tracer_diff_cell.m_enabled) {
    omega_parallel_for(
        "compute_tr_tend",
        {m_params.m_ntracers, m_mesh->m_ncells, m_mesh->m_nlayers_vec},
        KOKKOS_LAMBDA(Int l, Int icell, Int kchunk) {
          tracer_diff_cell(tr_tend_cell, l, icell, kchunk, norm_tr_cell,
                           mean_h_edge);
        });
  }
  if (tracer_hyperdiff_cell.m_enabled) {
    omega_parallel_for(
        "compute_tr_tend",
        {m_params.m_ntracers, m_mesh->m_ncells, m_mesh->m_nlayers_vec},
        KOKKOS_LAMBDA(Int l, Int icell, Int kchunk) {
          tracer_hyperdiff_cell(tr_tend_cell, l, icell, kchunk, tr_del2_cell);
        });
  }
#else
  omega_parallel_for(
      "compute_tr_tend",
      {m_params.m_ntracers, m_mesh->m_ncells, m_mesh->m_nlayers_vec},
      KOKKOS_LAMBDA(Int l, Int icell, Int kchunk) {
        if (add_mode == AddMode::replace) {
          for (Int kvec = 0; kvec < vector_length; ++kvec) {
            const Int k = kchunk * vector_length + kvec;
            tr_tend_cell(l, icell, k) = 0;
          }
        }

        if (tracer_hadv_cell.m_enabled) {
          tracer_hadv_cell(tr_tend_cell, l, icell, kchunk, vn_edge,
                           norm_tr_cell, flux_h_edge);
        }

        if (tracer_diff_cell.m_enabled) {
          tracer_diff_cell(tr_tend_cell, l, icell, kchunk, norm_tr_cell,
                           mean_h_edge);
        }

        if (tracer_hyperdiff_cell.m_enabled) {
          tracer_hyperdiff_cell(tr_tend_cell, l, icell, kchunk, tr_del2_cell);
        }
      });
#endif
}

void ShallowWaterModel::compute_tendency(const ShallowWaterState &tend,
                                         const ShallowWaterState &state, Real t,
                                         AddMode add_mode) const {

  timer_start("auxstate");
  m_aux_state.compute(state.m_h_cell, state.m_vn_edge, state.m_tr_cell,
                      m_f_vertex, m_mesh);
  timer_end("auxstate");

  timer_start("tendencies");
  if (!m_params.m_disable_h_tendency) {
    timer_start("h_tendency");
    compute_h_tendency(tend.m_h_cell, state.m_h_cell, state.m_vn_edge,
                       add_mode);
    timer_end("h_tendency");
  }

  if (!m_params.m_disable_vn_tendency) {
    timer_start("vn_tendency");
    compute_vn_tendency(tend.m_vn_edge, state.m_h_cell, state.m_vn_edge,
                        add_mode);
    timer_end("vn_tendency");
  }

  if (m_params.m_ntracers > 0) {
    timer_start("tr_tendency");
    compute_tr_tendency(tend.m_tr_cell, state.m_tr_cell, state.m_vn_edge,
                        add_mode);
    timer_end("tr_tendency");
  }

  additional_tendency(tend.m_h_cell, tend.m_vn_edge, state.m_h_cell,
                      state.m_vn_edge, t);
  timer_end("tendencies");
}

} // namespace omega
