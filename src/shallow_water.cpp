#include <shallow_water.hpp>

namespace omega {

// State

ShallowWaterState::ShallowWaterState(MPASMesh *mesh, Int ntracers)
    : m_h_cell("h_cell", mesh->m_ncells, mesh->m_nlayers),
      m_vn_edge("vn_edge", mesh->m_nedges, mesh->m_nlayers),
      m_tr_cell("tr_cell", ntracers, mesh->m_ncells, mesh->m_nlayers) {}

ShallowWaterState::ShallowWaterState(const ShallowWaterModel &sw)
    : ShallowWaterState(sw.m_mesh, sw.m_ntracers) {}

ShallowWaterModel::ShallowWaterModel(MPASMesh *mesh,
                                             const ShallowWaterParams &params)
    : m_mesh(mesh), m_grav(params.m_grav),
      m_disable_h_tendency(params.m_disable_h_tendency),
      m_disable_vn_tendency(params.m_disable_vn_tendency),
      m_ntracers(params.m_ntracers), m_f_vertex("f_vertex", mesh->m_nvertices),
      m_f_edge("f_edge", mesh->m_nedges),
      m_drag_coeff(params.m_drag_coeff),
      m_visc_del2(params.m_visc_del2), m_visc_del4(params.m_visc_del4),
      m_eddy_diff2(params.m_eddy_diff2), m_eddy_diff4(params.m_eddy_diff4),
      m_ke_cell("ke_cell", mesh->m_ncells, mesh->m_nlayers),
      m_div_cell("div_cell", mesh->m_ncells, mesh->m_nlayers),
      m_norm_tr_cell("norm_tr_cell", params.m_ntracers, mesh->m_ncells,
                     mesh->m_nlayers),
      m_h_flux_edge("h_flux_edge", mesh->m_nedges, mesh->m_nlayers),
      m_h_mean_edge("h_mean_edge", mesh->m_nedges, mesh->m_nlayers),
      m_h_drag_edge("h_drag_edge", mesh->m_nedges, mesh->m_nlayers),
      m_norm_rvort_edge("norm_rvort_edge", mesh->m_nedges, mesh->m_nlayers),
      m_norm_f_edge("norm_f_edge", mesh->m_nedges, mesh->m_nlayers),
      m_rvort_vertex("rvort_vertex", mesh->m_nvertices, mesh->m_nlayers),
      m_norm_rvort_vertex("norm_rvort_vertex", mesh->m_nvertices,
                          mesh->m_nlayers),
      m_norm_f_vertex("norm_f_vertex", mesh->m_nvertices, mesh->m_nlayers) {

  deep_copy(m_f_vertex, params.m_f0);
  deep_copy(m_f_edge, params.m_f0);
}

void ShallowWaterModel::compute_auxiliary_variables(RealConst2d h_cell,
                                                    RealConst2d vn_edge,
                                                    RealConst3d tr_cell) const {
  compute_vertex_auxiliary_variables(h_cell, vn_edge, tr_cell);
  compute_cell_auxiliary_variables(h_cell, vn_edge, tr_cell);
  compute_edge_auxiliary_variables(h_cell, vn_edge, tr_cell);
}

void ShallowWaterModel::compute_tendency(const ShallowWaterState &tend,
                                             const ShallowWaterState &state,
                                             Real t, AddMode add_mode) const {

  compute_auxiliary_variables(state.m_h_cell, state.m_vn_edge, state.m_tr_cell);

  if (!m_disable_h_tendency) {
    compute_h_tendency(tend.m_h_cell, state.m_h_cell, state.m_vn_edge,
                       add_mode);
  }

  if (!m_disable_vn_tendency) {
    compute_vn_tendency(tend.m_vn_edge, state.m_h_cell, state.m_vn_edge,
                        add_mode);
  }

  if (m_ntracers > 0) {
    compute_tr_tendency(tend.m_tr_cell, state.m_tr_cell, state.m_vn_edge,
                        add_mode);
  }

  additional_tendency(tend.m_h_cell, tend.m_vn_edge, state.m_h_cell,
                      state.m_vn_edge, t);
}

Real ShallowWaterModel::mass_integral(RealConst2d h_cell) const {
  OMEGA_SCOPE(area_cell, m_mesh->m_area_cell);
  OMEGA_SCOPE(max_level_cell, m_mesh->m_max_level_cell);

  Real total_mass;
  omega_parallel_reduce(
      "compute_column_mass", {m_mesh->m_ncells},
      KOKKOS_LAMBDA(Int icell, Real & column_mass) {
        for (Int k = 0; k < max_level_cell(icell); ++k) {
          column_mass += area_cell(icell) * h_cell(icell, k);
        }
      },
      total_mass);
  return total_mass;
}

Real ShallowWaterModel::circulation_integral(RealConst2d vn_edge) const {

  OMEGA_SCOPE(dc_edge, m_mesh->m_dc_edge);
  OMEGA_SCOPE(edges_on_vertex, m_mesh->m_edges_on_vertex);
  OMEGA_SCOPE(edge_sign_on_vertex, m_mesh->m_edge_sign_on_vertex);
  OMEGA_SCOPE(area_triangle, m_mesh->m_area_triangle);
  OMEGA_SCOPE(f_vertex, m_f_vertex);
  OMEGA_SCOPE(max_level_vertex_bot, m_mesh->m_max_level_vertex_bot);

  Real total_circulation;
  omega_parallel_reduce(
      "compute_column_circulation", {m_mesh->m_nvertices},
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

void ShallowWaterModel::compute_cell_auxiliary_variables(
    RealConst2d h_cell, RealConst2d vn_edge, RealConst3d tr_cell) const {
  OMEGA_SCOPE(nedges_on_cell, m_mesh->m_nedges_on_cell);
  OMEGA_SCOPE(edges_on_cell, m_mesh->m_edges_on_cell);
  OMEGA_SCOPE(edge_sign_on_cell, m_mesh->m_edge_sign_on_cell);
  OMEGA_SCOPE(dv_edge, m_mesh->m_dv_edge);
  OMEGA_SCOPE(dc_edge, m_mesh->m_dc_edge);
  OMEGA_SCOPE(area_cell, m_mesh->m_area_cell);

  OMEGA_SCOPE(ke_cell, m_ke_cell);
  OMEGA_SCOPE(div_cell, m_div_cell);
  OMEGA_SCOPE(norm_tr_cell, m_norm_tr_cell);
  OMEGA_SCOPE(ntracers, m_ntracers);

  omega_parallel_for(
      "compute_cell_auxiliarys", {m_mesh->m_ncells, m_mesh->m_nlayers},
      KOKKOS_LAMBDA(Int icell, Int k) {
        Real ke = -0;
        Real div = -0;
        for (Int j = 0; j < nedges_on_cell(icell); ++j) {
          Int jedge = edges_on_cell(icell, j);
          Real area_edge = dv_edge(jedge) * dc_edge(jedge);
          ke += area_edge * vn_edge(jedge, k) * vn_edge(jedge, k) * 0.25_fp;
          div +=
              dv_edge(jedge) * edge_sign_on_cell(icell, j) * vn_edge(jedge, k);
        }
        Real inv_area_cell = 1._fp / area_cell(icell);
        ke *= inv_area_cell;
        div *= inv_area_cell;

        div_cell(icell, k) = div;
        ke_cell(icell, k) = ke;

        Real inv_h = 1._fp / h_cell(icell, k);
        for (Int l = 0; l < ntracers; ++l) {
          norm_tr_cell(l, icell, k) = tr_cell(l, icell, k) * inv_h;
        }
      });
}

void ShallowWaterModel::compute_vertex_auxiliary_variables(
    RealConst2d h_cell, RealConst2d vn_edge, RealConst3d tr_cell) const {
  OMEGA_SCOPE(dc_edge, m_mesh->m_dc_edge);
  OMEGA_SCOPE(edges_on_vertex, m_mesh->m_edges_on_vertex);
  OMEGA_SCOPE(edge_sign_on_vertex, m_mesh->m_edge_sign_on_vertex);
  OMEGA_SCOPE(area_triangle, m_mesh->m_area_triangle);
  OMEGA_SCOPE(kiteareas_on_vertex, m_mesh->m_kiteareas_on_vertex);
  OMEGA_SCOPE(cells_on_vertex, m_mesh->m_cells_on_vertex);

  OMEGA_SCOPE(rvort_vertex, m_rvort_vertex);
  OMEGA_SCOPE(f_vertex, m_f_vertex);
  OMEGA_SCOPE(norm_rvort_vertex, m_norm_rvort_vertex);
  OMEGA_SCOPE(norm_f_vertex, m_norm_f_vertex);

  omega_parallel_for(
      "compute_vertex_auxiliarys", {m_mesh->m_nvertices, m_mesh->m_nlayers},
      KOKKOS_LAMBDA(Int ivertex, Int k) {
        Real inv_area_triangle = 1._fp / area_triangle(ivertex);
        Real rcirc = -0;
        for (Int j = 0; j < 3; ++j) {
          Int jedge = edges_on_vertex(ivertex, j);
          rcirc += dc_edge(jedge) * edge_sign_on_vertex(ivertex, j) *
                   vn_edge(jedge, k);
        }
        Real rvort = rcirc * inv_area_triangle;

        Real h = -0;
        for (Int j = 0; j < 3; ++j) {
          Int jcell = cells_on_vertex(ivertex, j);
          h += kiteareas_on_vertex(ivertex, j) * h_cell(jcell, k);
        }
        h *= inv_area_triangle;
        Real inv_h = 1._fp / h;

        rvort_vertex(ivertex, k) = rvort;
        norm_rvort_vertex(ivertex, k) = rvort * inv_h;
        norm_f_vertex(ivertex, k) = f_vertex(ivertex) * inv_h;
      });
}

void ShallowWaterModel::compute_edge_auxiliary_variables(
    RealConst2d h_cell, RealConst2d vn_edge, RealConst3d tr_cell) const {

  OMEGA_SCOPE(cells_on_edge, m_mesh->m_cells_on_edge);
  OMEGA_SCOPE(vertices_on_edge, m_mesh->m_vertices_on_edge);

  OMEGA_SCOPE(h_mean_edge, m_h_mean_edge);
  OMEGA_SCOPE(h_flux_edge, m_h_flux_edge);
  OMEGA_SCOPE(h_drag_edge, m_h_drag_edge);
  OMEGA_SCOPE(norm_rvort_edge, m_norm_rvort_edge);
  OMEGA_SCOPE(norm_f_edge, m_norm_f_edge);
  OMEGA_SCOPE(norm_rvort_vertex, m_norm_rvort_vertex);
  OMEGA_SCOPE(norm_f_vertex, m_norm_f_vertex);

  omega_parallel_for(
      "compute_edge_auxiliarys", {m_mesh->m_nedges, m_mesh->m_nlayers},
      KOKKOS_LAMBDA(Int iedge, Int k) {
        Real h_mean = -0;
        for (Int j = 0; j < 2; ++j) {
          Int jcell = cells_on_edge(iedge, j);
          h_mean += h_cell(jcell, k);
        }
        h_mean *= 0.5_fp;

        Real norm_f = -0;
        Real norm_rvort = -0;
        for (Int j = 0; j < 2; ++j) {
          Int jvertex = vertices_on_edge(iedge, j);
          norm_rvort += norm_rvort_vertex(jvertex, k);
          norm_f += norm_f_vertex(jvertex, k);
        }
        norm_rvort *= 0.5_fp;
        norm_f *= 0.5_fp;

        h_mean_edge(iedge, k) = h_mean;
        h_flux_edge(iedge, k) = h_mean;
        h_drag_edge(iedge, k) = h_mean;

        norm_rvort_edge(iedge, k) = norm_rvort;
        norm_f_edge(iedge, k) = norm_f;
      });
}

void ShallowWaterModel::compute_h_tendency(Real2d h_tend_cell,
                                           RealConst2d h_cell,
                                           RealConst2d vn_edge,
                                           AddMode add_mode) const {
  OMEGA_SCOPE(nedges_on_cell, m_mesh->m_nedges_on_cell);
  OMEGA_SCOPE(edges_on_cell, m_mesh->m_edges_on_cell);
  OMEGA_SCOPE(dv_edge, m_mesh->m_dv_edge);
  OMEGA_SCOPE(edge_sign_on_cell, m_mesh->m_edge_sign_on_cell);
  OMEGA_SCOPE(area_cell, m_mesh->m_area_cell);

  OMEGA_SCOPE(h_flux_edge, m_h_flux_edge);
  omega_parallel_for(
      "compute_h_tend", {m_mesh->m_ncells, m_mesh->m_nlayers},
      KOKKOS_LAMBDA(Int icell, Int k) {
        Real accum = -0;
        for (Int j = 0; j < nedges_on_cell(icell); ++j) {
          Int jedge = edges_on_cell(icell, j);
          accum += dv_edge(jedge) * edge_sign_on_cell(icell, j) *
                   h_flux_edge(jedge, k) * vn_edge(jedge, k);
        }

        Real inv_area_cell = 1._fp / area_cell(icell);
        if (add_mode == AddMode::increment) {
          h_tend_cell(icell, k) += -accum * inv_area_cell;
        }

        if (add_mode == AddMode::replace) {
          h_tend_cell(icell, k) = -accum * inv_area_cell;
        }
      });
}

void ShallowWaterModel::compute_vn_tendency(Real2d vn_tend_edge,
                                            RealConst2d h_cell,
                                            RealConst2d vn_edge,
                                            AddMode add_mode) const {
  OMEGA_SCOPE(max_level_edge_top, m_mesh->m_max_level_edge_top);
  OMEGA_SCOPE(nedges_on_edge, m_mesh->m_nedges_on_edge);
  OMEGA_SCOPE(edges_on_edge, m_mesh->m_edges_on_edge);
  OMEGA_SCOPE(weights_on_edge, m_mesh->m_weights_on_edge);
  OMEGA_SCOPE(dc_edge, m_mesh->m_dc_edge);
  OMEGA_SCOPE(dv_edge, m_mesh->m_dv_edge);
  OMEGA_SCOPE(cells_on_edge, m_mesh->m_cells_on_edge);
  OMEGA_SCOPE(vertices_on_edge, m_mesh->m_vertices_on_edge);
  OMEGA_SCOPE(edge_mask, m_mesh->m_edge_mask);
  OMEGA_SCOPE(mesh_scaling_del2, m_mesh->m_mesh_scaling_del2);
  OMEGA_SCOPE(mesh_scaling_del4, m_mesh->m_mesh_scaling_del4);

  OMEGA_SCOPE(grav, m_grav);
  OMEGA_SCOPE(norm_rvort_edge, m_norm_rvort_edge);
  OMEGA_SCOPE(norm_f_edge, m_norm_f_edge);
  OMEGA_SCOPE(h_flux_edge, m_h_flux_edge);
  // OMEGA_SCOPE(h_drag_edge, m_h_drag_edge);
  OMEGA_SCOPE(ke_cell, m_ke_cell);
  OMEGA_SCOPE(div_cell, m_div_cell);
  OMEGA_SCOPE(rvort_vertex, m_rvort_vertex);
  // OMEGA_SCOPE(drag_coeff, m_drag_coeff);
  OMEGA_SCOPE(visc_del2, m_visc_del2);
  OMEGA_SCOPE(visc_del4, m_visc_del4);

  Real2d del2u_edge, del2rvort_vertex, del2div_cell;
  if (visc_del4 > 0) {
    del2u_edge = Real2d("del2u_edge", m_mesh->m_nedges, m_mesh->m_nlayers);
    del2rvort_vertex =
        Real2d("del2rvort_vertex", m_mesh->m_nvertices, m_mesh->m_nlayers);
    del2div_cell = Real2d("del2div_cell", m_mesh->m_ncells, m_mesh->m_nlayers);

    OMEGA_SCOPE(nedges_on_cell, m_mesh->m_nedges_on_cell);
    OMEGA_SCOPE(edges_on_cell, m_mesh->m_edges_on_cell);
    OMEGA_SCOPE(edge_sign_on_cell, m_mesh->m_edge_sign_on_cell);
    OMEGA_SCOPE(area_cell, m_mesh->m_area_cell);
    OMEGA_SCOPE(edges_on_vertex, m_mesh->m_edges_on_vertex);
    OMEGA_SCOPE(area_triangle, m_mesh->m_area_triangle);
    OMEGA_SCOPE(edge_sign_on_vertex, m_mesh->m_edge_sign_on_vertex);

    omega_parallel_for(
        "compute_del2u_edge", {m_mesh->m_nedges, m_mesh->m_nlayers},
        KOKKOS_LAMBDA(Int iedge, Int k) {
          Int icell0 = cells_on_edge(iedge, 0);
          Int icell1 = cells_on_edge(iedge, 1);

          Int ivertex0 = vertices_on_edge(iedge, 0);
          Int ivertex1 = vertices_on_edge(iedge, 1);

          Real dc_edge_inv = 1._fp / dc_edge(iedge);
          Real dv_edge_inv =
              1._fp / std::max(dv_edge(iedge), 0.25_fp * dc_edge(iedge)); // huh

          Real del2u =
              ((div_cell(icell1, k) - div_cell(icell0, k)) * dc_edge_inv -
               (rvort_vertex(ivertex1, k) - rvort_vertex(ivertex0, k)) *
                   dv_edge_inv);

          del2u_edge(iedge, k) = del2u;
        });

    omega_parallel_for(
        "compute_del2div_cell", {m_mesh->m_ncells, m_mesh->m_nlayers},
        KOKKOS_LAMBDA(Int icell, Int k) {
          Real del2div = -0;
          for (Int j = 0; j < nedges_on_cell(icell); ++j) {
            Int jedge = edges_on_cell(icell, j);
            del2div += dv_edge(jedge) * edge_sign_on_cell(icell, j) *
                       del2u_edge(jedge, k);
          }
          Real inv_area_cell = 1._fp / area_cell(icell);
          del2div *= inv_area_cell;
          del2div_cell(icell, k) = del2div;
        });

    omega_parallel_for(
        "compute_del2rvort_vertex", {m_mesh->m_nvertices, m_mesh->m_nlayers},
        KOKKOS_LAMBDA(Int ivertex, Int k) {
          Real del2rvort = -0;
          for (Int j = 0; j < 3; ++j) {
            Int jedge = edges_on_vertex(ivertex, j);
            del2rvort += dc_edge(jedge) * edge_sign_on_vertex(ivertex, j) *
                         vn_edge(jedge, k);
          }
          Real inv_area_triangle = 1._fp / area_triangle(ivertex);
          del2rvort *= inv_area_triangle;

          del2rvort_vertex(ivertex, k) = del2rvort;
        });
  }

  omega_parallel_for(
      "compute_vtend", {m_mesh->m_nedges, m_mesh->m_nlayers},
      KOKKOS_LAMBDA(Int iedge, Int k) {
        Real vn_tend = -0;

        Real qt = -0;
        for (Int j = 0; j < nedges_on_edge(iedge); ++j) {
          Int jedge = edges_on_edge(iedge, j);

          Real norm_vort = (norm_rvort_edge(iedge, k) + norm_f_edge(iedge, k) +
                            norm_rvort_edge(jedge, k) + norm_f_edge(jedge, k)) *
                           0.5_fp;

          qt += weights_on_edge(iedge, j) * h_flux_edge(jedge, k) *
                vn_edge(jedge, k) * norm_vort;
        }

        Int icell0 = cells_on_edge(iedge, 0);
        Int icell1 = cells_on_edge(iedge, 1);

        Real ke_cell0 = ke_cell(icell0, k);
        Real ke_cell1 = ke_cell(icell1, k);

        Real inv_dc_edge = 1._fp / dc_edge(iedge);
        Real grad_B = (ke_cell1 - ke_cell0 +
                       grav * (h_cell(icell1, k) - h_cell(icell0, k))) *
                      inv_dc_edge;

        vn_tend = qt - grad_B;

        // Real inv_h_drag_edge = 1._fp / h_drag_edge(iedge, k);
        // Real drag_force = (k == (max_level_edge_top(iedge) - 1))
        //                       ? -drag_coeff * std::sqrt(ke_cell0 + ke_cell1)
        //                       *
        //                             vn_edge(iedge, k) * inv_h_drag_edge
        //                       : 0;
        // vn_tend += drag_force;

        Real inv_dv_edge = 1._fp / dv_edge(iedge);
        // viscosity
        if (visc_del2 > 0) {
          Int ivertex0 = vertices_on_edge(iedge, 0);
          Int ivertex1 = vertices_on_edge(iedge, 1);
          Real visc2 =
              visc_del2 * mesh_scaling_del2(iedge) *
              ((div_cell(icell1, k) - div_cell(icell0, k)) * inv_dc_edge -
               (rvort_vertex(ivertex1, k) - rvort_vertex(ivertex0, k)) *
                   inv_dv_edge);
          vn_tend += visc2 * edge_mask(iedge, k);
        }

        // hyperviscosity
        if (visc_del4 > 0) {
          Int ivertex0 = vertices_on_edge(iedge, 0);
          Int ivertex1 = vertices_on_edge(iedge, 1);
          Real visc4 =
              visc_del4 * mesh_scaling_del4(iedge) *
              ((del2div_cell(icell1, k) - del2div_cell(icell0, k)) *
                   inv_dc_edge -
               (del2rvort_vertex(ivertex1, k) - del2rvort_vertex(ivertex0, k)) *
                   inv_dv_edge);
          vn_tend -= visc4 * edge_mask(iedge, k);
        }

        if (add_mode == AddMode::increment) {
          vn_tend_edge(iedge, k) += vn_tend;
        }
        if (add_mode == AddMode::replace) {
          vn_tend_edge(iedge, k) = vn_tend;
        }
      });
}

void ShallowWaterModel::compute_tr_tendency(Real3d tr_tend_cell,
                                            RealConst3d tr_cell,
                                            RealConst2d vn_edge,
                                            AddMode add_mode) const {
  OMEGA_SCOPE(nedges_on_cell, m_mesh->m_nedges_on_cell);
  OMEGA_SCOPE(edges_on_cell, m_mesh->m_edges_on_cell);
  OMEGA_SCOPE(dv_edge, m_mesh->m_dv_edge);
  OMEGA_SCOPE(dc_edge, m_mesh->m_dc_edge);
  OMEGA_SCOPE(edge_sign_on_cell, m_mesh->m_edge_sign_on_cell);
  OMEGA_SCOPE(area_cell, m_mesh->m_area_cell);
  OMEGA_SCOPE(cells_on_edge, m_mesh->m_cells_on_edge);
  OMEGA_SCOPE(mesh_scaling_del2, m_mesh->m_mesh_scaling_del2);
  OMEGA_SCOPE(mesh_scaling_del4, m_mesh->m_mesh_scaling_del4);

  OMEGA_SCOPE(h_flux_edge, m_h_flux_edge);
  OMEGA_SCOPE(h_mean_edge, m_h_mean_edge);
  OMEGA_SCOPE(norm_tr_cell, m_norm_tr_cell);
  OMEGA_SCOPE(ntracers, m_ntracers);
  OMEGA_SCOPE(eddy_diff2, m_eddy_diff2);
  OMEGA_SCOPE(eddy_diff4, m_eddy_diff4);

  Real3d tmp_tr_del2_cell;
  if (eddy_diff4 > 0) {
    tmp_tr_del2_cell = Real3d("tmp_tr_del2_cell", ntracers, m_mesh->m_ncells,
                              m_mesh->m_nlayers);
    omega_parallel_for(
        "compute_tmp_tr_del2_cell",
        {ntracers, m_mesh->m_ncells, m_mesh->m_nlayers},
        KOKKOS_LAMBDA(Int l, Int icell, Int k) {
          Real tr_del2 = -0;
          for (Int j = 0; j < nedges_on_cell(icell); ++j) {
            Int jedge = edges_on_cell(icell, j);

            Int jcell0 = cells_on_edge(jedge, 0);
            Int jcell1 = cells_on_edge(jedge, 1);

            Real inv_dc_edge = 1._fp / dc_edge(jedge);
            Real grad_tr_edge =
                (norm_tr_cell(l, jcell1, k) - norm_tr_cell(l, jcell0, k)) *
                inv_dc_edge;

            tr_del2 += dv_edge(jedge) * edge_sign_on_cell(icell, j) *
                       h_mean_edge(jedge, k) * mesh_scaling_del2(jedge) *
                       grad_tr_edge;
          }
          Real inv_area_cell = 1._fp / area_cell(icell);
          tmp_tr_del2_cell(l, icell, k) = tr_del2 * inv_area_cell;
        });
  }

  omega_parallel_for(
      "compute_tr_tend", {ntracers, m_mesh->m_ncells, m_mesh->m_nlayers},
      KOKKOS_LAMBDA(Int l, Int icell, Int k) {
        Real tr_tend = -0;

        for (Int j = 0; j < nedges_on_cell(icell); ++j) {
          Int jedge = edges_on_cell(icell, j);

          Int jcell0 = cells_on_edge(jedge, 0);
          Int jcell1 = cells_on_edge(jedge, 1);

          Real norm_tr_edge =
              (norm_tr_cell(l, jcell0, k) + norm_tr_cell(l, jcell1, k)) *
              0.5_fp;

          // advection
          Real tr_flux =
              -h_flux_edge(jedge, k) * norm_tr_edge * vn_edge(jedge, k);

          Real inv_dc_edge = 1._fp / dc_edge(jedge);
          // diffusion
          if (eddy_diff2 > 0) {
            Real grad_tr_edge =
                (norm_tr_cell(l, jcell1, k) - norm_tr_cell(l, jcell0, k)) *
                inv_dc_edge;
            tr_flux += eddy_diff2 * h_mean_edge(jedge, k) * grad_tr_edge;
          }

          // hyperdiffusion
          if (eddy_diff4 > 0) {
            Real grad_tr_del2_edge = (tmp_tr_del2_cell(l, jcell1, k) -
                                      tmp_tr_del2_cell(l, jcell0, k)) *
                                     inv_dc_edge;
            tr_flux -=
                eddy_diff4 * grad_tr_del2_edge * mesh_scaling_del4(jedge);
          }

          tr_tend += dv_edge(jedge) * edge_sign_on_cell(icell, j) * tr_flux;
        }

        Real inv_area_cell = 1._fp / area_cell(icell);
        if (add_mode == AddMode::increment) {
          tr_tend_cell(l, icell, k) += tr_tend * inv_area_cell;
        }

        if (add_mode == AddMode::replace) {
          tr_tend_cell(l, icell, k) = tr_tend * inv_area_cell;
        }
      });
}

Real ShallowWaterModel::energy_integral(RealConst2d h_cell,
                                        RealConst2d vn_edge) const {

  OMEGA_SCOPE(nedges_on_cell, m_mesh->m_nedges_on_cell);
  OMEGA_SCOPE(edges_on_cell, m_mesh->m_edges_on_cell);
  OMEGA_SCOPE(dv_edge, m_mesh->m_dv_edge);
  OMEGA_SCOPE(dc_edge, m_mesh->m_dc_edge);
  OMEGA_SCOPE(area_cell, m_mesh->m_area_cell);
  OMEGA_SCOPE(max_level_cell, m_mesh->m_max_level_cell);
  OMEGA_SCOPE(grav, m_grav);

  Real total_energy;
  omega_parallel_reduce(
      "compute_column_energy", {m_mesh->m_ncells},
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

} // namespace omega
