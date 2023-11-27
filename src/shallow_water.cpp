#include <shallow_water.hpp>

namespace omega {

// Base

ShallowWaterModelBase::ShallowWaterModelBase(MPASMesh *mesh,
                                             const ShallowWaterParams &params)
    : m_mesh(mesh), m_grav(params.m_grav),
      m_disable_h_tendency(params.m_disable_h_tendency),
      m_disable_vn_tendency(params.m_disable_vn_tendency),
      m_ntracers(params.m_ntracers), m_f_vertex("f_vertex", mesh->m_nvertices),
      m_f_edge("f_edge", mesh->m_nedges) {
  yakl::memset(m_f_vertex, params.m_f0);
  yakl::memset(m_f_edge, params.m_f0);
}

void ShallowWaterModelBase::compute_auxiliary_variables(
    RealConst2d h_cell, RealConst2d vn_edge, RealConst3d tr_cell) const {}

void ShallowWaterModelBase::compute_tendency(const ShallowWaterState &tend,
                                             const ShallowWaterState &state,
                                             Real t, AddMode add_mode) const {
  yakl::timer_start("compute_tendency");

  yakl::timer_start("compute_auxiliary_variables");
  compute_auxiliary_variables(state.m_h_cell, state.m_vn_edge, state.m_tr_cell);
  yakl::timer_stop("compute_auxiliary_variables");

  yakl::timer_start("h_tendency");
  if (!m_disable_h_tendency) {
    compute_h_tendency(tend.m_h_cell, state.m_h_cell, state.m_vn_edge,
                       add_mode);
  }
  yakl::timer_stop("h_tendency");

  yakl::timer_start("vn_tendency");
  if (!m_disable_vn_tendency) {
    compute_vn_tendency(tend.m_vn_edge, state.m_h_cell, state.m_vn_edge,
                        add_mode);
  }
  yakl::timer_stop("vn_tendency");

  if (m_ntracers > 0) {
    yakl::timer_start("tr_tendency");
    compute_tr_tendency(tend.m_tr_cell, state.m_tr_cell, state.m_vn_edge,
                        add_mode);
    yakl::timer_stop("tr_tendency");
  }

  yakl::timer_start("additional_tendency");
  additional_tendency(tend.m_h_cell, tend.m_vn_edge, state.m_h_cell,
                      state.m_vn_edge, t);
  yakl::timer_stop("additional_tendency");

  yakl::timer_stop("compute_tendency");
}

Real ShallowWaterModelBase::mass_integral(RealConst2d h_cell) const {
  Real1d column_mass("column_mass", m_mesh->m_ncells);

  YAKL_SCOPE(area_cell, m_mesh->m_area_cell);
  YAKL_SCOPE(max_level_cell, m_mesh->m_max_level_cell);

  parallel_for(
      "compute_column_mass", m_mesh->m_ncells, YAKL_LAMBDA(Int icell) {
        column_mass(icell) = 0;
        for (Int k = 0; k < max_level_cell(icell); ++k) {
          column_mass(icell) += area_cell(icell) * h_cell(icell, k);
        }
      });
  return yakl::intrinsics::sum(column_mass);
}

Real ShallowWaterModelBase::circulation_integral(RealConst2d vn_edge) const {
  Real1d column_circulation("column_circulation", m_mesh->m_nvertices);

  YAKL_SCOPE(dc_edge, m_mesh->m_dc_edge);
  YAKL_SCOPE(edges_on_vertex, m_mesh->m_edges_on_vertex);
  YAKL_SCOPE(edge_sign_on_vertex, m_mesh->m_edge_sign_on_vertex);
  YAKL_SCOPE(area_triangle, m_mesh->m_area_triangle);
  YAKL_SCOPE(f_vertex, m_f_vertex);
  YAKL_SCOPE(max_level_vertex_bot, m_mesh->m_max_level_vertex_bot);

  parallel_for(
      "compute_column_circulation", m_mesh->m_nvertices,
      YAKL_LAMBDA(Int ivertex) {
        column_circulation(ivertex) = 0;
        for (Int k = 0; k < max_level_vertex_bot(ivertex); ++k) {
          Real cir_i = -0;
          for (Int j = 0; j < 3; ++j) {
            Int jedge = edges_on_vertex(ivertex, j);
            cir_i += dc_edge(jedge) * edge_sign_on_vertex(ivertex, j) *
                     vn_edge(jedge, k);
          }
          column_circulation(ivertex) +=
              cir_i + f_vertex(ivertex) * area_triangle(ivertex);
        }
      });
  return yakl::intrinsics::sum(column_circulation);
}

// State

ShallowWaterState::ShallowWaterState(MPASMesh *mesh, Int ntracers)
    : m_h_cell("h_cell", mesh->m_ncells, mesh->m_nlayers),
      m_vn_edge("vn_edge", mesh->m_nedges, mesh->m_nlayers),
      m_tr_cell("tr_cell", ntracers, mesh->m_ncells, mesh->m_nlayers) {}

ShallowWaterState::ShallowWaterState(const ShallowWaterModelBase &sw)
    : ShallowWaterState(sw.m_mesh, sw.m_ntracers) {}

// Nonlinear

ShallowWaterModel::ShallowWaterModel(MPASMesh *mesh,
                                     const ShallowWaterParams &params)
    : ShallowWaterModelBase(mesh, params), m_drag_coeff(params.m_drag_coeff),
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
      m_norm_f_vertex("norm_f_vertex", mesh->m_nvertices, mesh->m_nlayers) {}

void ShallowWaterModel::compute_auxiliary_variables(RealConst2d h_cell,
                                                    RealConst2d vn_edge,
                                                    RealConst3d tr_cell) const {
  compute_vertex_auxiliary_variables(h_cell, vn_edge, tr_cell);
  compute_cell_auxiliary_variables(h_cell, vn_edge, tr_cell);
  compute_edge_auxiliary_variables(h_cell, vn_edge, tr_cell);
}

void ShallowWaterModel::compute_cell_auxiliary_variables(
    RealConst2d h_cell, RealConst2d vn_edge, RealConst3d tr_cell) const {
  YAKL_SCOPE(nedges_on_cell, m_mesh->m_nedges_on_cell);
  YAKL_SCOPE(edges_on_cell, m_mesh->m_edges_on_cell);
  YAKL_SCOPE(edge_sign_on_cell, m_mesh->m_edge_sign_on_cell);
  YAKL_SCOPE(dv_edge, m_mesh->m_dv_edge);
  YAKL_SCOPE(dc_edge, m_mesh->m_dc_edge);
  YAKL_SCOPE(area_cell, m_mesh->m_area_cell);

  YAKL_SCOPE(ke_cell, m_ke_cell);
  YAKL_SCOPE(div_cell, m_div_cell);
  YAKL_SCOPE(norm_tr_cell, m_norm_tr_cell);
  YAKL_SCOPE(ntracers, m_ntracers);
  YAKL_SCOPE(nlayers, m_mesh->m_nlayers);

  parallel_outer(
      "compute_cell_auxiliarys", m_mesh->m_ncells,
      YAKL_LAMBDA(Int icell, InnerHandler handler) {
        parallel_inner(
            nlayers,
            [&](const Int k) {
              Real ke = -0;
              Real div = -0;
              for (Int j = 0; j < nedges_on_cell(icell); ++j) {
                Int jedge = edges_on_cell(icell, j);
                Real area_edge = dv_edge(jedge) * dc_edge(jedge);
                ke +=
                    area_edge * vn_edge(jedge, k) * vn_edge(jedge, k) * 0.25_fp;
                div += dv_edge(jedge) * edge_sign_on_cell(icell, j) *
                       vn_edge(jedge, k);
              }
              Real inv_area_cell = 1._fp / area_cell(icell);
              ke *= inv_area_cell;
              div *= inv_area_cell;

              //div_cell(icell, k) = div;
              __builtin_nontemporal_store(div, &div_cell(icell, k));
              //ke_cell(icell, k) = ke;
              __builtin_nontemporal_store(ke, &ke_cell(icell, k));

              Real inv_h = 1._fp / h_cell(icell, k);
              for (Int l = 0; l < ntracers; ++l) {
                //norm_tr_cell(l, icell, k) = tr_cell(l, icell, k) * inv_h;
                __builtin_nontemporal_store(tr_cell(l, icell, k) * inv_h, &norm_tr_cell(l, icell, k));
              }
            },
            handler);
      },
      LaunchConfig<nlayers_max>());
}

void ShallowWaterModel::compute_vertex_auxiliary_variables(
    RealConst2d h_cell, RealConst2d vn_edge, RealConst3d tr_cell) const {
  YAKL_SCOPE(dc_edge, m_mesh->m_dc_edge);
  YAKL_SCOPE(edges_on_vertex, m_mesh->m_edges_on_vertex);
  YAKL_SCOPE(edge_sign_on_vertex, m_mesh->m_edge_sign_on_vertex);
  YAKL_SCOPE(area_triangle, m_mesh->m_area_triangle);
  YAKL_SCOPE(kiteareas_on_vertex, m_mesh->m_kiteareas_on_vertex);
  YAKL_SCOPE(cells_on_vertex, m_mesh->m_cells_on_vertex);

  YAKL_SCOPE(rvort_vertex, m_rvort_vertex);
  YAKL_SCOPE(f_vertex, m_f_vertex);
  YAKL_SCOPE(norm_rvort_vertex, m_norm_rvort_vertex);
  YAKL_SCOPE(norm_f_vertex, m_norm_f_vertex);
  YAKL_SCOPE(nlayers, m_mesh->m_nlayers);

  parallel_outer(
      "compute_vertex_auxiliarys", m_mesh->m_nvertices,
      YAKL_LAMBDA(Int ivertex, InnerHandler handler) {
        parallel_inner(
            nlayers,
            [&](const Int k) {
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

              //rvort_vertex(ivertex, k) = rvort;
              __builtin_nontemporal_store(rvort, &rvort_vertex(ivertex, k));
              //norm_rvort_vertex(ivertex, k) = rvort * inv_h;
              __builtin_nontemporal_store(rvort * inv_h, &norm_rvort_vertex(ivertex, k));
              //norm_f_vertex(ivertex, k) = f_vertex(ivertex) * inv_h;
              __builtin_nontemporal_store(f_vertex(ivertex) * inv_h, &norm_f_vertex(ivertex, k));
            },
            handler);
      },
      LaunchConfig<nlayers_max>());
}

void ShallowWaterModel::compute_edge_auxiliary_variables(
    RealConst2d h_cell, RealConst2d vn_edge, RealConst3d tr_cell) const {

  YAKL_SCOPE(cells_on_edge, m_mesh->m_cells_on_edge);
  YAKL_SCOPE(vertices_on_edge, m_mesh->m_vertices_on_edge);

  YAKL_SCOPE(h_mean_edge, m_h_mean_edge);
  YAKL_SCOPE(h_flux_edge, m_h_flux_edge);
  YAKL_SCOPE(h_drag_edge, m_h_drag_edge);
  YAKL_SCOPE(norm_rvort_edge, m_norm_rvort_edge);
  YAKL_SCOPE(norm_f_edge, m_norm_f_edge);
  YAKL_SCOPE(norm_rvort_vertex, m_norm_rvort_vertex);
  YAKL_SCOPE(norm_f_vertex, m_norm_f_vertex);
  YAKL_SCOPE(nlayers, m_mesh->m_nlayers);

  parallel_outer(
      "compute_edge_auxiliarys", m_mesh->m_nedges,
      YAKL_LAMBDA(Int iedge, InnerHandler handler) {
        parallel_inner(
            nlayers,
            [&](const Int k) {
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

              //h_mean_edge(iedge, k) = h_mean;
              __builtin_nontemporal_store(h_mean, &h_mean_edge(iedge, k));
              //h_flux_edge(iedge, k) = h_mean;
              __builtin_nontemporal_store(h_mean, &h_flux_edge(iedge, k));
              //h_drag_edge(iedge, k) = h_mean;
              __builtin_nontemporal_store(h_mean, &h_drag_edge(iedge, k));

              //norm_rvort_edge(iedge, k) = norm_rvort;
              __builtin_nontemporal_store(norm_rvort, &norm_rvort_edge(iedge, k));
              //norm_f_edge(iedge, k) = norm_f;
              __builtin_nontemporal_store(norm_f, &norm_f_edge(iedge, k));
            },
            handler);
      },
      LaunchConfig<nlayers_max>());
}

void ShallowWaterModel::compute_h_tendency(Real2d h_tend_cell,
                                           RealConst2d h_cell,
                                           RealConst2d vn_edge,
                                           AddMode add_mode) const {
  YAKL_SCOPE(nedges_on_cell, m_mesh->m_nedges_on_cell);
  YAKL_SCOPE(edges_on_cell, m_mesh->m_edges_on_cell);
  YAKL_SCOPE(dv_edge, m_mesh->m_dv_edge);
  YAKL_SCOPE(edge_sign_on_cell, m_mesh->m_edge_sign_on_cell);
  YAKL_SCOPE(area_cell, m_mesh->m_area_cell);

  YAKL_SCOPE(h_flux_edge, m_h_flux_edge);
  YAKL_SCOPE(nlayers, m_mesh->m_nlayers);
  parallel_outer(
      "compute_htend", SimpleBounds<1>(m_mesh->m_ncells),
      YAKL_LAMBDA(Int icell, InnerHandler handler) {
        parallel_inner(
            nlayers,
            [&](const Int k) {
              Real accum = -0;

#pragma unroll 5
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
                //h_tend_cell(icell, k) = -accum * inv_area_cell;
                __builtin_nontemporal_store(-accum * inv_area_cell, &h_tend_cell(icell, k));
              }
            },
            handler);
      },
      LaunchConfig<nlayers_max>());
}

void ShallowWaterModel::compute_vn_tendency(Real2d vn_tend_edge,
                                            RealConst2d h_cell,
                                            RealConst2d vn_edge,
                                            AddMode add_mode) const {
  YAKL_SCOPE(max_level_edge_top, m_mesh->m_max_level_edge_top);
  YAKL_SCOPE(nedges_on_edge, m_mesh->m_nedges_on_edge);
  YAKL_SCOPE(edges_on_edge, m_mesh->m_edges_on_edge);
  YAKL_SCOPE(weights_on_edge, m_mesh->m_weights_on_edge);
  YAKL_SCOPE(dc_edge, m_mesh->m_dc_edge);
  YAKL_SCOPE(dv_edge, m_mesh->m_dv_edge);
  YAKL_SCOPE(cells_on_edge, m_mesh->m_cells_on_edge);
  YAKL_SCOPE(vertices_on_edge, m_mesh->m_vertices_on_edge);
  YAKL_SCOPE(edge_mask, m_mesh->m_edge_mask);
  YAKL_SCOPE(mesh_scaling_del2, m_mesh->m_mesh_scaling_del2);
  YAKL_SCOPE(mesh_scaling_del4, m_mesh->m_mesh_scaling_del4);

  YAKL_SCOPE(grav, m_grav);
  YAKL_SCOPE(norm_rvort_edge, m_norm_rvort_edge);
  YAKL_SCOPE(norm_f_edge, m_norm_f_edge);
  YAKL_SCOPE(h_flux_edge, m_h_flux_edge);
  // YAKL_SCOPE(h_drag_edge, m_h_drag_edge);
  YAKL_SCOPE(ke_cell, m_ke_cell);
  YAKL_SCOPE(div_cell, m_div_cell);
  YAKL_SCOPE(rvort_vertex, m_rvort_vertex);
  // YAKL_SCOPE(drag_coeff, m_drag_coeff);
  YAKL_SCOPE(visc_del2, m_visc_del2);
  YAKL_SCOPE(visc_del4, m_visc_del4);
  YAKL_SCOPE(nlayers, m_mesh->m_nlayers);

  Real2d del2u_edge, del2rvort_vertex, del2div_cell;
  if (visc_del4 > 0) {
    del2u_edge = Real2d("del2u_edge", m_mesh->m_nedges, m_mesh->m_nlayers);
    del2rvort_vertex =
        Real2d("del2rvort_vertex", m_mesh->m_nvertices, m_mesh->m_nlayers);
    del2div_cell = Real2d("del2div_cell", m_mesh->m_ncells, m_mesh->m_nlayers);

    YAKL_SCOPE(nedges_on_cell, m_mesh->m_nedges_on_cell);
    YAKL_SCOPE(edges_on_cell, m_mesh->m_edges_on_cell);
    YAKL_SCOPE(edge_sign_on_cell, m_mesh->m_edge_sign_on_cell);
    YAKL_SCOPE(area_cell, m_mesh->m_area_cell);
    YAKL_SCOPE(edges_on_vertex, m_mesh->m_edges_on_vertex);
    YAKL_SCOPE(area_triangle, m_mesh->m_area_triangle);
    YAKL_SCOPE(edge_sign_on_vertex, m_mesh->m_edge_sign_on_vertex);

    parallel_outer(
        "compute_del2u_edge", m_mesh->m_nedges,
        YAKL_LAMBDA(Int iedge, InnerHandler handler) {
          parallel_inner(
              nlayers,
              [&](const Int k) {
                Int icell0 = cells_on_edge(iedge, 0);
                Int icell1 = cells_on_edge(iedge, 1);

                Int ivertex0 = vertices_on_edge(iedge, 0);
                Int ivertex1 = vertices_on_edge(iedge, 1);

                Real dc_edge_inv = 1._fp / dc_edge(iedge);
                Real dv_edge_inv =
                    1._fp /
                    std::max(dv_edge(iedge), 0.25_fp * dc_edge(iedge)); // huh

                Real del2u =
                    ((div_cell(icell1, k) - div_cell(icell0, k)) * dc_edge_inv -
                     (rvort_vertex(ivertex1, k) - rvort_vertex(ivertex0, k)) *
                         dv_edge_inv);

                //del2u_edge(iedge, k) = del2u;
                __builtin_nontemporal_store(del2u, &del2u_edge(iedge, k));
              },
              handler);
        },
        LaunchConfig<nlayers_max>());

    parallel_outer(
        "compute_del2div_cell", m_mesh->m_ncells,
        YAKL_LAMBDA(Int icell, InnerHandler handler) {
          parallel_inner(
              nlayers,
              [&](const Int k) {
                Real del2div = -0;
#pragma unroll 5
                for (Int j = 0; j < nedges_on_cell(icell); ++j) {
                  Int jedge = edges_on_cell(icell, j);
                  del2div += dv_edge(jedge) * edge_sign_on_cell(icell, j) *
                             del2u_edge(jedge, k);
                }
                Real inv_area_cell = 1._fp / area_cell(icell);
                del2div *= inv_area_cell;
                //del2div_cell(icell, k) = del2div;
                __builtin_nontemporal_store(del2div, &del2div_cell(icell, k));
              },
              handler);
        },
        LaunchConfig<nlayers_max>());

    parallel_outer(
        "compute_del2rvort_vertex", m_mesh->m_nvertices,
        YAKL_LAMBDA(Int ivertex, InnerHandler handler) {
          parallel_inner(
              nlayers,
              [&](const Int k) {
                Real del2rvort = -0;
                for (Int j = 0; j < 3; ++j) {
                  Int jedge = edges_on_vertex(ivertex, j);
                  del2rvort += dc_edge(jedge) *
                               edge_sign_on_vertex(ivertex, j) *
                               vn_edge(jedge, k);
                }
                Real inv_area_triangle = 1._fp / area_triangle(ivertex);
                del2rvort *= inv_area_triangle;

                //del2rvort_vertex(ivertex, k) = del2rvort;
                __builtin_nontemporal_store(del2rvort, &del2rvort_vertex(ivertex, k));
              },
              handler);
        },
        LaunchConfig<nlayers_max>());
  }

  parallel_outer(
      "compute_vtend", m_mesh->m_nedges,
      YAKL_LAMBDA(Int iedge, InnerHandler handler) {
        parallel_inner(
            nlayers,
            [&](const Int k) {
              Real vn_tend = -0;

              Real qt = -0;
#pragma unroll 5
              for (Int j = 0; j < nedges_on_edge(iedge); ++j) {
                Int jedge = edges_on_edge(iedge, j);

                Real norm_vort =
                    (norm_rvort_edge(iedge, k) + norm_f_edge(iedge, k) +
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
              //                       ? -drag_coeff * std::sqrt(ke_cell0 +
              //                       ke_cell1)
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
                     (del2rvort_vertex(ivertex1, k) -
                      del2rvort_vertex(ivertex0, k)) *
                         inv_dv_edge);
                vn_tend -= visc4 * edge_mask(iedge, k);
              }

              if (add_mode == AddMode::increment) {
                vn_tend_edge(iedge, k) += vn_tend;
              }
              if (add_mode == AddMode::replace) {
                //vn_tend_edge(iedge, k) = vn_tend;
                __builtin_nontemporal_store(vn_tend, &vn_tend_edge(iedge, k));
              }
            },
            handler);
      },
      LaunchConfig<nlayers_max>());
}

void ShallowWaterModel::compute_tr_tendency(Real3d tr_tend_cell,
                                            RealConst3d tr_cell,
                                            RealConst2d vn_edge,
                                            AddMode add_mode) const {
  YAKL_SCOPE(nedges_on_cell, m_mesh->m_nedges_on_cell);
  YAKL_SCOPE(edges_on_cell, m_mesh->m_edges_on_cell);
  YAKL_SCOPE(dv_edge, m_mesh->m_dv_edge);
  YAKL_SCOPE(dc_edge, m_mesh->m_dc_edge);
  YAKL_SCOPE(edge_sign_on_cell, m_mesh->m_edge_sign_on_cell);
  YAKL_SCOPE(area_cell, m_mesh->m_area_cell);
  YAKL_SCOPE(cells_on_edge, m_mesh->m_cells_on_edge);
  YAKL_SCOPE(mesh_scaling_del2, m_mesh->m_mesh_scaling_del2);
  YAKL_SCOPE(mesh_scaling_del4, m_mesh->m_mesh_scaling_del4);

  YAKL_SCOPE(h_flux_edge, m_h_flux_edge);
  YAKL_SCOPE(h_mean_edge, m_h_mean_edge);
  YAKL_SCOPE(norm_tr_cell, m_norm_tr_cell);
  YAKL_SCOPE(ntracers, m_ntracers);
  YAKL_SCOPE(eddy_diff2, m_eddy_diff2);
  YAKL_SCOPE(eddy_diff4, m_eddy_diff4);
  YAKL_SCOPE(nlayers, m_mesh->m_nlayers);

  Real3d tmp_tr_del2_cell;
  if (eddy_diff4 > 0) {
    tmp_tr_del2_cell = Real3d("tmp_tr_del2_cell", ntracers, m_mesh->m_ncells,
                              m_mesh->m_nlayers);
    parallel_outer(
        "compute_tmp_tr_del2_cell", SimpleBounds<2>(ntracers, m_mesh->m_ncells),
        YAKL_LAMBDA(Int l, Int icell, InnerHandler handler) {
          parallel_inner(
              nlayers,
              [&](const Int k) {
                Real tr_del2 = -0;
#pragma unroll 5
                for (Int j = 0; j < nedges_on_cell(icell); ++j) {
                  Int jedge = edges_on_cell(icell, j);

                  Int jcell0 = cells_on_edge(jedge, 0);
                  Int jcell1 = cells_on_edge(jedge, 1);

                  Real inv_dc_edge = 1._fp / dc_edge(jedge);
                  Real grad_tr_edge = (norm_tr_cell(l, jcell1, k) -
                                       norm_tr_cell(l, jcell0, k)) *
                                      inv_dc_edge;

                  tr_del2 += dv_edge(jedge) * edge_sign_on_cell(icell, j) *
                             h_mean_edge(jedge, k) * mesh_scaling_del2(jedge) *
                             grad_tr_edge;
                }
                Real inv_area_cell = 1._fp / area_cell(icell);
                //tmp_tr_del2_cell(l, icell, k) = tr_del2 * inv_area_cell;
                __builtin_nontemporal_store(tr_del2 * inv_area_cell, &tmp_tr_del2_cell(l, icell, k));
              },
              handler);
        },
        LaunchConfig<nlayers_max>());
  }

  parallel_outer(
      "compute_tr_tend", SimpleBounds<2>(ntracers, m_mesh->m_ncells),
      YAKL_LAMBDA(Int l, Int icell, InnerHandler handler) {
        parallel_inner(
            nlayers,
            [&](const Int k) {
              Real tr_tend = -0;

#pragma unroll 5
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
                  Real grad_tr_edge = (norm_tr_cell(l, jcell1, k) -
                                       norm_tr_cell(l, jcell0, k)) *
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

                tr_tend +=
                    dv_edge(jedge) * edge_sign_on_cell(icell, j) * tr_flux;
              }

              Real inv_area_cell = 1._fp / area_cell(icell);
              if (add_mode == AddMode::increment) {
                tr_tend_cell(l, icell, k) += tr_tend * inv_area_cell;
              }

              if (add_mode == AddMode::replace) {
                //tr_tend_cell(l, icell, k) = tr_tend * inv_area_cell;
                __builtin_nontemporal_store(tr_tend * inv_area_cell, &tr_tend_cell(l, icell, k));
              }
            },
            handler);
      },
      LaunchConfig<nlayers_max>());
}

Real ShallowWaterModel::energy_integral(RealConst2d h_cell,
                                        RealConst2d vn_edge) const {
  Real1d column_energy("column_energy", m_mesh->m_ncells);

  YAKL_SCOPE(nedges_on_cell, m_mesh->m_nedges_on_cell);
  YAKL_SCOPE(edges_on_cell, m_mesh->m_edges_on_cell);
  YAKL_SCOPE(dv_edge, m_mesh->m_dv_edge);
  YAKL_SCOPE(dc_edge, m_mesh->m_dc_edge);
  YAKL_SCOPE(area_cell, m_mesh->m_area_cell);
  YAKL_SCOPE(max_level_cell, m_mesh->m_max_level_cell);
  YAKL_SCOPE(grav, m_grav);

  parallel_for(
      "compute_column_energy", m_mesh->m_ncells, YAKL_LAMBDA(Int icell) {
        column_energy(icell) = 0;
        for (Int k = 0; k < max_level_cell(icell); ++k) {
          Real K = 0;
          for (Int j = 0; j < nedges_on_cell(icell); ++j) {
            Int jedge = edges_on_cell(icell, j);
            Real area_edge = dv_edge(jedge) * dc_edge(jedge);
            K += area_edge * vn_edge(jedge, k) * vn_edge(jedge, k) / 4;
          }
          K /= area_cell(icell);
          column_energy(icell) += area_cell(icell) * (grav * h_cell(icell, k) *
                                                          h_cell(icell, k) / 2 +
                                                      h_cell(icell, k) * K);
        }
      });
  return yakl::intrinsics::sum(column_energy);
}

// Linear

LinearShallowWaterModel::LinearShallowWaterModel(
    MPASMesh *mesh, const LinearShallowWaterParams &params)
    : ShallowWaterModelBase(mesh, params), m_h0(params.m_h0) {}

void LinearShallowWaterModel::compute_h_tendency(Real2d h_tend_cell,
                                                 RealConst2d h_cell,
                                                 RealConst2d vn_edge,
                                                 AddMode add_mode) const {
  YAKL_SCOPE(nedges_on_cell, m_mesh->m_nedges_on_cell);
  YAKL_SCOPE(edges_on_cell, m_mesh->m_edges_on_cell);
  YAKL_SCOPE(dv_edge, m_mesh->m_dv_edge);
  YAKL_SCOPE(edge_sign_on_cell, m_mesh->m_edge_sign_on_cell);
  YAKL_SCOPE(area_cell, m_mesh->m_area_cell);
  YAKL_SCOPE(h0, m_h0);

  parallel_for(
      "compute_htend", SimpleBounds<2>(m_mesh->m_ncells, m_mesh->m_nlayers),
      YAKL_LAMBDA(Int icell, Int k) {
        Real accum = -0;
        for (Int j = 0; j < nedges_on_cell(icell); ++j) {
          Int jedge = edges_on_cell(icell, j);
          accum +=
              dv_edge(jedge) * edge_sign_on_cell(icell, j) * vn_edge(jedge, k);
        }
        if (add_mode == AddMode::increment) {
          h_tend_cell(icell, k) += -h0 * accum / area_cell(icell);
        }
        if (add_mode == AddMode::replace) {
          h_tend_cell(icell, k) = -h0 * accum / area_cell(icell);
        }
      });
}

void LinearShallowWaterModel::compute_vn_tendency(Real2d vn_tend_edge,
                                                  RealConst2d h_cell,
                                                  RealConst2d vn_edge,
                                                  AddMode add_mode) const {
  YAKL_SCOPE(nedges_on_edge, m_mesh->m_nedges_on_edge);
  YAKL_SCOPE(edges_on_edge, m_mesh->m_edges_on_edge);
  YAKL_SCOPE(weights_on_edge, m_mesh->m_weights_on_edge);
  YAKL_SCOPE(dc_edge, m_mesh->m_dc_edge);
  YAKL_SCOPE(cells_on_edge, m_mesh->m_cells_on_edge);
  YAKL_SCOPE(grav, m_grav);
  YAKL_SCOPE(f_edge, m_f_edge);

  parallel_for(
      "compute_vtend", SimpleBounds<2>(m_mesh->m_nedges, m_mesh->m_nlayers),
      YAKL_LAMBDA(Int iedge, Int k) {
        Real vt = -0;
        for (Int j = 0; j < nedges_on_edge(iedge); ++j) {
          Int jedge = edges_on_edge(iedge, j);
          vt += weights_on_edge(iedge, j) * vn_edge(jedge, k);
        }

        Int icell0 = cells_on_edge(iedge, 0);
        Int icell1 = cells_on_edge(iedge, 1);
        Real grad_h = (h_cell(icell1, k) - h_cell(icell0, k)) / dc_edge(iedge);

        if (add_mode == AddMode::increment) {
          vn_tend_edge(iedge, k) += f_edge(iedge) * vt - grav * grad_h;
        }
        if (add_mode == AddMode::replace) {
          vn_tend_edge(iedge, k) = f_edge(iedge) * vt - grav * grad_h;
        }
      });
}

Real LinearShallowWaterModel::energy_integral(RealConst2d h_cell,
                                              RealConst2d vn_edge) const {
  Real1d column_energy("column_energy", m_mesh->m_ncells);

  YAKL_SCOPE(nedges_on_cell, m_mesh->m_nedges_on_cell);
  YAKL_SCOPE(edges_on_cell, m_mesh->m_edges_on_cell);
  YAKL_SCOPE(dv_edge, m_mesh->m_dv_edge);
  YAKL_SCOPE(dc_edge, m_mesh->m_dc_edge);
  YAKL_SCOPE(area_cell, m_mesh->m_area_cell);
  YAKL_SCOPE(max_level_cell, m_mesh->m_max_level_cell);
  YAKL_SCOPE(grav, m_grav);
  YAKL_SCOPE(h0, m_h0);

  parallel_for(
      "compute_column_energy", m_mesh->m_ncells, YAKL_LAMBDA(Int icell) {
        column_energy(icell) = 0;
        for (Int k = 0; k < max_level_cell(icell); ++k) {
          Real K = 0;
          for (Int j = 0; j < nedges_on_cell(icell); ++j) {
            Int jedge = edges_on_cell(icell, j);
            Real area_edge = dv_edge(jedge) * dc_edge(jedge);
            K += area_edge * vn_edge(jedge, k) * vn_edge(jedge, k) / 4;
          }
          K /= area_cell(icell);
          column_energy(icell) +=
              area_cell(icell) *
              (grav * h_cell(icell, k) * h_cell(icell, k) / 2 + h0 * K);
        }
      });
  return yakl::intrinsics::sum(column_energy);
}

} // namespace omega
