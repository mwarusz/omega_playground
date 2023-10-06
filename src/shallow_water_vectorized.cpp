#include <shallow_water.hpp>

namespace omega {

using yakl::simd::iterate_over_pack;
using yakl::simd::Pack;
using yakl::simd::PackIterConfig;

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
      // m_rvort_cell("rvort_cell", mesh->m_ncells, mesh->m_nlayers),
      // m_norm_rvort_cell("norm_rvort_cell", mesh->m_ncells, mesh->m_nlayers),
      m_norm_tr_cell("norm_tr_cell", params.m_ntracers, mesh->m_ncells,
                     mesh->m_nlayers),
      m_h_flux_edge("h_flux_edge", mesh->m_nedges, mesh->m_nlayers),
      m_h_mean_edge("h_mean_edge", mesh->m_nedges, mesh->m_nlayers),
      m_h_drag_edge("h_drag_edge", mesh->m_nedges, mesh->m_nlayers),
      m_vt_edge("vt_edge", mesh->m_nedges, mesh->m_nlayers),
      m_norm_rvort_edge("norm_rvort_edge", mesh->m_nedges, mesh->m_nlayers),
      m_norm_f_edge("norm_f_edge", mesh->m_nedges, mesh->m_nlayers),
      // m_rcirc_vertex("rcirc_vertex", mesh->m_nvertices, mesh->m_nlayers),
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
  YAKL_SCOPE(max_level_cell, m_mesh->m_max_level_cell);
  YAKL_SCOPE(nedges_on_cell, m_mesh->m_nedges_on_cell);
  YAKL_SCOPE(edges_on_cell, m_mesh->m_edges_on_cell);
  YAKL_SCOPE(edge_sign_on_cell, m_mesh->m_edge_sign_on_cell);
  YAKL_SCOPE(dv_edge, m_mesh->m_dv_edge);
  YAKL_SCOPE(dc_edge, m_mesh->m_dc_edge);
  YAKL_SCOPE(vertices_on_cell, m_mesh->m_vertices_on_cell);
  YAKL_SCOPE(kite_index_on_cell, m_mesh->m_kite_index_on_cell);
  YAKL_SCOPE(kiteareas_on_vertex, m_mesh->m_kiteareas_on_vertex);
  YAKL_SCOPE(area_cell, m_mesh->m_area_cell);

  // YAKL_SCOPE(rvort_cell, m_rvort_cell);
  YAKL_SCOPE(ke_cell, m_ke_cell);
  YAKL_SCOPE(div_cell, m_div_cell);
  YAKL_SCOPE(norm_tr_cell, m_norm_tr_cell);
  YAKL_SCOPE(ntracers, m_ntracers);

  parallel_for(
      "compute_cell_auxiliarys",
      SimpleBounds<2>(m_mesh->m_ncells, m_mesh->m_nlayers / vector_length),
      YAKL_LAMBDA(Int icell, Int kv) {
        Pack<Real, vector_length> ke_pack;
        Pack<Real, vector_length> div_pack;
        ke_pack = 0;
        div_pack = 0;

        for (Int j = 0; j < nedges_on_cell(icell); ++j) {
          Int jedge = edges_on_cell(icell, j);
          Real area_edge = dv_edge(jedge) * dc_edge(jedge);

          Pack<Real, vector_length> vn_pack;
          iterate_over_pack(
              [&](Int klane) {
                Int k = kv * vector_length + klane;
                vn_pack(klane) = vn_edge(jedge, k);
              },
              PackIterConfig<vector_length, true>());

          ke_pack += area_edge * vn_pack * vn_pack * 0.25_fp;
          div_pack += dv_edge(jedge) * edge_sign_on_cell(icell, j) * vn_pack;
        }
        Real inv_area_cell = 1._fp / area_cell(icell);
        ke_pack *= inv_area_cell;
        div_pack *= inv_area_cell;

        Pack<Real, vector_length> h_pack;
        iterate_over_pack(
            [&](Int klane) {
              Int k = kv * vector_length + klane;
              div_cell(icell, k) = div_pack(klane);
              ke_cell(icell, k) = ke_pack(klane);
              h_pack(klane) = h_cell(icell, k);
            },
            PackIterConfig<vector_length, true>());

        auto inv_h_pack = 1._fp / h_pack;
        for (Int l = 0; l < ntracers; ++l) {
          Pack<Real, vector_length> norm_tr_pack;
          iterate_over_pack(
              [&](Int klane) {
                Int k = kv * vector_length + klane;
                norm_tr_pack(klane) = tr_cell(l, icell, k);
              },
              PackIterConfig<vector_length, true>());

          norm_tr_pack *= inv_h_pack;

          iterate_over_pack(
              [&](Int klane) {
                Int k = kv * vector_length + klane;
                norm_tr_cell(l, icell, k) = norm_tr_pack(klane);
              },
              PackIterConfig<vector_length, true>());
        }
      });
}

void ShallowWaterModel::compute_vertex_auxiliary_variables(
    RealConst2d h_cell, RealConst2d vn_edge, RealConst3d tr_cell) const {
  YAKL_SCOPE(dc_edge, m_mesh->m_dc_edge);
  YAKL_SCOPE(edges_on_vertex, m_mesh->m_edges_on_vertex);
  YAKL_SCOPE(edge_sign_on_vertex, m_mesh->m_edge_sign_on_vertex);
  YAKL_SCOPE(area_triangle, m_mesh->m_area_triangle);
  YAKL_SCOPE(kiteareas_on_vertex, m_mesh->m_kiteareas_on_vertex);
  YAKL_SCOPE(cells_on_vertex, m_mesh->m_cells_on_vertex);
  // YAKL_SCOPE(max_level_vertex_bot, m_mesh->m_max_level_vertex_bot);

  // YAKL_SCOPE(rcirc_vertex, m_rcirc_vertex);
  YAKL_SCOPE(rvort_vertex, m_rvort_vertex);
  YAKL_SCOPE(f_vertex, m_f_vertex);
  YAKL_SCOPE(norm_rvort_vertex, m_norm_rvort_vertex);
  YAKL_SCOPE(norm_f_vertex, m_norm_f_vertex);

  parallel_for(
      "compute_vertex_auxiliarys",
      SimpleBounds<2>(m_mesh->m_nvertices, m_mesh->m_nlayers / vector_length),
      YAKL_LAMBDA(Int ivertex, Int kv) {
        Real inv_area_triangle = 1._fp / area_triangle(ivertex);
        Pack<Real, vector_length> rvort_pack;
        rvort_pack = 0;
        for (Int j = 0; j < 3; ++j) {
          Int jedge = edges_on_vertex(ivertex, j);

          Pack<Real, vector_length> vn_pack;
          iterate_over_pack(
              [&](Int klane) {
                Int k = kv * vector_length + klane;
                vn_pack(klane) = vn_edge(jedge, k);
              },
              PackIterConfig<vector_length, true>());

          rvort_pack +=
              dc_edge(jedge) * edge_sign_on_vertex(ivertex, j) * vn_pack;
        }
        rvort_pack *= inv_area_triangle;

        Pack<Real, vector_length> h_vertex_pack;
        h_vertex_pack = 0;
        for (Int j = 0; j < 3; ++j) {
          Int jcell = cells_on_vertex(ivertex, j);

          Pack<Real, vector_length> h_cell_pack;
          iterate_over_pack(
              [&](Int klane) {
                Int k = kv * vector_length + klane;
                h_cell_pack(klane) = h_cell(jcell, k);
              },
              PackIterConfig<vector_length, true>());

          h_vertex_pack += kiteareas_on_vertex(ivertex, j) * h_cell_pack;
        }
        h_vertex_pack *= inv_area_triangle;

        auto inv_h_vertex_pack = 1._fp / h_vertex_pack;
        auto norm_rvort_pack = inv_h_vertex_pack * rvort_pack;
        auto norm_f_pack = inv_h_vertex_pack * f_vertex(ivertex);

        iterate_over_pack(
            [&](Int klane) {
              Int k = kv * vector_length + klane;
              rvort_vertex(ivertex, k) = rvort_pack(klane);
              norm_rvort_vertex(ivertex, k) = norm_rvort_pack(klane);
              norm_f_vertex(ivertex, k) = norm_f_pack(klane);
            },
            PackIterConfig<vector_length, true>());
      });
}

void ShallowWaterModel::compute_edge_auxiliary_variables(
    RealConst2d h_cell, RealConst2d vn_edge, RealConst3d tr_cell) const {

  YAKL_SCOPE(max_level_edge_top, m_mesh->m_max_level_edge_top);
  YAKL_SCOPE(cells_on_edge, m_mesh->m_cells_on_edge);
  YAKL_SCOPE(nedges_on_edge, m_mesh->m_nedges_on_edge);
  YAKL_SCOPE(edges_on_edge, m_mesh->m_edges_on_edge);
  YAKL_SCOPE(weights_on_edge, m_mesh->m_weights_on_edge);
  YAKL_SCOPE(max_level_edge_bot, m_mesh->m_max_level_edge_bot);
  YAKL_SCOPE(vertices_on_edge, m_mesh->m_vertices_on_edge);

  YAKL_SCOPE(h_mean_edge, m_h_mean_edge);
  YAKL_SCOPE(h_flux_edge, m_h_flux_edge);
  YAKL_SCOPE(h_drag_edge, m_h_drag_edge);
  YAKL_SCOPE(vt_edge, m_vt_edge);
  YAKL_SCOPE(norm_rvort_edge, m_norm_rvort_edge);
  YAKL_SCOPE(norm_f_edge, m_norm_f_edge);
  YAKL_SCOPE(norm_rvort_vertex, m_norm_rvort_vertex);
  YAKL_SCOPE(norm_f_vertex, m_norm_f_vertex);

  parallel_for(
      "compute_edge_auxiliarys",
      SimpleBounds<2>(m_mesh->m_nedges, m_mesh->m_nlayers / vector_length),
      YAKL_LAMBDA(Int iedge, Int kv) {
        Pack<Real, vector_length> h_mean_pack;
        h_mean_pack = 0;
        for (Int j = 0; j < 2; ++j) {
          Int jcell = cells_on_edge(iedge, j);

          Pack<Real, vector_length> h_pack;
          iterate_over_pack(
              [&](Int klane) {
                Int k = kv * vector_length + klane;
                h_pack(klane) = h_cell(jcell, k);
              },
              PackIterConfig<vector_length, true>());

          h_mean_pack += h_pack;
        }
        h_mean_pack *= 0.5_fp;

        Pack<Real, vector_length> vt_pack;
        for (Int j = 0; j < nedges_on_edge(iedge); ++j) {
          Int jedge = edges_on_edge(iedge, j);

          Pack<Real, vector_length> vn_pack;
          iterate_over_pack(
              [&](Int klane) {
                Int k = kv * vector_length + klane;
                vn_pack(klane) = vn_edge(jedge, k);
              },
              PackIterConfig<vector_length, true>());

          vt_pack += weights_on_edge(iedge, j) * vn_pack;
        }

        Pack<Real, vector_length> norm_f_edge_pack;
        Pack<Real, vector_length> norm_rvort_edge_pack;
        norm_f_edge_pack = 0;
        norm_rvort_edge_pack = 0;
        for (Int j = 0; j < 2; ++j) {
          Int jvertex = vertices_on_edge(iedge, j);

          Pack<Real, vector_length> norm_f_vertex_pack;
          Pack<Real, vector_length> norm_rvort_vertex_pack;
          iterate_over_pack(
              [&](Int klane) {
                Int k = kv * vector_length + klane;
                norm_f_vertex_pack(klane) = norm_rvort_vertex(jvertex, k);
                norm_rvort_vertex_pack(klane) = norm_f_vertex(jvertex, k);
              },
              PackIterConfig<vector_length, true>());

          norm_rvort_edge_pack += norm_rvort_vertex_pack;
          norm_f_edge_pack += norm_f_vertex_pack;
        }
        norm_rvort_edge_pack *= 0.5_fp;
        norm_f_edge_pack *= 0.5_fp;

        iterate_over_pack(
            [&](Int klane) {
              Int k = kv * vector_length + klane;
              h_mean_edge(iedge, k) = h_mean_pack(klane);
              h_flux_edge(iedge, k) = h_mean_pack(klane);
              h_drag_edge(iedge, k) = h_mean_pack(klane);

              vt_edge(iedge, k) = vt_pack(klane);

              norm_rvort_edge(iedge, k) = norm_rvort_edge_pack(klane);
              norm_f_edge(iedge, k) = norm_f_edge_pack(klane);
            },
            PackIterConfig<vector_length, true>());
      });
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
  YAKL_SCOPE(max_level_cell, m_mesh->m_max_level_cell);

  YAKL_SCOPE(h_flux_edge, m_h_flux_edge);
  parallel_for(
      "compute_htend",
      SimpleBounds<2>(m_mesh->m_ncells, m_mesh->m_nlayers / vector_length),
      YAKL_LAMBDA(Int icell, Int kv) {
        Pack<Real, vector_length> h_tend_cell_pack;
        h_tend_cell_pack = 0;
        for (Int j = 0; j < nedges_on_cell(icell); ++j) {
          Int jedge = edges_on_cell(icell, j);

          Pack<Real, vector_length> h_flux_edge_pack, vn_edge_pack;
          iterate_over_pack(
              [&](Int klane) {
                Int k = kv * vector_length + klane;
                h_flux_edge_pack(klane) = h_flux_edge(jedge, k);
                vn_edge_pack(klane) = vn_edge(jedge, k);
              },
              PackIterConfig<vector_length, true>());

          h_tend_cell_pack += dv_edge(jedge) * edge_sign_on_cell(icell, j) *
                              h_flux_edge_pack * vn_edge_pack;
        }

        Real inv_area_cell = 1._fp / area_cell(icell);
        h_tend_cell_pack *= -inv_area_cell;

        iterate_over_pack(
            [&](Int klane) {
              Int k = kv * vector_length + klane;
              if (add_mode == AddMode::increment) {
                h_tend_cell(icell, k) += h_tend_cell_pack(klane);
              }
              if (add_mode == AddMode::replace) {
                h_tend_cell(icell, k) = h_tend_cell_pack(klane);
              }
            },
            PackIterConfig<vector_length, true>());
      });
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
  YAKL_SCOPE(h_drag_edge, m_h_drag_edge);
  YAKL_SCOPE(ke_cell, m_ke_cell);
  YAKL_SCOPE(div_cell, m_div_cell);
  YAKL_SCOPE(rvort_vertex, m_rvort_vertex);
  YAKL_SCOPE(drag_coeff, m_drag_coeff);
  YAKL_SCOPE(visc_del2, m_visc_del2);
  YAKL_SCOPE(visc_del4, m_visc_del4);

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

    parallel_for(
        "compute_del2u_edge",
        SimpleBounds<2>(m_mesh->m_nedges, m_mesh->m_nlayers / vector_length),
        YAKL_LAMBDA(Int iedge, Int kv) {
          Int icell0 = cells_on_edge(iedge, 0);
          Int icell1 = cells_on_edge(iedge, 1);

          Int ivertex0 = vertices_on_edge(iedge, 0);
          Int ivertex1 = vertices_on_edge(iedge, 1);

          Real dc_edge_inv = 1._fp / dc_edge(iedge);
          Real dv_edge_inv =
              1._fp / std::max(dv_edge(iedge), 0.25_fp * dc_edge(iedge)); // huh

          Pack<Real, vector_length> div0_pack;
          Pack<Real, vector_length> div1_pack;
          Pack<Real, vector_length> rvort0_pack;
          Pack<Real, vector_length> rvort1_pack;

          iterate_over_pack(
              [&](Int klane) {
                Int k = kv * vector_length + klane;
                div0_pack(klane) = div_cell(icell0, k);
                div1_pack(klane) = div_cell(icell1, k);
                rvort0_pack(klane) = rvort_vertex(ivertex0, k);
                rvort1_pack(klane) = rvort_vertex(ivertex1, k);
              },
              PackIterConfig<vector_length, true>());

          Pack<Real, vector_length> del2u_pack;
          del2u_pack = (div1_pack - div0_pack) * dc_edge_inv -
                       (rvort1_pack - rvort0_pack) * dv_edge_inv;

          iterate_over_pack(
              [&](Int klane) {
                Int k = kv * vector_length + klane;
                del2u_edge(iedge, k) = del2u_pack(klane);
              },
              PackIterConfig<vector_length, true>());
        });

    parallel_for(
        "compute_del2div_cell",
        SimpleBounds<2>(m_mesh->m_ncells, m_mesh->m_nlayers / vector_length),
        YAKL_LAMBDA(Int icell, Int kv) {
          Pack<Real, vector_length> del2div_pack;
          del2div_pack = 0;
          for (Int j = 0; j < nedges_on_cell(icell); ++j) {
            Int jedge = edges_on_cell(icell, j);

            Pack<Real, vector_length> del2u_pack;
            iterate_over_pack(
                [&](Int klane) {
                  Int k = kv * vector_length + klane;
                  del2u_pack(klane) = del2u_edge(jedge, k);
                },
                PackIterConfig<vector_length, true>());

            del2div_pack +=
                dv_edge(jedge) * edge_sign_on_cell(icell, j) * del2u_pack;
          }
          Real inv_area_cell = 1._fp / area_cell(icell);
          del2div_pack *= inv_area_cell;
          iterate_over_pack(
              [&](Int klane) {
                Int k = kv * vector_length + klane;
                del2div_cell(icell, k) = del2div_pack(klane);
              },
              PackIterConfig<vector_length, true>());
        });

    parallel_for(
        "compute_del2rvort_vertex",
        SimpleBounds<2>(m_mesh->m_nvertices, m_mesh->m_nlayers / vector_length),
        YAKL_LAMBDA(Int ivertex, Int kv) {
          Pack<Real, vector_length> del2rvort_pack;
          del2rvort_pack = 0;
          for (Int j = 0; j < 3; ++j) {
            Int jedge = edges_on_vertex(ivertex, j);

            Pack<Real, vector_length> vn_pack;
            iterate_over_pack(
                [&](Int klane) {
                  Int k = kv * vector_length + klane;
                  vn_pack(klane) = vn_edge(jedge, k);
                },
                PackIterConfig<vector_length, true>());

            del2rvort_pack +=
                dc_edge(jedge) * edge_sign_on_vertex(ivertex, j) * vn_pack;
          }
          Real inv_area_triangle = 1._fp / area_triangle(ivertex);
          del2rvort_pack *= inv_area_triangle;

          iterate_over_pack(
              [&](Int klane) {
                Int k = kv * vector_length + klane;
                del2rvort_vertex(ivertex, k) = del2rvort_pack(klane);
              },
              PackIterConfig<vector_length, true>());
        });
  }

  parallel_for(
      "compute_vtend",
      SimpleBounds<2>(m_mesh->m_nedges, m_mesh->m_nlayers / vector_length),
      YAKL_LAMBDA(Int iedge, Int kv) {
        Pack<Real, vector_length> vn_tend_pack;
        vn_tend_pack = 0;

        for (Int j = 0; j < nedges_on_edge(iedge); ++j) {
          Int jedge = edges_on_edge(iedge, j);

          Pack<Real, vector_length> norm_rvort_iedge_pack;
          Pack<Real, vector_length> norm_f_iedge_pack;
          Pack<Real, vector_length> norm_rvort_jedge_pack;
          Pack<Real, vector_length> norm_f_jedge_pack;
          Pack<Real, vector_length> norm_rvort_pack;
          Pack<Real, vector_length> h_flux_pack;
          Pack<Real, vector_length> vn_pack;
          iterate_over_pack(
              [&](Int klane) {
                Int k = kv * vector_length + klane;
                h_flux_pack(klane) = h_flux_edge(jedge, k);
                vn_pack(klane) = vn_edge(jedge, k);
                norm_rvort_iedge_pack(klane) = norm_rvort_edge(iedge, k);
                norm_f_iedge_pack(klane) = norm_f_edge(iedge, k);
                norm_rvort_jedge_pack(klane) = norm_rvort_edge(jedge, k);
                norm_f_jedge_pack(klane) = norm_f_edge(jedge, k);
              },
              PackIterConfig<vector_length, true>());

          norm_rvort_pack = (norm_rvort_iedge_pack + norm_f_iedge_pack +
                             norm_rvort_jedge_pack + norm_f_jedge_pack) *
                            0.5_fp;

          vn_tend_pack += weights_on_edge(iedge, j) * h_flux_pack * vn_pack *
                          norm_rvort_pack;
        }

        Int icell0 = cells_on_edge(iedge, 0);
        Int icell1 = cells_on_edge(iedge, 1);

        Pack<Real, vector_length> ke0_pack;
        Pack<Real, vector_length> ke1_pack;
        Pack<Real, vector_length> h0_pack;
        Pack<Real, vector_length> h1_pack;
        iterate_over_pack(
            [&](Int klane) {
              Int k = kv * vector_length + klane;
              h0_pack(klane) = h_cell(icell0, k);
              ke0_pack(klane) = ke_cell(icell0, k);
              h1_pack(klane) = h_cell(icell1, k);
              ke1_pack(klane) = ke_cell(icell1, k);
            },
            PackIterConfig<vector_length, true>());

        Real inv_dc_edge = 1._fp / dc_edge(iedge);
        vn_tend_pack -=
            (ke1_pack - ke0_pack + grav * (h1_pack - h0_pack)) * inv_dc_edge;

        // TODO: vectorize this
        // Real inv_h_drag_edge = 1._fp / h_drag_edge(iedge, k);
        // Real drag_force = (k == (max_level_edge_top(iedge) - 1))
        //                      ? -drag_coeff * std::sqrt(ke_cell0 + ke_cell1) *
        //                            vn_edge(iedge, k) * inv_h_drag_edge
        //                      : 0;

        Real inv_dv_edge = 1._fp / dv_edge(iedge);
        // viscosity
        if (visc_del2 > 0) {
          Int ivertex0 = vertices_on_edge(iedge, 0);
          Int ivertex1 = vertices_on_edge(iedge, 1);

          Pack<Real, vector_length> div0_pack;
          Pack<Real, vector_length> div1_pack;
          Pack<Real, vector_length> rvort0_pack;
          Pack<Real, vector_length> rvort1_pack;
          Pack<Real, vector_length> edge_mask_pack;
          iterate_over_pack(
              [&](Int klane) {
                Int k = kv * vector_length + klane;
                div0_pack(klane) = div_cell(icell0, k);
                rvort0_pack(klane) = rvort_vertex(ivertex0, k);
                div1_pack(klane) = div_cell(icell1, k);
                rvort1_pack(klane) = rvort_vertex(ivertex1, k);
                edge_mask_pack(klane) = edge_mask(iedge, k);
              },
              PackIterConfig<vector_length, true>());

          vn_tend_pack += visc_del2 * mesh_scaling_del2(iedge) *
                          edge_mask_pack *
                          ((div1_pack - div0_pack) * inv_dc_edge -
                           (rvort1_pack - rvort0_pack) * inv_dv_edge);
        }

        // hyperviscosity
        if (visc_del4 > 0) {
          Int ivertex0 = vertices_on_edge(iedge, 0);
          Int ivertex1 = vertices_on_edge(iedge, 1);

          Pack<Real, vector_length> del2div0_pack;
          Pack<Real, vector_length> del2div1_pack;
          Pack<Real, vector_length> del2rvort0_pack;
          Pack<Real, vector_length> del2rvort1_pack;
          Pack<Real, vector_length> edge_mask_pack;

          iterate_over_pack(
              [&](Int klane) {
                Int k = kv * vector_length + klane;
                del2div0_pack(klane) = del2div_cell(icell0, k);
                del2rvort0_pack(klane) = del2rvort_vertex(ivertex0, k);
                del2div1_pack(klane) = del2div_cell(icell1, k);
                del2rvort1_pack(klane) = del2rvort_vertex(ivertex1, k);
                edge_mask_pack(klane) = edge_mask(iedge, k);
              },
              PackIterConfig<vector_length, true>());

          vn_tend_pack -= visc_del4 * mesh_scaling_del4(iedge) *
                          edge_mask_pack *
                          ((del2div1_pack - del2div0_pack) * inv_dc_edge -
                           (del2rvort1_pack - del2rvort0_pack) * inv_dv_edge);
        }

        iterate_over_pack(
            [&](Int klane) {
              Int k = kv * vector_length + klane;
              if (add_mode == AddMode::increment) {
                vn_tend_edge(iedge, k) += vn_tend_pack(klane);
              }
              if (add_mode == AddMode::replace) {
                vn_tend_edge(iedge, k) = vn_tend_pack(klane);
              }
            },
            PackIterConfig<vector_length, true>());
      });
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

  Real3d tmp_tr_del2_cell;
  if (eddy_diff4 > 0) {
    tmp_tr_del2_cell = Real3d("tmp_tr_del2_cell", ntracers, m_mesh->m_ncells,
                              m_mesh->m_nlayers);
    parallel_for(
        "compute_tmp_tr_del2_cell",
        SimpleBounds<3>(ntracers, m_mesh->m_ncells,
                        m_mesh->m_nlayers / vector_length),
        YAKL_LAMBDA(Int l, Int icell, Int kv) {
          Pack<Real, vector_length> tr_del2_pack;
          tr_del2_pack = 0;
          for (Int j = 0; j < nedges_on_cell(icell); ++j) {
            Int jedge = edges_on_cell(icell, j);
            Int jcell0 = cells_on_edge(jedge, 0);
            Int jcell1 = cells_on_edge(jedge, 1);

            Real inv_dc_edge = 1._fp / dc_edge(jedge);

            Pack<Real, vector_length> norm_tr0_pack;
            Pack<Real, vector_length> norm_tr1_pack;
            Pack<Real, vector_length> h_mean_pack;
            iterate_over_pack(
                [&](Int klane) {
                  Int k = kv * vector_length + klane;
                  norm_tr0_pack(klane) = norm_tr_cell(l, jcell0, k);
                  norm_tr1_pack(klane) = norm_tr_cell(l, jcell1, k);
                  h_mean_pack(klane) = h_mean_edge(jedge, k);
                },
                PackIterConfig<vector_length, true>());

            tr_del2_pack += dv_edge(jedge) * edge_sign_on_cell(icell, j) *
                            mesh_scaling_del2(jedge) * inv_dc_edge *
                            h_mean_pack * (norm_tr1_pack - norm_tr0_pack);
          }
          Real inv_area_cell = 1._fp / area_cell(icell);
          tr_del2_pack *= inv_area_cell;

          iterate_over_pack(
              [&](Int klane) {
                Int k = kv * vector_length + klane;
                tmp_tr_del2_cell(l, icell, k) = tr_del2_pack(klane);
              },
              PackIterConfig<vector_length, true>());
        });
  }

  parallel_for(
      "compute_tr_tend",
      SimpleBounds<3>(ntracers, m_mesh->m_ncells,
                      m_mesh->m_nlayers / vector_length),
      YAKL_LAMBDA(Int l, Int icell, Int kv) {
        Pack<Real, vector_length> tr_tend_pack;
        tr_tend_pack = 0;

        for (Int j = 0; j < nedges_on_cell(icell); ++j) {
          Int jedge = edges_on_cell(icell, j);

          Int jcell0 = cells_on_edge(jedge, 0);
          Int jcell1 = cells_on_edge(jedge, 1);

          Pack<Real, vector_length> norm_tr0_pack;
          Pack<Real, vector_length> norm_tr1_pack;
          Pack<Real, vector_length> h_flux_pack;
          Pack<Real, vector_length> vn_pack;
          iterate_over_pack(
              [&](Int klane) {
                Int k = kv * vector_length + klane;
                norm_tr0_pack(klane) = norm_tr_cell(l, jcell0, k);
                norm_tr1_pack(klane) = norm_tr_cell(l, jcell1, k);
                h_flux_pack(klane) = h_flux_edge(jedge, k);
                vn_pack(klane) = vn_edge(jedge, k);
              },
              PackIterConfig<vector_length, true>());

          Pack<Real, vector_length> tr_flux_pack;
          // advection
          tr_flux_pack =
              -h_flux_pack * 0.5_fp * (norm_tr0_pack + norm_tr1_pack) * vn_pack;

          Real inv_dc_edge = 1._fp / dc_edge(jedge);
          // diffusion
          if (eddy_diff2 > 0) {
            Pack<Real, vector_length> h_mean_pack;
            iterate_over_pack(
                [&](Int klane) {
                  Int k = kv * vector_length + klane;
                  h_mean_pack(klane) = h_mean_edge(jedge, k);
                },
                PackIterConfig<vector_length, true>());
            tr_flux_pack += eddy_diff2 * h_mean_pack * inv_dc_edge *
                            (norm_tr1_pack - norm_tr0_pack);
          }

          // hyperdiffusion
          if (eddy_diff4 > 0) {
            Pack<Real, vector_length> tr0_del2_pack;
            Pack<Real, vector_length> tr1_del2_pack;
            iterate_over_pack(
                [&](Int klane) {
                  Int k = kv * vector_length + klane;
                  tr0_del2_pack(klane) = tmp_tr_del2_cell(l, jcell0, k);
                  tr1_del2_pack(klane) = tmp_tr_del2_cell(l, jcell1, k);
                },
                PackIterConfig<vector_length, true>());
            tr_flux_pack -=
                eddy_diff4 * inv_dc_edge * (tr1_del2_pack - tr0_del2_pack);
          }

          tr_tend_pack += dv_edge(jedge) * edge_sign_on_cell(icell, j) *
                          tr_flux_pack * mesh_scaling_del4(jedge);
        }

        Real inv_area_cell = 1._fp / area_cell(icell);
        tr_tend_pack *= inv_area_cell;

        iterate_over_pack(
            [&](Int klane) {
              Int k = kv * vector_length + klane;
              if (add_mode == AddMode::increment) {
                tr_tend_cell(l, icell, k) += tr_tend_pack(klane);
              }
              if (add_mode == AddMode::replace) {
                tr_tend_cell(l, icell, k) = tr_tend_pack(klane);
              }
            },
            PackIterConfig<vector_length, true>());
      });
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
  YAKL_SCOPE(max_level_cell, m_mesh->m_max_level_cell);
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
  YAKL_SCOPE(dv_edge, m_mesh->m_dv_edge);
  YAKL_SCOPE(dc_edge, m_mesh->m_dc_edge);
  YAKL_SCOPE(cells_on_edge, m_mesh->m_cells_on_edge);
  YAKL_SCOPE(max_level_edge_top, m_mesh->m_max_level_edge_top);
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
