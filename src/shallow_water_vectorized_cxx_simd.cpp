#include <shallow_water.hpp>

#include <experimental/simd>
namespace stdex = std::experimental;

namespace omega {
constexpr auto aligned = stdex::element_aligned;
using Pack = stdex::fixed_size_simd<Real, vector_length>;
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

  parallel_for(
      "compute_cell_auxiliarys",
      SimpleBounds<2>(m_mesh->m_ncells, m_mesh->m_nlayers / vector_length),
      YAKL_LAMBDA(Int icell, Int kv) {
        Int k = kv * vector_length;
        Pack ke_pack = 0;
        Pack div_pack = 0;

        for (Int j = 0; j < nedges_on_cell(icell); ++j) {
          Int jedge = edges_on_cell(icell, j);
          Real area_edge = dv_edge(jedge) * dc_edge(jedge);

          Pack vn_pack;
          vn_pack.copy_from(&vn_edge(jedge, k), aligned);
          ke_pack += area_edge * vn_pack * vn_pack * 0.25_fp;
          div_pack += dv_edge(jedge) * edge_sign_on_cell(icell, j) * vn_pack;
        }
        Real inv_area_cell = 1._fp / area_cell(icell);
        ke_pack *= inv_area_cell;
        div_pack *= inv_area_cell;

        Pack h_pack;
        div_pack.copy_to(&div_cell(icell, k), aligned);
        ke_pack.copy_to(&ke_cell(icell, k), aligned);
        h_pack.copy_from(&h_cell(icell, k), aligned);

        auto inv_h_pack = 1._fp / h_pack;
        for (Int l = 0; l < ntracers; ++l) {
          Pack norm_tr_pack;
          norm_tr_pack.copy_from(&tr_cell(l, icell, k), aligned);
          norm_tr_pack *= inv_h_pack;
          norm_tr_pack.copy_to(&norm_tr_cell(l, icell, k), aligned);
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

  YAKL_SCOPE(rvort_vertex, m_rvort_vertex);
  YAKL_SCOPE(f_vertex, m_f_vertex);
  YAKL_SCOPE(norm_rvort_vertex, m_norm_rvort_vertex);
  YAKL_SCOPE(norm_f_vertex, m_norm_f_vertex);

  parallel_for(
      "compute_vertex_auxiliarys",
      SimpleBounds<2>(m_mesh->m_nvertices, m_mesh->m_nlayers / vector_length),
      YAKL_LAMBDA(Int ivertex, Int kv) {
        Int k = kv * vector_length;
        Real inv_area_triangle = 1._fp / area_triangle(ivertex);
        Pack rvort_pack;
        rvort_pack = 0;
        for (Int j = 0; j < 3; ++j) {
          Int jedge = edges_on_vertex(ivertex, j);

          Pack vn_pack;
          vn_pack.copy_from(&vn_edge(jedge, k), aligned);

          rvort_pack +=
              dc_edge(jedge) * edge_sign_on_vertex(ivertex, j) * vn_pack;
        }
        rvort_pack *= inv_area_triangle;

        Pack h_vertex_pack;
        h_vertex_pack = 0;
        for (Int j = 0; j < 3; ++j) {
          Int jcell = cells_on_vertex(ivertex, j);

          Pack h_cell_pack;
          h_cell_pack.copy_from(&h_cell(jcell, k), aligned);

          h_vertex_pack += kiteareas_on_vertex(ivertex, j) * h_cell_pack;
        }
        h_vertex_pack *= inv_area_triangle;

        auto inv_h_vertex_pack = 1._fp / h_vertex_pack;
        auto norm_rvort_pack = inv_h_vertex_pack * rvort_pack;
        auto norm_f_pack = inv_h_vertex_pack * f_vertex(ivertex);

        rvort_pack.copy_to(&rvort_vertex(ivertex, k), aligned);
        norm_rvort_pack.copy_to(&norm_rvort_vertex(ivertex, k), aligned);
        norm_f_pack.copy_to(&norm_f_vertex(ivertex, k), aligned);
      });
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

  parallel_for(
      "compute_edge_auxiliarys",
      SimpleBounds<2>(m_mesh->m_nedges, m_mesh->m_nlayers / vector_length),
      YAKL_LAMBDA(Int iedge, Int kv) {
        Int k = kv * vector_length;
        Pack h_mean_pack;
        h_mean_pack = 0;
        for (Int j = 0; j < 2; ++j) {
          Int jcell = cells_on_edge(iedge, j);

          Pack h_pack;
          h_pack.copy_from(&h_cell(jcell, k), aligned);

          h_mean_pack += h_pack;
        }
        h_mean_pack *= 0.5_fp;

        Pack norm_f_edge_pack;
        Pack norm_rvort_edge_pack;
        norm_f_edge_pack = 0;
        norm_rvort_edge_pack = 0;
        for (Int j = 0; j < 2; ++j) {
          Int jvertex = vertices_on_edge(iedge, j);

          Pack norm_f_vertex_pack;
          Pack norm_rvort_vertex_pack;
          norm_rvort_vertex_pack.copy_from(&norm_rvort_vertex(jvertex, k),
                                           aligned);
          norm_f_vertex_pack.copy_from(&norm_f_vertex(jvertex, k), aligned);

          norm_rvort_edge_pack += norm_rvort_vertex_pack;
          norm_f_edge_pack += norm_f_vertex_pack;
        }
        norm_rvort_edge_pack *= 0.5_fp;
        norm_f_edge_pack *= 0.5_fp;

        h_mean_pack.copy_to(&h_mean_edge(iedge, k), aligned);
        h_mean_pack.copy_to(&h_flux_edge(iedge, k), aligned);
        h_mean_pack.copy_to(&h_drag_edge(iedge, k), aligned);
        norm_rvort_edge_pack.copy_to(&norm_rvort_edge(iedge, k), aligned);
        norm_f_edge_pack.copy_to(&norm_f_edge(iedge, k), aligned);
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

  YAKL_SCOPE(h_flux_edge, m_h_flux_edge);
  parallel_for(
      "compute_htend",
      SimpleBounds<2>(m_mesh->m_ncells, m_mesh->m_nlayers / vector_length),
      YAKL_LAMBDA(Int icell, Int kv) {
        Int k = kv * vector_length;
        Pack h_tend_cell_pack;
        h_tend_cell_pack = 0;
        for (Int j = 0; j < nedges_on_cell(icell); ++j) {
          Int jedge = edges_on_cell(icell, j);

          Pack h_flux_edge_pack, vn_edge_pack;
          h_flux_edge_pack.copy_from(&h_flux_edge(jedge, k), aligned);
          vn_edge_pack.copy_from(&vn_edge(jedge, k), aligned);

          h_tend_cell_pack += dv_edge(jedge) * edge_sign_on_cell(icell, j) *
                              h_flux_edge_pack * vn_edge_pack;
        }

        Real inv_area_cell = 1._fp / area_cell(icell);
        h_tend_cell_pack *= -inv_area_cell;

        if (add_mode == AddMode::increment) {
          Pack prev_h_tend_cell_pack;
          prev_h_tend_cell_pack.copy_from(&h_tend_cell(icell, k), aligned);
          prev_h_tend_cell_pack += h_tend_cell_pack;
          prev_h_tend_cell_pack.copy_to(&h_tend_cell(icell, k), aligned);
        }
        if (add_mode == AddMode::replace) {
          h_tend_cell_pack.copy_to(&h_tend_cell(icell, k), aligned);
        }
      });
}

void ShallowWaterModel::compute_vn_tendency(Real2d vn_tend_edge,
                                            RealConst2d h_cell,
                                            RealConst2d vn_edge,
                                            AddMode add_mode) const {
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
          Int k = kv * vector_length;
          Int icell0 = cells_on_edge(iedge, 0);
          Int icell1 = cells_on_edge(iedge, 1);

          Int ivertex0 = vertices_on_edge(iedge, 0);
          Int ivertex1 = vertices_on_edge(iedge, 1);

          Real dc_edge_inv = 1._fp / dc_edge(iedge);
          Real dv_edge_inv =
              1._fp / std::max(dv_edge(iedge), 0.25_fp * dc_edge(iedge)); // huh

          Pack div0_pack;
          Pack div1_pack;
          Pack rvort0_pack;
          Pack rvort1_pack;

          div0_pack.copy_from(&div_cell(icell0, k), aligned);
          div1_pack.copy_from(&div_cell(icell1, k), aligned);
          rvort0_pack.copy_from(&rvort_vertex(ivertex0, k), aligned);
          rvort1_pack.copy_from(&rvort_vertex(ivertex1, k), aligned);

          Pack del2u_pack;
          del2u_pack = (div1_pack - div0_pack) * dc_edge_inv -
                       (rvort1_pack - rvort0_pack) * dv_edge_inv;

          del2u_pack.copy_to(&del2u_edge(iedge, k), aligned);
        });

    parallel_for(
        "compute_del2div_cell",
        SimpleBounds<2>(m_mesh->m_ncells, m_mesh->m_nlayers / vector_length),
        YAKL_LAMBDA(Int icell, Int kv) {
          Int k = kv * vector_length;
          Pack del2div_pack;
          del2div_pack = 0;
          for (Int j = 0; j < nedges_on_cell(icell); ++j) {
            Int jedge = edges_on_cell(icell, j);

            Pack del2u_pack;
            del2u_pack.copy_from(&del2u_edge(jedge, k), aligned);

            del2div_pack +=
                dv_edge(jedge) * edge_sign_on_cell(icell, j) * del2u_pack;
          }
          Real inv_area_cell = 1._fp / area_cell(icell);
          del2div_pack *= inv_area_cell;
          del2div_pack.copy_to(&del2div_cell(icell, k), aligned);
        });

    parallel_for(
        "compute_del2rvort_vertex",
        SimpleBounds<2>(m_mesh->m_nvertices, m_mesh->m_nlayers / vector_length),
        YAKL_LAMBDA(Int ivertex, Int kv) {
          Int k = kv * vector_length;
          Pack del2rvort_pack;
          del2rvort_pack = 0;
          for (Int j = 0; j < 3; ++j) {
            Int jedge = edges_on_vertex(ivertex, j);

            Pack vn_pack;
            vn_pack.copy_from(&vn_edge(jedge, k), aligned);

            del2rvort_pack +=
                dc_edge(jedge) * edge_sign_on_vertex(ivertex, j) * vn_pack;
          }
          Real inv_area_triangle = 1._fp / area_triangle(ivertex);
          del2rvort_pack *= inv_area_triangle;

          del2rvort_pack.copy_to(&del2rvort_vertex(ivertex, k), aligned);
        });
  }

  parallel_for(
      "compute_vtend",
      SimpleBounds<2>(m_mesh->m_nedges, m_mesh->m_nlayers / vector_length),
      YAKL_LAMBDA(Int iedge, Int kv) {
        Int k = kv * vector_length;
        Pack vn_tend_pack;
        vn_tend_pack = 0;

        for (Int j = 0; j < nedges_on_edge(iedge); ++j) {
          Int jedge = edges_on_edge(iedge, j);

          Pack norm_rvort_iedge_pack;
          Pack norm_f_iedge_pack;
          Pack norm_rvort_jedge_pack;
          Pack norm_f_jedge_pack;
          Pack norm_rvort_pack;
          Pack h_flux_pack;
          Pack vn_pack;
          h_flux_pack.copy_from(&h_flux_edge(jedge, k), aligned);
          vn_pack.copy_from(&vn_edge(jedge, k), aligned);
          norm_rvort_iedge_pack.copy_from(&norm_rvort_edge(iedge, k), aligned);
          norm_f_iedge_pack.copy_from(&norm_f_edge(iedge, k), aligned);
          norm_rvort_jedge_pack.copy_from(&norm_rvort_edge(jedge, k), aligned);
          norm_f_jedge_pack.copy_from(&norm_f_edge(jedge, k), aligned);

          norm_rvort_pack = (norm_rvort_iedge_pack + norm_f_iedge_pack +
                             norm_rvort_jedge_pack + norm_f_jedge_pack) *
                            0.5_fp;

          vn_tend_pack += weights_on_edge(iedge, j) * h_flux_pack * vn_pack *
                          norm_rvort_pack;
        }

        Int icell0 = cells_on_edge(iedge, 0);
        Int icell1 = cells_on_edge(iedge, 1);

        Pack ke0_pack;
        Pack ke1_pack;
        Pack h0_pack;
        Pack h1_pack;
        h0_pack.copy_from(&h_cell(icell0, k), aligned);
        ke0_pack.copy_from(&ke_cell(icell0, k), aligned);
        h1_pack.copy_from(&h_cell(icell1, k), aligned);
        ke1_pack.copy_from(&ke_cell(icell1, k), aligned);

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

          Pack div0_pack;
          Pack div1_pack;
          Pack rvort0_pack;
          Pack rvort1_pack;
          Pack edge_mask_pack;
          div0_pack.copy_from(&div_cell(icell0, k), aligned);
          rvort0_pack.copy_from(&rvort_vertex(ivertex0, k), aligned);
          div1_pack.copy_from(&div_cell(icell1, k), aligned);
          rvort1_pack.copy_from(&rvort_vertex(ivertex1, k), aligned);
          edge_mask_pack.copy_from(&edge_mask(iedge, k), aligned);

          vn_tend_pack += visc_del2 * mesh_scaling_del2(iedge) *
                          edge_mask_pack *
                          ((div1_pack - div0_pack) * inv_dc_edge -
                           (rvort1_pack - rvort0_pack) * inv_dv_edge);
        }

        // hyperviscosity
        if (visc_del4 > 0) {
          Int ivertex0 = vertices_on_edge(iedge, 0);
          Int ivertex1 = vertices_on_edge(iedge, 1);

          Pack del2div0_pack;
          Pack del2div1_pack;
          Pack del2rvort0_pack;
          Pack del2rvort1_pack;
          Pack edge_mask_pack;

          del2div0_pack.copy_from(&del2div_cell(icell0, k), aligned);
          del2rvort0_pack.copy_from(&del2rvort_vertex(ivertex0, k), aligned);
          del2div1_pack.copy_from(&del2div_cell(icell1, k), aligned);
          del2rvort1_pack.copy_from(&del2rvort_vertex(ivertex1, k), aligned);
          edge_mask_pack.copy_from(&edge_mask(iedge, k), aligned);

          vn_tend_pack -= visc_del4 * mesh_scaling_del4(iedge) *
                          edge_mask_pack *
                          ((del2div1_pack - del2div0_pack) * inv_dc_edge -
                           (del2rvort1_pack - del2rvort0_pack) * inv_dv_edge);
        }

        if (add_mode == AddMode::increment) {
          Pack prev_vn_tend_pack;
          prev_vn_tend_pack.copy_from(&vn_tend_edge(iedge, k), aligned);
          prev_vn_tend_pack += vn_tend_pack;
          prev_vn_tend_pack.copy_to(&vn_tend_edge(iedge, k), aligned);
        }
        if (add_mode == AddMode::replace) {
          vn_tend_pack.copy_to(&vn_tend_edge(iedge, k), aligned);
        }
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
          Int k = kv * vector_length;
          Pack tr_del2_pack;
          tr_del2_pack = 0;
          for (Int j = 0; j < nedges_on_cell(icell); ++j) {
            Int jedge = edges_on_cell(icell, j);
            Int jcell0 = cells_on_edge(jedge, 0);
            Int jcell1 = cells_on_edge(jedge, 1);

            Real inv_dc_edge = 1._fp / dc_edge(jedge);

            Pack norm_tr0_pack;
            Pack norm_tr1_pack;
            Pack h_mean_pack;
            norm_tr0_pack.copy_from(&norm_tr_cell(l, jcell0, k), aligned);
            norm_tr1_pack.copy_from(&norm_tr_cell(l, jcell1, k), aligned);
            h_mean_pack.copy_from(&h_mean_edge(jedge, k), aligned);

            tr_del2_pack += dv_edge(jedge) * edge_sign_on_cell(icell, j) *
                            mesh_scaling_del2(jedge) * inv_dc_edge *
                            h_mean_pack * (norm_tr1_pack - norm_tr0_pack);
          }
          Real inv_area_cell = 1._fp / area_cell(icell);
          tr_del2_pack *= inv_area_cell;

          tr_del2_pack.copy_to(&tmp_tr_del2_cell(l, icell, k), aligned);
        });
  }

  parallel_for(
      "compute_tr_tend",
      SimpleBounds<3>(ntracers, m_mesh->m_ncells,
                      m_mesh->m_nlayers / vector_length),
      YAKL_LAMBDA(Int l, Int icell, Int kv) {
        Int k = kv * vector_length;
        Pack tr_tend_pack = 0;

        for (Int j = 0; j < nedges_on_cell(icell); ++j) {
          Int jedge = edges_on_cell(icell, j);

          Int jcell0 = cells_on_edge(jedge, 0);
          Int jcell1 = cells_on_edge(jedge, 1);

          Pack norm_tr0_pack;
          Pack norm_tr1_pack;
          Pack h_flux_pack;
          Pack vn_pack;
          norm_tr0_pack.copy_from(&norm_tr_cell(l, jcell0, k), aligned);
          norm_tr1_pack.copy_from(&norm_tr_cell(l, jcell1, k), aligned);
          h_flux_pack.copy_from(&h_flux_edge(jedge, k), aligned);
          vn_pack.copy_from(&vn_edge(jedge, k), aligned);

          Pack tr_flux_pack;
          // advection
          tr_flux_pack =
              -h_flux_pack * 0.5_fp * (norm_tr0_pack + norm_tr1_pack) * vn_pack;

          Real inv_dc_edge = 1._fp / dc_edge(jedge);
          // diffusion
          if (eddy_diff2 > 0) {
            Pack h_mean_pack;
            h_mean_pack.copy_from(&h_mean_edge(jedge, k), aligned);

            tr_flux_pack += eddy_diff2 * h_mean_pack * inv_dc_edge *
                            (norm_tr1_pack - norm_tr0_pack);
          }

          // hyperdiffusion
          if (eddy_diff4 > 0) {
            Pack tr0_del2_pack;
            Pack tr1_del2_pack;
            tr0_del2_pack.copy_from(&tmp_tr_del2_cell(l, jcell0, k), aligned);
            tr1_del2_pack.copy_from(&tmp_tr_del2_cell(l, jcell1, k), aligned);

            tr_flux_pack -= eddy_diff4 * inv_dc_edge *
                            mesh_scaling_del4(jedge) *
                            (tr1_del2_pack - tr0_del2_pack);
          }

          tr_tend_pack +=
              dv_edge(jedge) * edge_sign_on_cell(icell, j) * tr_flux_pack;
        }

        Real inv_area_cell = 1._fp / area_cell(icell);
        tr_tend_pack *= inv_area_cell;

        if (add_mode == AddMode::increment) {
          Pack prev_tr_tend_pack;
          prev_tr_tend_pack.copy_from(&tr_tend_cell(l, icell, k), aligned);
          prev_tr_tend_pack += tr_tend_pack;
          prev_tr_tend_pack.copy_to(&tr_tend_cell(l, icell, k), aligned);
        }
        if (add_mode == AddMode::replace) {
          tr_tend_pack.copy_to(&tr_tend_cell(l, icell, k), aligned);
        }
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
