#include <shallow_water.hpp>

namespace omega {

// Base

ShallowWaterModelBase::ShallowWaterModelBase(PlanarHexagonalMesh &mesh,
                                             const ShallowWaterParams &params)
    : m_mesh(&mesh), m_grav(params.m_grav),
      m_disable_h_tendency(params.m_disable_h_tendency),
      m_disable_vn_tendency(params.m_disable_vn_tendency),
      m_ntracers(params.m_ntracers), m_f_vertex("f_vertex", mesh.m_nvertices),
      m_f_edge("f_edge", mesh.m_nedges) {
  yakl::memset(m_f_vertex, params.m_f0);
  yakl::memset(m_f_edge, params.m_f0);
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

ShallowWaterState::ShallowWaterState(const PlanarHexagonalMesh &mesh,
                                     Int ntracers)
    : m_h_cell("h_cell", mesh.m_ncells, mesh.m_nlayers),
      m_vn_edge("vn_edge", mesh.m_nedges, mesh.m_nlayers),
      m_tr_cell("tr_cell", ntracers, mesh.m_nedges, mesh.m_nlayers) {}

ShallowWaterState::ShallowWaterState(const ShallowWaterModelBase &sw)
    : ShallowWaterState(*sw.m_mesh, sw.m_ntracers) {}

// Nonlinear

ShallowWaterModel::ShallowWaterModel(PlanarHexagonalMesh &mesh,
                                     const ShallowWaterParams &params)
    : ShallowWaterModelBase(mesh, params), m_drag_coeff(params.m_drag_coeff),
      m_visc_del2(params.m_visc_del2),
      m_h_flux_edge("h_flux_edge", mesh.m_nedges, mesh.m_nlayers),
      m_h_mean_edge("h_mean_edge", mesh.m_nedges, mesh.m_nlayers),
      m_h_drag_edge("h_drag_edge", mesh.m_nedges, mesh.m_nlayers),
      m_rcirc_vertex("rcirc_vertex", mesh.m_nvertices, mesh.m_nlayers),
      m_rvort_vertex("rvort_vertex", mesh.m_nvertices, mesh.m_nlayers),
      m_rvort_cell("rvort_cell", mesh.m_ncells, mesh.m_nlayers),
      m_norm_tr_cell("norm_tr_cell", params.m_ntracers, mesh.m_ncells,
                     mesh.m_nlayers),
      m_ke_cell("ke_cell", mesh.m_ncells, mesh.m_nlayers),
      m_div_cell("div_cell", mesh.m_ncells, mesh.m_nlayers),
      m_vt_edge("vt_edge", mesh.m_nedges, mesh.m_nlayers),
      m_norm_rvort_vertex("norm_rvort_vertex", mesh.m_nvertices,
                          mesh.m_nlayers),
      m_norm_f_vertex("norm_f_vertex", mesh.m_nvertices, mesh.m_nlayers),
      m_norm_rvort_edge("norm_rvort_edge", mesh.m_nedges, mesh.m_nlayers),
      m_norm_f_edge("norm_f_edge", mesh.m_nedges, mesh.m_nlayers),
      m_norm_rvort_cell("norm_rvort_cell", mesh.m_ncells, mesh.m_nlayers) {}

void ShallowWaterModel::compute_auxiliary_variables(RealConst2d h_cell,
                                                    RealConst2d vn_edge,
                                                    RealConst3d tr_cell) const {

  YAKL_SCOPE(max_level_edge_top, m_mesh->m_max_level_edge_top);
  YAKL_SCOPE(cells_on_edge, m_mesh->m_cells_on_edge);
  YAKL_SCOPE(h_mean_edge, m_h_mean_edge);
  YAKL_SCOPE(h_flux_edge, m_h_flux_edge);
  YAKL_SCOPE(h_drag_edge, m_h_drag_edge);
  parallel_for(
      "compute_h_edge", SimpleBounds<2>(m_mesh->m_nedges, m_mesh->m_nlayers),
      YAKL_LAMBDA(Int iedge, Int k) {
        Real h_mean = -0;
        for (Int j = 0; j < 2; ++j) {
          Int jcell = cells_on_edge(iedge, j);
          h_mean += h_cell(jcell, k);
        }
        h_mean /= 2;

        h_mean_edge(iedge, k) = h_mean;
        h_flux_edge(iedge, k) = h_mean;
        h_drag_edge(iedge, k) = h_mean;
      });

  YAKL_SCOPE(dc_edge, m_mesh->m_dc_edge);
  YAKL_SCOPE(edges_on_vertex, m_mesh->m_edges_on_vertex);
  YAKL_SCOPE(edge_sign_on_vertex, m_mesh->m_edge_sign_on_vertex);
  YAKL_SCOPE(area_triangle, m_mesh->m_area_triangle);
  YAKL_SCOPE(max_level_vertex_bot, m_mesh->m_max_level_vertex_bot);
  YAKL_SCOPE(rcirc_vertex, m_rcirc_vertex);
  YAKL_SCOPE(rvort_vertex, m_rvort_vertex);
  parallel_for(
      "compute_rcirc_and_rvort_vertex",
      SimpleBounds<2>(m_mesh->m_nvertices, m_mesh->m_nlayers),
      YAKL_LAMBDA(Int ivertex, Int k) {
        Real rcirc = -0;
        for (Int j = 0; j < 3; ++j) {
          Int jedge = edges_on_vertex(ivertex, j);
          rcirc += dc_edge(jedge) * edge_sign_on_vertex(ivertex, j) *
                   vn_edge(jedge, k);
        }
        rcirc_vertex(ivertex, k) = rcirc;
        rvort_vertex(ivertex, k) = rcirc / area_triangle(ivertex);
      });

  YAKL_SCOPE(max_level_cell, m_mesh->m_max_level_cell);
  YAKL_SCOPE(nedges_on_cell, m_mesh->m_nedges_on_cell);
  YAKL_SCOPE(edges_on_cell, m_mesh->m_edges_on_cell);
  YAKL_SCOPE(dv_edge, m_mesh->m_dv_edge);
  YAKL_SCOPE(vertices_on_cell, m_mesh->m_vertices_on_cell);
  YAKL_SCOPE(kite_index_on_cell, m_mesh->m_kite_index_on_cell);
  YAKL_SCOPE(kiteareas_on_vertex, m_mesh->m_kiteareas_on_vertex);
  YAKL_SCOPE(area_cell, m_mesh->m_area_cell);
  YAKL_SCOPE(rvort_cell, m_rvort_cell);
  YAKL_SCOPE(ke_cell, m_ke_cell);
  YAKL_SCOPE(div_cell, m_div_cell);
  YAKL_SCOPE(norm_tr_cell, m_norm_tr_cell);
  YAKL_SCOPE(ntracers, m_ntracers);

  parallel_for(
      "compute_auxiliarys_cell",
      SimpleBounds<2>(m_mesh->m_ncells, m_mesh->m_nlayers),
      YAKL_LAMBDA(Int icell, Int k) {
        Real ke = -0;
        Real rvort = -0;
        Real div = -0;
        for (Int j = 0; j < nedges_on_cell(icell); ++j) {
          Int jedge = edges_on_cell(icell, j);
          Real area_edge = dv_edge(jedge) * dc_edge(jedge);
          ke += area_edge * vn_edge(jedge, k) * vn_edge(jedge, k) / 4;
          div += dv_edge(jedge) * vn_edge(jedge, k);

          Int jvertex = vertices_on_cell(icell, j);
          Int jkite = kite_index_on_cell(icell, j);
          rvort +=
              kiteareas_on_vertex(jvertex, jkite) * rvort_vertex(jvertex, k);
        }
        ke /= area_cell(icell);
        div /= area_cell(icell);
        rvort /= area_cell(icell);

        div_cell(icell, k) = div;
        ke_cell(icell, k) = ke;
        rvort_cell(icell, k) = rvort;

        for (Int l = 0; l < ntracers; ++l) {
          norm_tr_cell(l, icell, k) = tr_cell(l, icell, k) / h_cell(icell, k);
        }
      });

  YAKL_SCOPE(nedges_on_edge, m_mesh->m_nedges_on_edge);
  YAKL_SCOPE(edges_on_edge, m_mesh->m_edges_on_edge);
  YAKL_SCOPE(weights_on_edge, m_mesh->m_weights_on_edge);
  YAKL_SCOPE(vt_edge, m_vt_edge);
  parallel_for(
      "compute_vt_edge", SimpleBounds<2>(m_mesh->m_nedges, m_mesh->m_nlayers),
      YAKL_LAMBDA(Int iedge, Int k) {
        Real vt = -0;
        for (Int j = 0; j < nedges_on_edge(iedge); ++j) {
          Int jedge = edges_on_edge(iedge, j);
          vt += weights_on_edge(iedge, j) * vn_edge(jedge, k);
        }
        vt_edge(iedge, k) = vt;
      });

  YAKL_SCOPE(f_vertex, m_f_vertex);
  YAKL_SCOPE(cells_on_vertex, m_mesh->m_cells_on_vertex);
  YAKL_SCOPE(norm_rvort_vertex, m_norm_rvort_vertex);
  YAKL_SCOPE(norm_f_vertex, m_norm_f_vertex);

  parallel_for(
      "compute_norm_rvort_and_f_vertex",
      SimpleBounds<2>(m_mesh->m_nvertices, m_mesh->m_nlayers),
      YAKL_LAMBDA(Int ivertex, Int k) {
        Real h = -0;
        for (Int j = 0; j < 3; ++j) {
          Int jcell = cells_on_vertex(ivertex, j);
          h += kiteareas_on_vertex(ivertex, j) * h_cell(jcell, k);
        }
        h /= area_triangle(ivertex);
        norm_rvort_vertex(ivertex, k) = rvort_vertex(ivertex, k) / h;
        norm_f_vertex(ivertex, k) = f_vertex(ivertex) / h;
      });

  YAKL_SCOPE(max_level_edge_bot, m_mesh->m_max_level_edge_bot);
  YAKL_SCOPE(vertices_on_edge, m_mesh->m_vertices_on_edge);
  YAKL_SCOPE(norm_rvort_edge, m_norm_rvort_edge);
  YAKL_SCOPE(norm_f_edge, m_norm_f_edge);
  parallel_for(
      "compute_norm_rvort_and_f_edge",
      SimpleBounds<2>(m_mesh->m_nedges, m_mesh->m_nlayers),
      YAKL_LAMBDA(Int iedge, Int k) {
        Real norm_f = -0;
        Real norm_rvort = -0;

        for (Int j = 0; j < 2; ++j) {
          Int jvertex = vertices_on_edge(iedge, j);
          norm_rvort += norm_rvort_vertex(jvertex, k);
          norm_f += norm_f_vertex(jvertex, k);
        }
        norm_rvort /= 2;
        norm_f /= 2;

        norm_rvort_edge(iedge, k) = norm_rvort;
        norm_f_edge(iedge, k) = norm_f;
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
      "compute_htend", SimpleBounds<2>(m_mesh->m_ncells, m_mesh->m_nlayers),
      YAKL_LAMBDA(Int icell, Int k) {
        Real accum = -0;
        for (Int j = 0; j < nedges_on_cell(icell); ++j) {
          Int jedge = edges_on_cell(icell, j);
          accum += dv_edge(jedge) * edge_sign_on_cell(icell, j) *
                   h_flux_edge(jedge, k) * vn_edge(jedge, k);
        }

        if (add_mode == AddMode::increment) {
          h_tend_cell(icell, k) += -accum / area_cell(icell);
        }

        if (add_mode == AddMode::replace) {
          h_tend_cell(icell, k) = -accum / area_cell(icell);
        }
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

  parallel_for(
      "compute_vtend", SimpleBounds<2>(m_mesh->m_nedges, m_mesh->m_nlayers),
      YAKL_LAMBDA(Int iedge, Int k) {
        Real vn_tend = -0;

        Real qt = -0;
        for (Int j = 0; j < nedges_on_edge(iedge); ++j) {
          Int jedge = edges_on_edge(iedge, j);

          Real norm_vort = (norm_rvort_edge(iedge, k) + norm_f_edge(iedge, k) +
                            norm_rvort_edge(jedge, k) + norm_f_edge(jedge, k)) /
                           2;

          qt += weights_on_edge(iedge, j) * h_flux_edge(jedge, k) *
                vn_edge(jedge, k) * norm_vort;
        }

        Int icell0 = cells_on_edge(iedge, 0);
        Int icell1 = cells_on_edge(iedge, 1);

        Real ke_cell0 = ke_cell(icell0, k);
        Real ke_cell1 = ke_cell(icell1, k);

        Real grad_B = (ke_cell1 - ke_cell0 +
                       grav * (h_cell(icell1, k) - h_cell(icell0, k))) /
                      dc_edge(iedge);

        Real drag_force = (k == (max_level_edge_top(iedge) - 1))
                              ? -drag_coeff * std::sqrt(ke_cell0 + ke_cell1) *
                                    vn_edge(iedge, k) / h_drag_edge(iedge, k)
                              : 0;

        Int ivertex0 = vertices_on_edge(iedge, 0);
        Int ivertex1 = vertices_on_edge(iedge, 1);
        // TODO: add mesh scaling
        Real visc2 =
            visc_del2 *
            ((div_cell(icell1, k) - div_cell(icell0, k)) / dc_edge(iedge) -
             (rvort_vertex(ivertex1, k) - rvort_vertex(ivertex0, k)) /
                 dv_edge(iedge));

        vn_tend = qt - grad_B + drag_force + visc2;
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
  YAKL_SCOPE(nedges_on_cell, m_mesh->m_nedges_on_cell);
  YAKL_SCOPE(edges_on_cell, m_mesh->m_edges_on_cell);
  YAKL_SCOPE(dv_edge, m_mesh->m_dv_edge);
  YAKL_SCOPE(edge_sign_on_cell, m_mesh->m_edge_sign_on_cell);
  YAKL_SCOPE(area_cell, m_mesh->m_area_cell);
  YAKL_SCOPE(cells_on_edge, m_mesh->m_cells_on_edge);

  YAKL_SCOPE(h_flux_edge, m_h_flux_edge);
  YAKL_SCOPE(norm_tr_cell, m_norm_tr_cell);
  YAKL_SCOPE(ntracers, m_ntracers);
  parallel_for(
      "compute_tr_tend",
      SimpleBounds<3>(ntracers, m_mesh->m_ncells, m_mesh->m_nlayers),
      YAKL_LAMBDA(Int l, Int icell, Int k) {
        Real accum = -0;
        for (Int j = 0; j < nedges_on_cell(icell); ++j) {
          Int jedge = edges_on_cell(icell, j);

          Int jcell0 = cells_on_edge(jedge, 0);
          Int jcell1 = cells_on_edge(jedge, 1);

          Real norm_tr_edge =
              (norm_tr_cell(l, jcell0, k) + norm_tr_cell(l, jcell1, k)) / 2;

          accum += dv_edge(jedge) * edge_sign_on_cell(icell, j) *
                   h_flux_edge(jedge, k) * norm_tr_edge * vn_edge(jedge, k);
        }

        if (add_mode == AddMode::increment) {
          tr_tend_cell(l, icell, k) += -accum / area_cell(icell);
        }

        if (add_mode == AddMode::replace) {
          tr_tend_cell(l, icell, k) = -accum / area_cell(icell);
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
    PlanarHexagonalMesh &mesh, const LinearShallowWaterParams &params)
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
