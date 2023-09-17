#include <shallow_water.hpp>

namespace omega {

// Base

ShallowWaterBase::ShallowWaterBase(PlanarHexagonalMesh &mesh, Real f0)
    : mesh(&mesh), f_vertex("f_vertex", mesh.nvertices),
      f_edge("f_edge", mesh.nedges) {
  yakl::memset(f_vertex, f0);
  yakl::memset(f_edge, f0);
}
ShallowWaterBase::ShallowWaterBase(PlanarHexagonalMesh &mesh, Real f0,
                                   Real grav)
    : mesh(&mesh), f_vertex("f_vertex", mesh.nvertices),
      f_edge("f_edge", mesh.nedges), grav(grav) {
  yakl::memset(f_vertex, f0);
  yakl::memset(f_edge, f0);
}

Real ShallowWaterBase::mass_integral(RealConst2d h_cell) const {
  Real1d column_mass("column_mass", mesh->ncells);

  YAKL_SCOPE(area_cell, mesh->area_cell);
  YAKL_SCOPE(max_level_cell, mesh->max_level_cell);

  parallel_for(
      "compute_column_mass", mesh->ncells, YAKL_LAMBDA(Int icell) {
        column_mass(icell) = 0;
        for (Int k = 0; k < max_level_cell(icell); ++k) {
          column_mass(icell) += area_cell(icell) * h_cell(icell, k);
        }
      });
  return yakl::intrinsics::sum(column_mass);
}

Real ShallowWaterBase::circulation_integral(RealConst2d vn_edge) const {
  Real1d column_circulation("column_circulation", mesh->nvertices);

  YAKL_SCOPE(dc_edge, mesh->dc_edge);
  YAKL_SCOPE(edges_on_vertex, mesh->edges_on_vertex);
  YAKL_SCOPE(edge_sign_on_vertex, mesh->edge_sign_on_vertex);
  YAKL_SCOPE(area_triangle, mesh->area_triangle);
  YAKL_SCOPE(f_vertex, this->f_vertex);
  YAKL_SCOPE(max_level_vertex_bot, mesh->max_level_vertex_bot);

  parallel_for(
      "compute_column_circulation", mesh->nvertices, YAKL_LAMBDA(Int ivertex) {
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

// Nonlinear

ShallowWater::ShallowWater(PlanarHexagonalMesh &mesh, Real f0, Real grav)
    : ShallowWaterBase(mesh, f0, grav),
      h_flux_edge("h_flux_edge", mesh.nedges, mesh.nlayers),
      h_mean_edge("h_mean_edge", mesh.nedges, mesh.nlayers),
      h_drag_edge("h_drag_edge", mesh.nedges, mesh.nlayers),
      rcirc_vertex("rcirc_vertex", mesh.nvertices, mesh.nlayers),
      rvort_vertex("rvort_vertex", mesh.nvertices, mesh.nlayers),
      rvort_cell("rvort_cell", mesh.ncells, mesh.nlayers),
      ke_cell("ke_cell", mesh.ncells, mesh.nlayers),
      vt_edge("vt_edge", mesh.nedges, mesh.nlayers),
      norm_rvort_vertex("norm_rvort_vertex", mesh.nvertices, mesh.nlayers),
      norm_f_vertex("norm_f_vertex", mesh.nvertices, mesh.nlayers),
      norm_rvort_edge("norm_rvort_edge", mesh.nedges, mesh.nlayers),
      norm_f_edge("norm_f_edge", mesh.nedges, mesh.nlayers),
      norm_rvort_cell("norm_rvort_cell", mesh.ncells, mesh.nlayers) {}

ShallowWater::ShallowWater(PlanarHexagonalMesh &mesh, Real f0)
    : ShallowWater(mesh, f0, 9.81) {}

void ShallowWater::compute_auxiliary_variables(RealConst2d h_cell,
                                               RealConst2d vn_edge) const {

  YAKL_SCOPE(max_level_edge_top, mesh->max_level_edge_top);
  YAKL_SCOPE(cells_on_edge, mesh->cells_on_edge);
  YAKL_SCOPE(h_mean_edge, this->h_mean_edge);
  YAKL_SCOPE(h_flux_edge, this->h_flux_edge);
  YAKL_SCOPE(h_drag_edge, this->h_drag_edge);
  parallel_for(
      "compute_h_edge", SimpleBounds<2>(mesh->nedges, mesh->nlayers),
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

  YAKL_SCOPE(dc_edge, mesh->dc_edge);
  YAKL_SCOPE(edges_on_vertex, mesh->edges_on_vertex);
  YAKL_SCOPE(edge_sign_on_vertex, mesh->edge_sign_on_vertex);
  YAKL_SCOPE(area_triangle, mesh->area_triangle);
  YAKL_SCOPE(max_level_vertex_bot, mesh->max_level_vertex_bot);
  YAKL_SCOPE(rcirc_vertex, this->rcirc_vertex);
  YAKL_SCOPE(rvort_vertex, this->rvort_vertex);
  parallel_for(
      "compute_rcirc_and_rvort_vertex",
      SimpleBounds<2>(mesh->nvertices, mesh->nlayers),
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

  YAKL_SCOPE(max_level_cell, mesh->max_level_cell);
  YAKL_SCOPE(nedges_on_cell, mesh->nedges_on_cell);
  YAKL_SCOPE(edges_on_cell, mesh->edges_on_cell);
  YAKL_SCOPE(dv_edge, mesh->dv_edge);
  YAKL_SCOPE(vertices_on_cell, mesh->vertices_on_cell);
  YAKL_SCOPE(kite_index_on_cell, mesh->kite_index_on_cell);
  YAKL_SCOPE(kiteareas_on_vertex, mesh->kiteareas_on_vertex);
  YAKL_SCOPE(area_cell, mesh->area_cell);
  YAKL_SCOPE(rvort_cell, this->rvort_cell);
  YAKL_SCOPE(ke_cell, this->ke_cell);

  parallel_for(
      "compute_rvort_and_ke_cell", SimpleBounds<2>(mesh->ncells, mesh->nlayers),
      YAKL_LAMBDA(Int icell, Int k) {
        Real ke = -0;
        Real rvort = -0;
        for (Int j = 0; j < nedges_on_cell(icell); ++j) {
          Int jedge = edges_on_cell(icell, j);
          Real area_edge = dv_edge(jedge) * dc_edge(jedge);
          ke += area_edge * vn_edge(jedge, k) * vn_edge(jedge, k) / 4;

          Int jvertex = vertices_on_cell(icell, j);
          Int jkite = kite_index_on_cell(icell, j);
          rvort +=
              kiteareas_on_vertex(jvertex, jkite) * rvort_vertex(jvertex, k);
        }
        ke /= area_cell(icell);
        rvort /= area_cell(icell);

        ke_cell(icell, k) = ke;
        rvort_cell(icell, k) = rvort;
      });

  YAKL_SCOPE(nedges_on_edge, mesh->nedges_on_edge);
  YAKL_SCOPE(edges_on_edge, mesh->edges_on_edge);
  YAKL_SCOPE(weights_on_edge, mesh->weights_on_edge);
  YAKL_SCOPE(vt_edge, this->vt_edge);
  parallel_for(
      "compute_vt_edge", SimpleBounds<2>(mesh->nedges, mesh->nlayers),
      YAKL_LAMBDA(Int iedge, Int k) {
        Real vt = -0;
        for (Int j = 0; j < nedges_on_edge(iedge); ++j) {
          Int jedge = edges_on_edge(iedge, j);
          vt += weights_on_edge(iedge, j) * vn_edge(jedge, k);
        }
        vt_edge(iedge, k) = vt;
      });

  YAKL_SCOPE(f_vertex, this->f_vertex);
  YAKL_SCOPE(cells_on_vertex, mesh->cells_on_vertex);
  YAKL_SCOPE(norm_rvort_vertex, this->norm_rvort_vertex);
  YAKL_SCOPE(norm_f_vertex, this->norm_f_vertex);

  parallel_for(
      "compute_norm_rvort_and_f_vertex",
      SimpleBounds<2>(mesh->nvertices, mesh->nlayers),
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

  YAKL_SCOPE(max_level_edge_bot, mesh->max_level_edge_bot);
  YAKL_SCOPE(vertices_on_edge, mesh->vertices_on_edge);
  YAKL_SCOPE(norm_rvort_edge, this->norm_rvort_edge);
  YAKL_SCOPE(norm_f_edge, this->norm_f_edge);
  parallel_for(
      "compute_norm_rvort_and_f_edge",
      SimpleBounds<2>(mesh->nedges, mesh->nlayers),
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

void ShallowWater::compute_h_tendency(Real2d h_tend_cell, RealConst2d h_cell,
                                      RealConst2d vn_edge,
                                      AddMode add_mode) const {
  YAKL_SCOPE(nedges_on_cell, mesh->nedges_on_cell);
  YAKL_SCOPE(edges_on_cell, mesh->edges_on_cell);
  YAKL_SCOPE(dv_edge, mesh->dv_edge);
  YAKL_SCOPE(edge_sign_on_cell, mesh->edge_sign_on_cell);
  YAKL_SCOPE(area_cell, mesh->area_cell);
  YAKL_SCOPE(max_level_cell, mesh->max_level_cell);

  YAKL_SCOPE(h_flux_edge, this->h_flux_edge);
  parallel_for(
      "compute_htend", SimpleBounds<2>(mesh->ncells, mesh->nlayers),
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

void ShallowWater::compute_vn_tendency(Real2d vn_tend_edge, RealConst2d h_cell,
                                       RealConst2d vn_edge,
                                       AddMode add_mode) const {
  YAKL_SCOPE(max_level_edge_top, mesh->max_level_edge_top);
  YAKL_SCOPE(nedges_on_edge, mesh->nedges_on_edge);
  YAKL_SCOPE(edges_on_edge, mesh->edges_on_edge);
  YAKL_SCOPE(weights_on_edge, mesh->weights_on_edge);
  YAKL_SCOPE(dc_edge, mesh->dc_edge);
  YAKL_SCOPE(cells_on_edge, mesh->cells_on_edge);

  YAKL_SCOPE(grav, this->grav);
  YAKL_SCOPE(norm_rvort_edge, this->norm_rvort_edge);
  YAKL_SCOPE(norm_f_edge, this->norm_f_edge);
  YAKL_SCOPE(h_flux_edge, this->h_flux_edge);
  YAKL_SCOPE(ke_cell, this->ke_cell);

  parallel_for(
      "compute_vtend", SimpleBounds<2>(mesh->nedges, mesh->nlayers),
      YAKL_LAMBDA(Int iedge, Int k) {
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

        Real grad_B = (ke_cell(icell1, k) - ke_cell(icell0, k) +
                       grav * (h_cell(icell1, k) - h_cell(icell0, k))) /
                      dc_edge(iedge);

        if (add_mode == AddMode::increment) {
          vn_tend_edge(iedge, k) += qt - grad_B;
        }
        if (add_mode == AddMode::replace) {
          vn_tend_edge(iedge, k) = qt - grad_B;
        }
      });
}

Real ShallowWater::energy_integral(RealConst2d h_cell,
                                   RealConst2d vn_edge) const {
  Real1d column_energy("column_energy", mesh->ncells);

  YAKL_SCOPE(nedges_on_cell, mesh->nedges_on_cell);
  YAKL_SCOPE(edges_on_cell, mesh->edges_on_cell);
  YAKL_SCOPE(dv_edge, mesh->dv_edge);
  YAKL_SCOPE(dc_edge, mesh->dc_edge);
  YAKL_SCOPE(area_cell, mesh->area_cell);
  YAKL_SCOPE(max_level_cell, mesh->max_level_cell);
  YAKL_SCOPE(grav, this->grav);

  parallel_for(
      "compute_column_energy", mesh->ncells, YAKL_LAMBDA(Int icell) {
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

LinearShallowWater::LinearShallowWater(PlanarHexagonalMesh &mesh, Real h0,
                                       Real f0, Real grav)
    : ShallowWaterBase(mesh, f0, grav), h0(h0) {}

LinearShallowWater::LinearShallowWater(PlanarHexagonalMesh &mesh, Real h0,
                                       Real f0)
    : LinearShallowWater(mesh, h0, f0, 9.81) {}

void LinearShallowWater::compute_h_tendency(Real2d h_tend_cell,
                                            RealConst2d h_cell,
                                            RealConst2d vn_edge,
                                            AddMode add_mode) const {
  YAKL_SCOPE(nedges_on_cell, mesh->nedges_on_cell);
  YAKL_SCOPE(edges_on_cell, mesh->edges_on_cell);
  YAKL_SCOPE(dv_edge, mesh->dv_edge);
  YAKL_SCOPE(edge_sign_on_cell, mesh->edge_sign_on_cell);
  YAKL_SCOPE(area_cell, mesh->area_cell);
  YAKL_SCOPE(max_level_cell, mesh->max_level_cell);
  YAKL_SCOPE(h0, this->h0);

  parallel_for(
      "compute_htend", SimpleBounds<2>(mesh->ncells, mesh->nlayers),
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

void LinearShallowWater::compute_vn_tendency(Real2d vn_tend_edge,
                                             RealConst2d h_cell,
                                             RealConst2d vn_edge,
                                             AddMode add_mode) const {
  YAKL_SCOPE(nedges_on_edge, mesh->nedges_on_edge);
  YAKL_SCOPE(edges_on_edge, mesh->edges_on_edge);
  YAKL_SCOPE(weights_on_edge, mesh->weights_on_edge);
  YAKL_SCOPE(dv_edge, mesh->dv_edge);
  YAKL_SCOPE(dc_edge, mesh->dc_edge);
  YAKL_SCOPE(cells_on_edge, mesh->cells_on_edge);
  YAKL_SCOPE(max_level_edge_top, mesh->max_level_edge_top);
  YAKL_SCOPE(grav, this->grav);
  YAKL_SCOPE(f_edge, this->f_edge);

  parallel_for(
      "compute_vtend", SimpleBounds<2>(mesh->nedges, mesh->nlayers),
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

Real LinearShallowWater::energy_integral(RealConst2d h_cell,
                                         RealConst2d vn_edge) const {
  Real1d column_energy("column_energy", mesh->ncells);

  YAKL_SCOPE(nedges_on_cell, mesh->nedges_on_cell);
  YAKL_SCOPE(edges_on_cell, mesh->edges_on_cell);
  YAKL_SCOPE(dv_edge, mesh->dv_edge);
  YAKL_SCOPE(dc_edge, mesh->dc_edge);
  YAKL_SCOPE(area_cell, mesh->area_cell);
  YAKL_SCOPE(max_level_cell, mesh->max_level_cell);
  YAKL_SCOPE(grav, this->grav);
  YAKL_SCOPE(h0, this->h0);

  parallel_for(
      "compute_column_energy", mesh->ncells, YAKL_LAMBDA(Int icell) {
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
