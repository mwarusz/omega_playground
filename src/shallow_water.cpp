#include <shallow_water.hpp>

namespace omega {

// Base

ShallowWaterBase::ShallowWaterBase(PlanarHexagonalMesh &mesh, Real f0)
    : mesh(&mesh), f0(f0) {}
ShallowWaterBase::ShallowWaterBase(PlanarHexagonalMesh &mesh, Real f0,
                                   Real grav)
    : mesh(&mesh), f0(f0), grav(grav) {}

Real ShallowWaterBase::mass_integral(RealConst2d h) const {
  Real1d column_mass("column_mass", mesh->ncells);

  YAKL_SCOPE(area_cell, mesh->area_cell);
  YAKL_SCOPE(max_level_cell, mesh->max_level_cell);

  parallel_for(
      "compute_column_mass", mesh->ncells, YAKL_LAMBDA(Int icell) {
        column_mass(icell) = 0;
        for (Int k = 0; k < max_level_cell(icell); ++k) {
          column_mass(icell) += area_cell(icell) * h(icell, k);
        }
      });
  return yakl::intrinsics::sum(column_mass);
}

Real ShallowWaterBase::circulation_integral(RealConst2d v) const {
  Real1d column_circulation("column_circulation", mesh->nvertices);

  YAKL_SCOPE(dc_edge, mesh->dc_edge);
  YAKL_SCOPE(edges_on_vertex, mesh->edges_on_vertex);
  YAKL_SCOPE(edge_sign_on_vertex, mesh->edge_sign_on_vertex);
  YAKL_SCOPE(area_triangle, mesh->area_triangle);
  YAKL_SCOPE(f0, this->f0);
  YAKL_SCOPE(max_level_vertex_bot, mesh->max_level_vertex_bot);

  parallel_for(
      "compute_column_circulation", mesh->nvertices, YAKL_LAMBDA(Int ivertex) {
        column_circulation(ivertex) = 0;
        for (Int k = 0; k < max_level_vertex_bot(ivertex); ++k) {
          Real cir_i = -0;
          for (Int j = 0; j < 3; ++j) {
            Int jedge = edges_on_vertex(ivertex, j);
            cir_i +=
                dc_edge(jedge) * edge_sign_on_vertex(ivertex, j) * v(jedge, k);
          }
          column_circulation(ivertex) += cir_i + f0 * area_triangle(ivertex);
        }
      });
  return yakl::intrinsics::sum(column_circulation);
}

// Nonlinear

ShallowWater::ShallowWater(PlanarHexagonalMesh &mesh, Real f0)
    : ShallowWaterBase(mesh, f0), hflux("hflux", mesh.nedges, mesh.nlayers) {}
ShallowWater::ShallowWater(PlanarHexagonalMesh &mesh, Real f0, Real grav)
    : ShallowWaterBase(mesh, f0, grav),
      hflux("hflux", mesh.nedges, mesh.nlayers) {}

void ShallowWater::compute_h_tendency(Real2d htend, RealConst2d h,
                                      RealConst2d v) const {
  YAKL_SCOPE(nedges_on_cell, mesh->nedges_on_cell);
  YAKL_SCOPE(edges_on_cell, mesh->edges_on_cell);
  YAKL_SCOPE(dv_edge, mesh->dv_edge);
  YAKL_SCOPE(cells_on_edge, mesh->cells_on_edge);
  YAKL_SCOPE(edge_sign_on_cell, mesh->edge_sign_on_cell);
  YAKL_SCOPE(area_cell, mesh->area_cell);
  YAKL_SCOPE(max_level_edge_top, mesh->max_level_edge_top);
  YAKL_SCOPE(max_level_cell, mesh->max_level_cell);

  YAKL_SCOPE(hflux, this->hflux);
  parallel_for(
      "compute_hflux", mesh->nedges, YAKL_LAMBDA(Int iedge) {
        for (Int k = 0; k < max_level_edge_top(k); ++k) {
          Real he = -0;
          for (Int j = 0; j < 2; ++j) {
            Int jcell = cells_on_edge(iedge, j);
            he += h(jcell, k);
          }
          he /= 2;
          hflux(iedge, k) = he * v(iedge, k);
        }
      });

  parallel_for(
      "compute_htend", mesh->ncells, YAKL_LAMBDA(Int icell) {
        for (Int k = 0; k < max_level_cell(k); ++k) {
          Real accum = -0;
          for (Int j = 0; j < nedges_on_cell(icell); ++j) {
            Int jedge = edges_on_cell(icell, j);
            accum +=
                dv_edge(jedge) * edge_sign_on_cell(icell, j) * hflux(jedge, k);
          }
          htend(icell, k) += -accum / area_cell(icell);
        }
      });
}

void ShallowWater::compute_v_tendency(Real2d vtend, RealConst2d h,
                                      RealConst2d v) const {
  YAKL_SCOPE(edges_on_vertex, mesh->edges_on_vertex);
  YAKL_SCOPE(edge_sign_on_vertex, mesh->edge_sign_on_vertex);
  YAKL_SCOPE(kiteareas_on_vertex, mesh->kiteareas_on_vertex);
  YAKL_SCOPE(cells_on_vertex, mesh->cells_on_vertex);
  YAKL_SCOPE(area_triangle, mesh->area_triangle);
  YAKL_SCOPE(nedges_on_edge, mesh->nedges_on_edge);
  YAKL_SCOPE(nedges_on_cell, mesh->nedges_on_cell);
  YAKL_SCOPE(edges_on_edge, mesh->edges_on_edge);
  YAKL_SCOPE(edges_on_cell, mesh->edges_on_cell);
  YAKL_SCOPE(area_cell, mesh->area_cell);
  YAKL_SCOPE(weights_on_edge, mesh->weights_on_edge);
  YAKL_SCOPE(vertices_on_edge, mesh->vertices_on_edge);
  YAKL_SCOPE(dv_edge, mesh->dv_edge);
  YAKL_SCOPE(dc_edge, mesh->dc_edge);
  YAKL_SCOPE(cells_on_edge, mesh->cells_on_edge);
  YAKL_SCOPE(grav, this->grav);
  YAKL_SCOPE(f0, this->f0);
  YAKL_SCOPE(hflux, this->hflux);

  YAKL_SCOPE(max_level_vertex_bot, mesh->max_level_vertex_bot);
  YAKL_SCOPE(max_level_edge_bot, mesh->max_level_edge_bot);
  YAKL_SCOPE(max_level_edge_top, mesh->max_level_edge_top);
  YAKL_SCOPE(max_level_cell, mesh->max_level_cell);

  Real2d qv("qv", mesh->nvertices, mesh->nlayers);
  parallel_for(
      "compute_qv", mesh->nvertices, YAKL_LAMBDA(Int ivertex) {
        for (Int k = 0; k < max_level_vertex_bot(ivertex); ++k) {
          Real qv_i = -0;
          for (Int j = 0; j < 3; ++j) {
            Int jedge = edges_on_vertex(ivertex, j);
            qv_i +=
                dc_edge(jedge) * edge_sign_on_vertex(ivertex, j) * v(jedge, k);
          }

          Real hv_i = -0;
          for (Int j = 0; j < 3; ++j) {
            Int jcell = cells_on_vertex(ivertex, j);
            hv_i += kiteareas_on_vertex(ivertex, j) * h(jcell, k);
          }
          qv(ivertex, k) = (qv_i + f0 * area_triangle(ivertex)) / hv_i;
        }
      });

  Real2d qe("qe", mesh->nedges, mesh->nlayers);
  parallel_for(
      "compute_qe", mesh->nedges, YAKL_LAMBDA(Int iedge) {
        for (Int k = 0; k < max_level_edge_bot(iedge); ++k) {
          Real qe_i = -0;
          for (Int j = 0; j < 2; ++j) {
            Int jvertex = vertices_on_edge(iedge, j);
            qe_i += qv(jvertex, k);
          }
          qe(iedge, k) = qe_i / 2;
        }
      });

  Real2d K("K", mesh->ncells, mesh->nlayers);
  parallel_for(
      "compute_K", mesh->ncells, YAKL_LAMBDA(Int icell) {
        for (Int k = 0; k < max_level_cell(icell); ++k) {
          Real K_i = -0;
          for (Int j = 0; j < nedges_on_cell(icell); ++j) {
            Int jedge = edges_on_cell(icell, j);
            Real area_edge = dv_edge(jedge) * dc_edge(jedge);
            K_i += area_edge * v(jedge, k) * v(jedge, k) / 4;
          }
          K_i /= area_cell(icell);
          K(icell, k) = K_i;
        }
      });

  parallel_for(
      "compute_vtend", mesh->nedges, YAKL_LAMBDA(Int iedge) {
        for (Int k = 0; k < max_level_edge_top(iedge); ++k) {
          Real qt = -0;
          for (Int j = 0; j < nedges_on_edge(iedge); ++j) {
            Int jedge = edges_on_edge(iedge, j);

            qt += weights_on_edge(iedge, j) * dv_edge(jedge) * hflux(jedge, k) *
                  (qe(iedge, k) + qe(jedge, k)) / 2;
          }
          qt /= dc_edge(iedge);

          Int icell0 = cells_on_edge(iedge, 0);
          Int icell1 = cells_on_edge(iedge, 1);

          Real grad_B = (K(icell1, k) - K(icell0, k) +
                         grav * (h(icell1, k) - h(icell0, k))) /
                        dc_edge(iedge);

          vtend(iedge, k) += qt - grad_B;
        }
      });
}

Real ShallowWater::energy_integral(RealConst2d h, RealConst2d v) const {
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
            K += area_edge * v(jedge, k) * v(jedge, k) / 4;
          }
          K /= area_cell(icell);
          column_energy(icell) +=
              area_cell(icell) *
              (grav * h(icell, k) * h(icell, k) / 2 + h(icell, k) * K);
        }
      });
  return yakl::intrinsics::sum(column_energy);
}

// Linear

LinearShallowWater::LinearShallowWater(PlanarHexagonalMesh &mesh, Real h0,
                                       Real f0)
    : ShallowWaterBase(mesh, f0), h0(h0) {}

LinearShallowWater::LinearShallowWater(PlanarHexagonalMesh &mesh, Real h0,
                                       Real f0, Real grav)
    : ShallowWaterBase(mesh, f0, grav), h0(h0) {}

void LinearShallowWater::compute_h_tendency(Real2d htend, RealConst2d h,
                                            RealConst2d v) const {
  YAKL_SCOPE(nedges_on_cell, mesh->nedges_on_cell);
  YAKL_SCOPE(edges_on_cell, mesh->edges_on_cell);
  YAKL_SCOPE(dv_edge, mesh->dv_edge);
  YAKL_SCOPE(edge_sign_on_cell, mesh->edge_sign_on_cell);
  YAKL_SCOPE(area_cell, mesh->area_cell);
  YAKL_SCOPE(max_level_cell, mesh->max_level_cell);
  YAKL_SCOPE(h0, this->h0);

  parallel_for(
      "compute_htend", mesh->ncells, YAKL_LAMBDA(Int icell) {
        for (Int k = 0; k < max_level_cell(icell); ++k) {
          Real accum = -0;
          for (Int j = 0; j < nedges_on_cell(icell); ++j) {
            Int jedge = edges_on_cell(icell, j);
            accum += dv_edge(jedge) * edge_sign_on_cell(icell, j) * v(jedge, k);
          }
          htend(icell, k) += -h0 * accum / area_cell(icell);
        }
      });
}

void LinearShallowWater::compute_v_tendency(Real2d vtend, RealConst2d h,
                                            RealConst2d v) const {
  YAKL_SCOPE(nedges_on_edge, mesh->nedges_on_edge);
  YAKL_SCOPE(edges_on_edge, mesh->edges_on_edge);
  YAKL_SCOPE(weights_on_edge, mesh->weights_on_edge);
  YAKL_SCOPE(dv_edge, mesh->dv_edge);
  YAKL_SCOPE(dc_edge, mesh->dc_edge);
  YAKL_SCOPE(cells_on_edge, mesh->cells_on_edge);
  YAKL_SCOPE(max_level_edge_top, mesh->max_level_edge_top);
  YAKL_SCOPE(grav, this->grav);
  YAKL_SCOPE(f0, this->f0);

  parallel_for(
      "compute_vtend", mesh->nedges, YAKL_LAMBDA(Int iedge) {
        for (Int k = 0; k < max_level_edge_top(iedge); ++k) {
          Real vt = -0;
          for (Int j = 0; j < nedges_on_edge(iedge); ++j) {
            Int jedge = edges_on_edge(iedge, j);
            vt += weights_on_edge(iedge, j) * dv_edge(jedge) * v(jedge, k);
          }
          vt /= dc_edge(iedge);

          Int icell0 = cells_on_edge(iedge, 0);
          Int icell1 = cells_on_edge(iedge, 1);
          Real grad_h = (h(icell1, k) - h(icell0, k)) / dc_edge(iedge);

          vtend(iedge, k) += f0 * vt - grav * grad_h;
        }
      });
}

Real LinearShallowWater::energy_integral(RealConst2d h, RealConst2d v) const {
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
            K += area_edge * v(jedge, k) * v(jedge, k) / 4;
          }
          K /= area_cell(icell);
          column_energy(icell) +=
              area_cell(icell) *
              (grav * h(icell, k) * h(icell, k) / 2 + h0 * K);
        }
      });
  return yakl::intrinsics::sum(column_energy);
}

} // namespace omega
