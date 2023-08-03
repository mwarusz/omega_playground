#include <shallow_water.hpp>

namespace omega {

// Nonlinear

ShallowWater::ShallowWater(PlanarHexagonalMesh &mesh) : mesh(&mesh) {}
ShallowWater::ShallowWater(PlanarHexagonalMesh &mesh, Real grav) :
    mesh(&mesh), grav(grav) {}

void ShallowWater::compute_v_tendency(Real1d vtend, Real1d h, Real1d v) const {
  YAKL_SCOPE(edges_on_vertex, mesh->edges_on_vertex);
  YAKL_SCOPE(orient_on_vertex, mesh->orient_on_vertex);
  YAKL_SCOPE(area_triangle, mesh->area_triangle);
  YAKL_SCOPE(nedges_on_edge, mesh->nedges_on_edge);
  YAKL_SCOPE(edges_on_edge, mesh->edges_on_edge);
  YAKL_SCOPE(weights_on_edge, mesh->weights_on_edge);
  YAKL_SCOPE(dv_edge, mesh->dv_edge);
  YAKL_SCOPE(dc_edge, mesh->dc_edge);
  YAKL_SCOPE(cells_on_edge, mesh->cells_on_edge);
  YAKL_SCOPE(grav, this->grav);
  YAKL_SCOPE(f0, this->f0);
  
  Real1d qv("qv", mesh.vertices);
  parallel_for("compute_qv", mesh->nvertices, YAKL_LAMBDA (Int ivertex) {
      Real qv_i = -0;
      for (Int j = 0; j < 3; ++j) {
        Int jedge = edges_on_vertex(ivertex, j);
        qv_i += dc_edge(jedge) * orient_on_vertex(ivertex, j) * edge_field(jedge);
      }
      qv(ivertex) = qv_i / area_triangle(ivertex);
  });

  //Real1d qe("qe", mesh.nedges);
  //parallel_for("compute_qe", mesh->nedges, YAKL_LAMBDA (Int iedge) {
  //    Real qe_i = -0;
  //    for (Int j = 0; j < 2; ++j) {
  //      Int jvertex = vertices_on_edge(iedge, j);
  //      qe_i += 
  //    }
  //});

  parallel_for("compute_v_tendency", mesh->nedges, YAKL_LAMBDA (Int iedge) {
      Real vt = -0;
      for (Int j = 0; j < nedges_on_edge(iedge); ++j) {
        Int jedge = edges_on_edge(iedge, j);
        vt += weights_on_edge(iedge, j) * dv_edge(jedge) * v(jedge);
      }
      vt /= dc_edge(iedge);

      Int icell0 = cells_on_edge(iedge, 0);
      Int icell1 = cells_on_edge(iedge, 1);
      Real grad_h = (h(icell1) - h(icell0)) / dc_edge(iedge);

      vtend(iedge) += f0 * vt - grav * grad_h;
  });
}

// Linear

LinearShallowWater::LinearShallowWater(PlanarHexagonalMesh &mesh, Real h0, Real f0) :
    ShallowWater(mesh), h0(h0), f0(f0) {}

LinearShallowWater::LinearShallowWater(PlanarHexagonalMesh &mesh, Real h0, Real f0, Real grav) :
   ShallowWater(mesh, grav), h0(h0), f0(f0) {}

void LinearShallowWater::compute_h_tendency(Real1d htend, Real1d h, Real1d v) const {
  YAKL_SCOPE(nedges_on_cell, mesh->nedges_on_cell);
  YAKL_SCOPE(edges_on_cell, mesh->edges_on_cell);
  YAKL_SCOPE(dv_edge, mesh->dv_edge);
  YAKL_SCOPE(orient_on_cell, mesh->orient_on_cell);
  YAKL_SCOPE(area_cell, mesh->area_cell);
  YAKL_SCOPE(h0, this->h0);

  parallel_for("compute_h_tendency", mesh->ncells, YAKL_LAMBDA (Int icell) {
      Real accum = -0;
      for (Int j = 0; j < nedges_on_cell(icell); ++j) {
        Int iedge = edges_on_cell(icell, j);
        accum += dv_edge(iedge) * orient_on_cell(icell, j) * v(iedge);
      }
      htend(icell) += -h0 * accum / area_cell(icell);
  });
}

void LinearShallowWater::compute_v_tendency(Real1d vtend, Real1d h, Real1d v) const {
  YAKL_SCOPE(nedges_on_edge, mesh->nedges_on_edge);
  YAKL_SCOPE(edges_on_edge, mesh->edges_on_edge);
  YAKL_SCOPE(weights_on_edge, mesh->weights_on_edge);
  YAKL_SCOPE(dv_edge, mesh->dv_edge);
  YAKL_SCOPE(dc_edge, mesh->dc_edge);
  YAKL_SCOPE(cells_on_edge, mesh->cells_on_edge);
  YAKL_SCOPE(grav, this->grav);
  YAKL_SCOPE(f0, this->f0);

  parallel_for("compute_v_tendency", mesh->nedges, YAKL_LAMBDA (Int iedge) {
      Real vt = -0;
      for (Int j = 0; j < nedges_on_edge(iedge); ++j) {
        Int jedge = edges_on_edge(iedge, j);
        vt += weights_on_edge(iedge, j) * dv_edge(jedge) * v(jedge);
      }
      vt /= dc_edge(iedge);

      Int icell0 = cells_on_edge(iedge, 0);
      Int icell1 = cells_on_edge(iedge, 1);
      Real grad_h = (h(icell1) - h(icell0)) / dc_edge(iedge);

      vtend(iedge) += f0 * vt - grav * grad_h;
  });
}

Real LinearShallowWater::compute_energy(Real1d h, Real1d v) const {
  Real1d cell_energy("cell_energy", mesh->ncells);
  
  YAKL_SCOPE(nedges_on_cell, mesh->nedges_on_cell);
  YAKL_SCOPE(edges_on_cell, mesh->edges_on_cell);
  YAKL_SCOPE(dv_edge, mesh->dv_edge);
  YAKL_SCOPE(dc_edge, mesh->dc_edge);
  YAKL_SCOPE(area_cell, mesh->area_cell);
  YAKL_SCOPE(grav, this->grav);
  YAKL_SCOPE(h0, this->h0);

  parallel_for("compute_energy", mesh->ncells, YAKL_LAMBDA (Int icell) {
      Real K = 0;
      for (Int j = 0; j < nedges_on_cell(icell); ++j) {
        Int jedge = edges_on_cell(icell, j);
        Real area_edge = dv_edge(jedge) * dc_edge(jedge);
        K += area_edge * v(jedge) * v(jedge) / 4;
      }
      K /= area_cell(icell);
      cell_energy(icell) = area_cell(icell) * (grav * h(icell) * h(icell) / 2 + h0 * K);
  });
  return yakl::intrinsics::sum(cell_energy);
}

}
