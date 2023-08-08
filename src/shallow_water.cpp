#include <shallow_water.hpp>

namespace omega {

// Base

ShallowWaterBase::ShallowWaterBase(PlanarHexagonalMesh &mesh, Real f0) : mesh(&mesh), f0(f0) {}
ShallowWaterBase::ShallowWaterBase(PlanarHexagonalMesh &mesh, Real f0, Real grav) :
    mesh(&mesh), f0(f0), grav(grav) {}

Real ShallowWaterBase::mass_integral(RealConst1d h) const {
  Real1d cell_mass("cell_mass", mesh->ncells);
  
  YAKL_SCOPE(area_cell, mesh->area_cell);

  parallel_for("compute_cell_mass", mesh->ncells, YAKL_LAMBDA (Int icell) {
      cell_mass(icell) = area_cell(icell) * h(icell);
  });
  return yakl::intrinsics::sum(cell_mass);
}

Real ShallowWaterBase::circulation_integral(RealConst1d v) const {
  Real1d cell_circulation("cell_circulation", mesh->nvertices);
  
  YAKL_SCOPE(dc_edge, mesh->dc_edge);
  YAKL_SCOPE(edges_on_vertex, mesh->edges_on_vertex);
  YAKL_SCOPE(orient_on_vertex, mesh->orient_on_vertex);
  YAKL_SCOPE(area_triangle, mesh->area_triangle);
  YAKL_SCOPE(f0, this->f0);

  parallel_for("compute_cell_circulation", mesh->nvertices, YAKL_LAMBDA (Int ivertex) {
      Real cir_i = -0;
      for (Int j = 0; j < 3; ++j) {
        Int jedge = edges_on_vertex(ivertex, j);
        cir_i += dc_edge(jedge) * orient_on_vertex(ivertex, j) * v(jedge);
      }
      cell_circulation(ivertex) = cir_i + f0 * area_triangle(ivertex);
  });
  return yakl::intrinsics::sum(cell_circulation);
}

// Nonlinear

ShallowWater::ShallowWater(PlanarHexagonalMesh &mesh, Real f0) :
  ShallowWaterBase(mesh, f0), hflux("hflux", mesh.nedges) {}
ShallowWater::ShallowWater(PlanarHexagonalMesh &mesh, Real f0, Real grav) :
  ShallowWaterBase(mesh, f0, grav), hflux("hflux", mesh.nedges) {}

void ShallowWater::compute_h_tendency(Real1d htend, RealConst1d h, RealConst1d v) const {
  YAKL_SCOPE(nedges_on_cell, mesh->nedges_on_cell);
  YAKL_SCOPE(edges_on_cell, mesh->edges_on_cell);
  YAKL_SCOPE(dv_edge, mesh->dv_edge);
  YAKL_SCOPE(cells_on_edge, mesh->cells_on_edge);
  YAKL_SCOPE(orient_on_cell, mesh->orient_on_cell);
  YAKL_SCOPE(area_cell, mesh->area_cell);
  
  YAKL_SCOPE(hflux, this->hflux);
  parallel_for("compute_hflux", mesh->nedges, YAKL_LAMBDA (Int iedge) {
      Real he = -0;
      for (Int j = 0; j < 2; ++j) {
        Int jcell = cells_on_edge(iedge, j);
        he += h(jcell);
      }
      he /= 2;
      hflux(iedge) = he * v(iedge);
  });

  parallel_for("compute_htend", mesh->ncells, YAKL_LAMBDA (Int icell) {
      Real accum = -0;
      for (Int j = 0; j < nedges_on_cell(icell); ++j) {
        Int jedge = edges_on_cell(icell, j);
        accum += dv_edge(jedge) * orient_on_cell(icell, j) * hflux(jedge);
      }
      htend(icell) += -accum / area_cell(icell);
  });
}

void ShallowWater::compute_v_tendency(Real1d vtend, RealConst1d h, RealConst1d v) const {
  YAKL_SCOPE(edges_on_vertex, mesh->edges_on_vertex);
  YAKL_SCOPE(orient_on_vertex, mesh->orient_on_vertex);
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
  
  Real1d qv("qv", mesh->nvertices);
  parallel_for("compute_qv", mesh->nvertices, YAKL_LAMBDA (Int ivertex) {
      Real qv_i = -0;
      for (Int j = 0; j < 3; ++j) {
        Int jedge = edges_on_vertex(ivertex, j);
        qv_i += dc_edge(jedge) * orient_on_vertex(ivertex, j) * v(jedge);
      }
      
      Real hv_i = -0;
      for (Int j = 0; j < 3; ++j) {
        Int jcell = cells_on_vertex(ivertex, j);
        hv_i += kiteareas_on_vertex(ivertex, j) * h(jcell);
      }

      qv(ivertex) = (qv_i + f0 * area_triangle(ivertex)) / hv_i;
  });

  Real1d qe("qe", mesh->nedges);
  parallel_for("compute_qe", mesh->nedges, YAKL_LAMBDA (Int iedge) {
      Real qe_i = -0;
      for (Int j = 0; j < 2; ++j) {
        Int jvertex = vertices_on_edge(iedge, j);
        qe_i += qv(jvertex);
      }
      qe(iedge) = qe_i / 2;
  });
  
  Real1d K("K", mesh->ncells);
  parallel_for("compute_K", mesh->ncells, YAKL_LAMBDA (Int icell) {
      Real K_i = -0;
      for (Int j = 0; j < nedges_on_cell(icell); ++j) {
        Int jedge = edges_on_cell(icell, j);
        Real area_edge = dv_edge(jedge) * dc_edge(jedge);
        K_i += area_edge * v(jedge) * v(jedge) / 4;
      }
      K_i /= area_cell(icell);
      K(icell) = K_i;
  });

  parallel_for("compute_vtend", mesh->nedges, YAKL_LAMBDA (Int iedge) {
      Real qt = -0;
      for (Int j = 0; j < nedges_on_edge(iedge); ++j) {
        Int jedge = edges_on_edge(iedge, j);
        
        qt += weights_on_edge(iedge, j) * dv_edge(jedge) * hflux(jedge) * (qe(iedge) + qe(jedge)) / 2;
      }
      qt /= dc_edge(iedge);

      Int icell0 = cells_on_edge(iedge, 0);
      Int icell1 = cells_on_edge(iedge, 1);

      Real grad_B = (K(icell1) - K(icell0) + grav * (h(icell1) - h(icell0))) / dc_edge(iedge);

      vtend(iedge) += qt - grad_B;
  });
}

Real ShallowWater::energy_integral(RealConst1d h, RealConst1d v) const {
  Real1d cell_energy("cell_energy", mesh->ncells);
  
  YAKL_SCOPE(nedges_on_cell, mesh->nedges_on_cell);
  YAKL_SCOPE(edges_on_cell, mesh->edges_on_cell);
  YAKL_SCOPE(dv_edge, mesh->dv_edge);
  YAKL_SCOPE(dc_edge, mesh->dc_edge);
  YAKL_SCOPE(area_cell, mesh->area_cell);
  YAKL_SCOPE(grav, this->grav);

  parallel_for("compute_cell_energy", mesh->ncells, YAKL_LAMBDA (Int icell) {
      Real K = 0;
      for (Int j = 0; j < nedges_on_cell(icell); ++j) {
        Int jedge = edges_on_cell(icell, j);
        Real area_edge = dv_edge(jedge) * dc_edge(jedge);
        K += area_edge * v(jedge) * v(jedge) / 4;
      }
      K /= area_cell(icell);
      cell_energy(icell) = area_cell(icell) * (grav * h(icell) * h(icell) / 2 + h(icell) * K);
  });
  return yakl::intrinsics::sum(cell_energy);
}


// Linear

LinearShallowWater::LinearShallowWater(PlanarHexagonalMesh &mesh, Real h0, Real f0) :
    ShallowWaterBase(mesh, f0), h0(h0) {}

LinearShallowWater::LinearShallowWater(PlanarHexagonalMesh &mesh, Real h0, Real f0, Real grav) :
   ShallowWaterBase(mesh, f0, grav), h0(h0) {}

void LinearShallowWater::compute_h_tendency(Real1d htend, RealConst1d h, RealConst1d v) const {
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

void LinearShallowWater::compute_v_tendency(Real1d vtend, RealConst1d h, RealConst1d v) const {
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

Real LinearShallowWater::energy_integral(RealConst1d h, RealConst1d v) const {
  Real1d cell_energy("cell_energy", mesh->ncells);
  
  YAKL_SCOPE(nedges_on_cell, mesh->nedges_on_cell);
  YAKL_SCOPE(edges_on_cell, mesh->edges_on_cell);
  YAKL_SCOPE(dv_edge, mesh->dv_edge);
  YAKL_SCOPE(dc_edge, mesh->dc_edge);
  YAKL_SCOPE(area_cell, mesh->area_cell);
  YAKL_SCOPE(grav, this->grav);
  YAKL_SCOPE(h0, this->h0);

  parallel_for("compute_cell_energy", mesh->ncells, YAKL_LAMBDA (Int icell) {
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
