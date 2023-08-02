#pragma once

#include <common.hpp>

namespace omega {

struct PlanarHexagonalMesh {
  static constexpr Int maxedges = 6;

  Int nx;
  Int ny;
  Int ncells;
  Int nedges;
  Int nvertices;
  Real dc;
  Real period_x;
  Real period_y;

  Int1d nedges_on_cell;
  Int2d cells_on_cell;
  Int2d edges_on_cell;
  Int2d vertices_on_cell;
  Int2d orient_on_cell;
  
  Real1d area_cell;
  Real1d lat_cell;
  Real1d lon_cell;
  Real1d x_cell;
  Real1d y_cell;
  Real1d z_cell;
  Real1d mesh_density;

  Int1d nedges_on_edge;
  Int2d cells_on_edge;
  Int2d vertices_on_edge;
  Int2d edges_on_edge;
  
  Real1d dc_edge;
  Real1d dv_edge;
  Real1d angle_edge;
  Real1d lat_edge;
  Real1d lon_edge;
  Real1d x_edge;
  Real1d y_edge;
  Real1d z_edge;
  Real2d weights_on_edge;
  
  Int2d edges_on_vertex;
  Int2d cells_on_vertex;
  Int2d orient_on_vertex;
  
  Real1d area_triangle;
  Real1d lat_vertex;
  Real1d lon_vertex;
  Real1d x_vertex;
  Real1d y_vertex;
  Real1d z_vertex;
  Real2d kiteareas_on_vertex;
  
  PlanarHexagonalMesh(Int nx, Int ny);
  PlanarHexagonalMesh(Int nx, Int ny, Real dc);
  YAKL_INLINE Int cellidx(Int icol, Int irow) const;
  YAKL_INLINE Int cell_on_cell(Int icol, Int irow, Int nb) const;
  YAKL_INLINE Int edge_on_cell(Int icell, Int icol, Int irow, Int nb) const;
  YAKL_INLINE Int vertex_on_cell(Int icell, Int icol, Int irow, Int nb) const;
  void compute_mesh_arrays();
};
}
