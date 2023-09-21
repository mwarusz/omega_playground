#pragma once

#include <common.hpp>

namespace omega {

struct PlanarHexagonalMesh {
  static constexpr Int maxedges = 6;

  Int m_nx;
  Int m_ny;
  Int m_nlayers;
  Int m_ncells;
  Int m_nedges;
  Int m_nvertices;
  Real m_dc;
  Real m_period_x;
  Real m_period_y;

  Int1d m_nedges_on_cell;
  Int1d m_max_level_cell;
  Int2d m_cells_on_cell;
  Int2d m_edges_on_cell;
  Int2d m_vertices_on_cell;
  Int2d m_edge_sign_on_cell;
  Int2d m_kite_index_on_cell;

  Real1d m_area_cell;
  Real1d m_lat_cell;
  Real1d m_lon_cell;
  Real1d m_x_cell;
  Real1d m_y_cell;
  Real1d m_z_cell;
  Real1d m_mesh_density;

  Int1d m_nedges_on_edge;
  Int1d m_max_level_edge_bot;
  Int1d m_max_level_edge_top;
  Int2d m_cells_on_edge;
  Int2d m_vertices_on_edge;
  Int2d m_edges_on_edge;

  Real1d m_dc_edge;
  Real1d m_dv_edge;
  Real1d m_angle_edge;
  Real1d m_lat_edge;
  Real1d m_lon_edge;
  Real1d m_x_edge;
  Real1d m_y_edge;
  Real1d m_z_edge;
  Real2d m_weights_on_edge;

  Int1d m_max_level_vertex_bot;
  Int1d m_max_level_vertex_top;
  Int2d m_edges_on_vertex;
  Int2d m_cells_on_vertex;
  Int2d m_edge_sign_on_vertex;

  Real1d m_area_triangle;
  Real1d m_lat_vertex;
  Real1d m_lon_vertex;
  Real1d m_x_vertex;
  Real1d m_y_vertex;
  Real1d m_z_vertex;
  Real2d m_kiteareas_on_vertex;

  PlanarHexagonalMesh(Int nx, Int ny, Int nlayers = 1);
  PlanarHexagonalMesh(Int nx, Int ny, Real dc, Int nlayers = 1);
  YAKL_INLINE Int cellidx(Int icol, Int irow) const;
  YAKL_INLINE Int cell_on_cell(Int icol, Int irow, Int nb) const;
  YAKL_INLINE Int edge_on_cell(Int icell, Int icol, Int irow, Int nb) const;
  YAKL_INLINE Int vertex_on_cell(Int icell, Int icol, Int irow, Int nb) const;

  void compute_mesh_arrays();
};
} // namespace omega
