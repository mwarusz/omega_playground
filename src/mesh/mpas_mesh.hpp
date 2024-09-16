#pragma once

#include <common.hpp>

namespace omega {

struct MPASMesh {
  static constexpr Int maxedges = 6;

  Int m_nlayers;
  Int m_nlayers_vec;
  Int m_ncells;
  Int m_nedges;
  Int m_nvertices;

  Int1d m_nedges_on_cell;
  Int2d m_edges_on_cell;
  Int2d m_cells_on_cell;
  Int2d m_vertices_on_cell;

  Int2d m_kite_index_on_cell;
  Int1d m_max_level_cell;

  Real1d m_area_cell;
  Real1d m_lat_cell;
  Real1d m_lon_cell;
  Real1d m_x_cell;
  Real1d m_y_cell;
  Real1d m_z_cell;
  Real1d m_mesh_density;
  Real2d m_edge_sign_on_cell;

  Int1d m_nedges_on_edge;
  Int2d m_edges_on_edge;
  Int2d m_cells_on_edge;
  Int2d m_vertices_on_edge;

  Int1d m_max_level_edge_bot;
  Int1d m_max_level_edge_top;

  Real1d m_dc_edge;
  Real1d m_dv_edge;
  Real1d m_angle_edge;
  Real1d m_lat_edge;
  Real1d m_lon_edge;
  Real1d m_x_edge;
  Real1d m_y_edge;
  Real1d m_z_edge;
  Real2d m_weights_on_edge;

  Real1d m_mesh_scaling_del2;
  Real1d m_mesh_scaling_del4;
  Real2d m_edge_mask;

  Int2d m_edges_on_vertex;
  Int2d m_cells_on_vertex;

  Int1d m_max_level_vertex_bot;
  Int1d m_max_level_vertex_top;

  Real1d m_area_triangle;
  Real1d m_lat_vertex;
  Real1d m_lon_vertex;
  Real1d m_x_vertex;
  Real1d m_y_vertex;
  Real1d m_z_vertex;
  Real2d m_kiteareas_on_vertex;
  Real2d m_edge_sign_on_vertex;

  MPASMesh(Int ncells, Int nedges, Int nvertices, Int nlayers);
  void finalize_mesh();
};

} // namespace omega
