#include "mpas_mesh.hpp"

namespace omega {

MPASMesh::MPASMesh(Int ncells, Int nedges, Int nvertices, Int nlayers) {
  m_ncells = ncells;
  m_nedges = nedges;
  m_nvertices = nvertices;
  m_nlayers = std::ceil(Real(nlayers) / vector_length) * vector_length;
  m_nlayers_vec = std::ceil(Real(nlayers) / vector_length);

  // cell properties
  m_nedges_on_cell = Int1d("nedges_on_cell", m_ncells);
  m_cells_on_cell = Int2d("cells_on_cell", m_ncells, maxedges);
  m_edges_on_cell = Int2d("edges_on_cell", m_ncells, maxedges);
  m_vertices_on_cell = Int2d("vertices_on_cell", m_ncells, maxedges);

  m_area_cell = Real1d("area_cell", m_ncells);
  m_lat_cell = Real1d("lat_cell", m_ncells);
  m_lon_cell = Real1d("lon_cell", m_ncells);
  m_x_cell = Real1d("x_cell", m_ncells);
  m_y_cell = Real1d("y_cell", m_ncells);
  m_z_cell = Real1d("z_cell", m_ncells);
  m_mesh_density = Real1d("mesh_density", m_ncells);

  // edge properties
  m_nedges_on_edge = Int1d("nedges_on_edge", m_nedges);
  m_cells_on_edge = Int2d("cells_on_edge", m_nedges, 2);
  m_vertices_on_edge = Int2d("vertices_on_edge", m_nedges, 2);
  m_edges_on_edge = Int2d("edges_on_edge", m_nedges, 2 * maxedges);

  m_dc_edge = Real1d("dc_edge", m_nedges);
  m_dv_edge = Real1d("dv_edge", m_nedges);
  m_angle_edge = Real1d("angle_edge", m_nedges);
  m_lat_edge = Real1d("lat_edge", m_nedges);
  m_lon_edge = Real1d("lon_edge", m_nedges);
  m_x_edge = Real1d("x_edge", m_nedges);
  m_y_edge = Real1d("y_edge", m_nedges);
  m_z_edge = Real1d("z_edge", m_nedges);
  m_weights_on_edge = Real2d("weights_on_edge", m_nedges, 2 * maxedges);

  // vertex properties
  m_edges_on_vertex = Int2d("edges_on_vertex", m_nvertices, 3);
  m_cells_on_vertex = Int2d("cells_on_vertex", m_nvertices, 3);

  m_area_triangle = Real1d("area_triangle", m_nvertices);
  m_lat_vertex = Real1d("lat_vertex", m_nvertices);
  m_lon_vertex = Real1d("lon_vertex", m_nvertices);
  m_x_vertex = Real1d("x_vertex", m_nvertices);
  m_y_vertex = Real1d("y_vertex", m_nvertices);
  m_z_vertex = Real1d("z_vertex", m_nvertices);
  m_kiteareas_on_vertex = Real2d("kiteareas_on_vertex", m_nvertices, 3);
}

void MPASMesh::finalize_mesh() {

  m_max_level_cell = Int1d("max_level_cell", m_ncells);
  m_edge_sign_on_cell = Real2d("edge_sign_on_cell", m_ncells, maxedges);
  m_kite_index_on_cell = Int2d("kite_index_on_cell", m_ncells, maxedges);

  m_max_level_edge_bot = Int1d("max_level_edge_bot", m_nedges);
  m_max_level_edge_top = Int1d("max_level_edge_top", m_nedges);
  m_edge_mask = Real2d("edge_mask", m_nedges, m_nlayers);
  m_mesh_scaling_del2 = Real1d("mesh_scaling_del2", m_nedges);
  m_mesh_scaling_del4 = Real1d("mesh_scaling_del4", m_nedges);

  m_max_level_vertex_bot = Int1d("max_level_vertex_bot", m_nvertices);
  m_max_level_vertex_top = Int1d("max_level_vertex_top", m_nvertices);
  m_edge_sign_on_vertex = Real2d("edge_sign_on_vertex", m_nvertices, 3);

  omega_parallel_for(
      "finalize_cell", {m_ncells}, KOKKOS_CLASS_LAMBDA(Int icell) {
        for (Int j = 0; j < m_nedges_on_cell(icell); ++j) {
          m_edge_sign_on_cell(icell, j) =
              m_cells_on_edge(m_edges_on_cell(icell, j), 0) == icell ? 1 : -1;
        }

        for (Int j = 0; j < m_nedges_on_cell(icell); ++j) {
          Int jvertex = m_vertices_on_cell(icell, j);
          for (Int l = 0; l < 3; ++l) {
            if (m_cells_on_vertex(jvertex, l) == icell) {
              m_kite_index_on_cell(icell, j) = l;
            }
          }
        }
      });

  omega_parallel_for(
      "finalize_vertex", {m_nvertices}, KOKKOS_CLASS_LAMBDA(Int ivertex) {
        for (Int j = 0; j < 3; ++j) {
          m_edge_sign_on_vertex(ivertex, j) =
              m_vertices_on_edge(m_edges_on_vertex(ivertex, j), 0) == ivertex
                  ? -1
                  : 1;
        }
      });

  deep_copy(m_mesh_scaling_del2, 1);
  deep_copy(m_mesh_scaling_del4, 1);
  deep_copy(m_edge_mask, 1);

  deep_copy(m_max_level_cell, m_nlayers);
  deep_copy(m_max_level_edge_bot, m_nlayers);
  deep_copy(m_max_level_edge_top, m_nlayers);
  deep_copy(m_max_level_vertex_bot, m_nlayers);
  deep_copy(m_max_level_vertex_top, m_nlayers);
}

} // namespace omega
