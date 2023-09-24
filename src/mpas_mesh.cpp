#include <mpas_mesh.hpp>

namespace omega {

void MPASMesh::finalize_mesh() {

  m_max_level_cell = Int1d("max_level_cell", m_ncells);
  m_edge_sign_on_cell = Int2d("edge_sign_on_cell", m_ncells, maxedges);
  m_kite_index_on_cell = Int2d("kite_index_on_cell", m_ncells, maxedges);

  m_max_level_edge_bot = Int1d("max_level_edge_bot", m_nedges);
  m_max_level_edge_top = Int1d("max_level_edge_top", m_nedges);
  m_edge_mask = Real2d("edge_mask", m_nedges, m_nlayers);
  m_mesh_scaling_del2 = Real1d("mesh_scaling_del2", m_nedges);
  m_mesh_scaling_del4 = Real1d("mesh_scaling_del4", m_nedges);

  m_max_level_vertex_bot = Int1d("max_level_vertex_bot", m_nvertices);
  m_max_level_vertex_top = Int1d("max_level_vertex_top", m_nvertices);
  m_edge_sign_on_vertex = Int2d("edge_sign_on_vertex", m_nvertices, 3);

  parallel_for(
      "finalize_cell", m_ncells, YAKL_CLASS_LAMBDA(Int icell) {
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

  parallel_for(
      "finalize_vertex", m_nvertices, YAKL_CLASS_LAMBDA(Int ivertex) {
        for (Int j = 0; j < 3; ++j) {
          m_edge_sign_on_vertex(ivertex, j) =
              m_vertices_on_edge(m_edges_on_vertex(ivertex, j), 0) == ivertex
                  ? -1
                  : 1;
        }
      });

  yakl::memset(m_mesh_scaling_del2, 1);
  yakl::memset(m_mesh_scaling_del4, 1);
  yakl::memset(m_edge_mask, 1);

  yakl::memset(m_max_level_cell, m_nlayers);
  yakl::memset(m_max_level_edge_bot, m_nlayers);
  yakl::memset(m_max_level_edge_top, m_nlayers);
  yakl::memset(m_max_level_vertex_bot, m_nlayers);
  yakl::memset(m_max_level_vertex_top, m_nlayers);
}

} // namespace omega
