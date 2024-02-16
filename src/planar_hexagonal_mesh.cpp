#include <planar_hexagonal_mesh.hpp>

namespace omega {

PlanarHexagonalMesh::PlanarHexagonalMesh(Int nx, Int ny, Int nlayers)
    : PlanarHexagonalMesh(nx, ny, 1. / nx, nlayers) {}

PlanarHexagonalMesh::PlanarHexagonalMesh(Int nx, Int ny, Real dc, Int nlayers)
    : m_nx(nx), m_ny(ny), m_dc(dc) {

  m_nlayers = std::ceil(Real(nlayers) / vector_length) * vector_length;
  m_ncells = nx * ny;
  m_nedges = 3 * m_ncells;
  m_nvertices = 2 * m_ncells;
  m_period_x = nx * dc;
  m_period_y = ny * dc * sqrt(3) / 2;

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

  compute_mesh_arrays();
}

KOKKOS_INLINE_FUNCTION Int PlanarHexagonalMesh::cellidx(Int icol,
                                                        Int irow) const {
  return irow * m_nx + icol;
}

KOKKOS_INLINE_FUNCTION Int PlanarHexagonalMesh::cell_on_cell(Int icol, Int irow,
                                                             Int nb) const {
  Int mx = icol == 0 ? m_nx - 1 : icol - 1;
  Int px = icol == (m_nx - 1) ? 0 : icol + 1;

  Int my = irow == 0 ? m_ny - 1 : irow - 1;
  Int py = irow == (m_ny - 1) ? 0 : irow + 1;

  if (irow % 2 == 0) {
    if (nb == 0) {
      return cellidx(mx, irow);
    }
    if (nb == 1) {
      return cellidx(mx, my);
    }
    if (nb == 2) {
      return cellidx(icol, my);
    }
    if (nb == 3) {
      return cellidx(px, irow);
    }
    if (nb == 4) {
      return cellidx(icol, py);
    }
    if (nb == 5) {
      return cellidx(mx, py);
    }
  } else {
    if (nb == 0) {
      return cellidx(mx, irow);
    }
    if (nb == 1) {
      return cellidx(icol, my);
    }
    if (nb == 2) {
      return cellidx(px, my);
    }
    if (nb == 3) {
      return cellidx(px, irow);
    }
    if (nb == 4) {
      return cellidx(px, py);
    }
    if (nb == 5) {
      return cellidx(icol, py);
    }
  }
  return -1;
}

KOKKOS_INLINE_FUNCTION Int PlanarHexagonalMesh::edge_on_cell(Int icell,
                                                             Int icol, Int irow,
                                                             Int nb) const {
  if (nb == 0) {
    return 3 * icell;
  }
  if (nb == 1) {
    return 3 * icell + 1;
  }
  if (nb == 2) {
    return 3 * icell + 2;
  }
  if (nb == 3) {
    return 3 * cell_on_cell(icol, irow, nb);
  }
  if (nb == 4) {
    return 3 * cell_on_cell(icol, irow, nb) + 1;
  }
  if (nb == 5) {
    return 3 * cell_on_cell(icol, irow, nb) + 2;
  }
  return -1;
}

KOKKOS_INLINE_FUNCTION Int PlanarHexagonalMesh::vertex_on_cell(Int icell,
                                                               Int icol,
                                                               Int irow,
                                                               Int nb) const {
  if (nb == 0) {
    return 2 * icell;
  }
  if (nb == 1) {
    return 2 * icell + 1;
  }
  if (nb == 2) {
    return 2 * cell_on_cell(icol, irow, 2);
  }
  if (nb == 3) {
    return 2 * cell_on_cell(icol, irow, 3) + 1;
  }
  if (nb == 4) {
    return 2 * cell_on_cell(icol, irow, 3);
  }
  if (nb == 5) {
    return 2 * cell_on_cell(icol, irow, 4) + 1;
  }
  return -1;
}

void PlanarHexagonalMesh::compute_mesh_arrays() {
  parallel_for(
      "compute_mesh_arrays", MDRangePolicy<2>({0, 0}, {m_ny, m_nx}),
      KOKKOS_CLASS_LAMBDA(Int irow, Int icol) {
        Int icell = cellidx(icol, irow);
        m_nedges_on_cell(icell) = 6;

        for (Int j = 0; j < maxedges; ++j) {
          m_cells_on_cell(icell, j) = cell_on_cell(icol, irow, j);
          m_edges_on_cell(icell, j) = edge_on_cell(icell, icol, irow, j);
          m_vertices_on_cell(icell, j) = vertex_on_cell(icell, icol, irow, j);
        }

        for (Int j = 0; j < 3; ++j) {
          m_cells_on_edge(m_edges_on_cell(icell, j), 1) = icell;
        }
        for (Int j = 3; j < maxedges; ++j) {
          m_cells_on_edge(m_edges_on_cell(icell, j), 0) = icell;
        }
        m_vertices_on_edge(m_edges_on_cell(icell, 0), 0) =
            m_vertices_on_cell(icell, 1);
        m_vertices_on_edge(m_edges_on_cell(icell, 0), 1) =
            m_vertices_on_cell(icell, 0);
        m_vertices_on_edge(m_edges_on_cell(icell, 1), 0) =
            m_vertices_on_cell(icell, 2);
        m_vertices_on_edge(m_edges_on_cell(icell, 1), 1) =
            m_vertices_on_cell(icell, 1);
        m_vertices_on_edge(m_edges_on_cell(icell, 2), 0) =
            m_vertices_on_cell(icell, 3);
        m_vertices_on_edge(m_edges_on_cell(icell, 2), 1) =
            m_vertices_on_cell(icell, 2);

        m_edges_on_edge(m_edges_on_cell(icell, 3), 0) =
            m_edges_on_cell(icell, 4);
        m_edges_on_edge(m_edges_on_cell(icell, 3), 1) =
            m_edges_on_cell(icell, 5);
        m_edges_on_edge(m_edges_on_cell(icell, 3), 2) =
            m_edges_on_cell(icell, 0);
        m_edges_on_edge(m_edges_on_cell(icell, 3), 3) =
            m_edges_on_cell(icell, 1);
        m_edges_on_edge(m_edges_on_cell(icell, 3), 4) =
            m_edges_on_cell(icell, 2);

        m_edges_on_edge(m_edges_on_cell(icell, 4), 0) =
            m_edges_on_cell(icell, 5);
        m_edges_on_edge(m_edges_on_cell(icell, 4), 1) =
            m_edges_on_cell(icell, 0);
        m_edges_on_edge(m_edges_on_cell(icell, 4), 2) =
            m_edges_on_cell(icell, 1);
        m_edges_on_edge(m_edges_on_cell(icell, 4), 3) =
            m_edges_on_cell(icell, 2);
        m_edges_on_edge(m_edges_on_cell(icell, 4), 4) =
            m_edges_on_cell(icell, 3);

        m_edges_on_edge(m_edges_on_cell(icell, 5), 0) =
            m_edges_on_cell(icell, 0);
        m_edges_on_edge(m_edges_on_cell(icell, 5), 1) =
            m_edges_on_cell(icell, 1);
        m_edges_on_edge(m_edges_on_cell(icell, 5), 2) =
            m_edges_on_cell(icell, 2);
        m_edges_on_edge(m_edges_on_cell(icell, 5), 3) =
            m_edges_on_cell(icell, 3);
        m_edges_on_edge(m_edges_on_cell(icell, 5), 4) =
            m_edges_on_cell(icell, 4);

        m_edges_on_edge(m_edges_on_cell(icell, 0), 5) =
            m_edges_on_cell(icell, 1);
        m_edges_on_edge(m_edges_on_cell(icell, 0), 6) =
            m_edges_on_cell(icell, 2);
        m_edges_on_edge(m_edges_on_cell(icell, 0), 7) =
            m_edges_on_cell(icell, 3);
        m_edges_on_edge(m_edges_on_cell(icell, 0), 8) =
            m_edges_on_cell(icell, 4);
        m_edges_on_edge(m_edges_on_cell(icell, 0), 9) =
            m_edges_on_cell(icell, 5);

        m_edges_on_edge(m_edges_on_cell(icell, 1), 5) =
            m_edges_on_cell(icell, 2);
        m_edges_on_edge(m_edges_on_cell(icell, 1), 6) =
            m_edges_on_cell(icell, 3);
        m_edges_on_edge(m_edges_on_cell(icell, 1), 7) =
            m_edges_on_cell(icell, 4);
        m_edges_on_edge(m_edges_on_cell(icell, 1), 8) =
            m_edges_on_cell(icell, 5);
        m_edges_on_edge(m_edges_on_cell(icell, 1), 9) =
            m_edges_on_cell(icell, 0);

        m_edges_on_edge(m_edges_on_cell(icell, 2), 5) =
            m_edges_on_cell(icell, 3);
        m_edges_on_edge(m_edges_on_cell(icell, 2), 6) =
            m_edges_on_cell(icell, 4);
        m_edges_on_edge(m_edges_on_cell(icell, 2), 7) =
            m_edges_on_cell(icell, 5);
        m_edges_on_edge(m_edges_on_cell(icell, 2), 8) =
            m_edges_on_cell(icell, 0);
        m_edges_on_edge(m_edges_on_cell(icell, 2), 9) =
            m_edges_on_cell(icell, 1);

        m_weights_on_edge(m_edges_on_cell(icell, 3), 0) = 1. / 3;
        m_weights_on_edge(m_edges_on_cell(icell, 3), 1) = 1. / 6;
        m_weights_on_edge(m_edges_on_cell(icell, 3), 2) = 0;
        m_weights_on_edge(m_edges_on_cell(icell, 3), 3) = 1. / 6;
        m_weights_on_edge(m_edges_on_cell(icell, 3), 4) = 1. / 3;

        m_weights_on_edge(m_edges_on_cell(icell, 4), 0) = 1. / 3;
        m_weights_on_edge(m_edges_on_cell(icell, 4), 1) = -1. / 6;
        m_weights_on_edge(m_edges_on_cell(icell, 4), 2) = 0;
        m_weights_on_edge(m_edges_on_cell(icell, 4), 3) = 1. / 6;
        m_weights_on_edge(m_edges_on_cell(icell, 4), 4) = -1. / 3;

        m_weights_on_edge(m_edges_on_cell(icell, 5), 0) = -1. / 3;
        m_weights_on_edge(m_edges_on_cell(icell, 5), 1) = -1. / 6;
        m_weights_on_edge(m_edges_on_cell(icell, 5), 2) = 0;
        m_weights_on_edge(m_edges_on_cell(icell, 5), 3) = -1. / 6;
        m_weights_on_edge(m_edges_on_cell(icell, 5), 4) = -1. / 3;

        m_weights_on_edge(m_edges_on_cell(icell, 0), 5) = 1. / 3;
        m_weights_on_edge(m_edges_on_cell(icell, 0), 6) = 1. / 6;
        m_weights_on_edge(m_edges_on_cell(icell, 0), 7) = 0;
        m_weights_on_edge(m_edges_on_cell(icell, 0), 8) = 1. / 6;
        m_weights_on_edge(m_edges_on_cell(icell, 0), 9) = 1. / 3;

        m_weights_on_edge(m_edges_on_cell(icell, 1), 5) = 1. / 3;
        m_weights_on_edge(m_edges_on_cell(icell, 1), 6) = -1. / 6;
        m_weights_on_edge(m_edges_on_cell(icell, 1), 7) = 0;
        m_weights_on_edge(m_edges_on_cell(icell, 1), 8) = 1. / 6;
        m_weights_on_edge(m_edges_on_cell(icell, 1), 9) = -1. / 3;

        m_weights_on_edge(m_edges_on_cell(icell, 2), 5) = -1. / 3;
        m_weights_on_edge(m_edges_on_cell(icell, 2), 6) = -1. / 6;
        m_weights_on_edge(m_edges_on_cell(icell, 2), 7) = 0;
        m_weights_on_edge(m_edges_on_cell(icell, 2), 8) = -1. / 6;
        m_weights_on_edge(m_edges_on_cell(icell, 2), 9) = -1. / 3;

        m_cells_on_vertex(m_vertices_on_cell(icell, 1), 2) = icell;
        m_cells_on_vertex(m_vertices_on_cell(icell, 3), 0) = icell;
        m_cells_on_vertex(m_vertices_on_cell(icell, 5), 1) = icell;
        m_cells_on_vertex(m_vertices_on_cell(icell, 0), 0) = icell;
        m_cells_on_vertex(m_vertices_on_cell(icell, 2), 1) = icell;
        m_cells_on_vertex(m_vertices_on_cell(icell, 4), 2) = icell;

        m_edges_on_vertex(m_vertices_on_cell(icell, 0), 0) =
            m_edges_on_cell(icell, 0);
        m_edges_on_vertex(m_vertices_on_cell(icell, 1), 0) =
            m_edges_on_cell(icell, 0);
        m_edges_on_vertex(m_vertices_on_cell(icell, 2), 2) =
            m_edges_on_cell(icell, 1);
        m_edges_on_vertex(m_vertices_on_cell(icell, 1), 2) =
            m_edges_on_cell(icell, 1);
        m_edges_on_vertex(m_vertices_on_cell(icell, 2), 1) =
            m_edges_on_cell(icell, 2);
        m_edges_on_vertex(m_vertices_on_cell(icell, 3), 1) =
            m_edges_on_cell(icell, 2);
      });

  parallel_for(
      "scale_weights", MDRangePolicy<2>({0, 0}, {m_nedges, 2 * maxedges}),
      KOKKOS_CLASS_LAMBDA(Int iedge, Int j) {
        m_weights_on_edge(iedge, j) *= 1. / sqrt(3);
      });

  parallel_for(
      "compute_cell_arrays", MDRangePolicy<2>({0, 0}, {m_ny, m_nx}),
      KOKKOS_CLASS_LAMBDA(Int irow, Int icol) {
        Int icell = cellidx(icol, irow);
        m_area_cell(icell) = m_dc * m_dc * sqrt(3) / 2;
        m_lat_cell(icell) = 0;
        m_lon_cell(icell) = 0;

        if (irow % 2 == 0) {
          m_x_cell(icell) = m_dc * icol + m_dc / 2;
          m_y_cell(icell) = m_dc * (irow + 1) * sqrt(3) / 2;
          m_z_cell(icell) = 0;
        } else {
          m_x_cell(icell) = m_dc * (icol + 1);
          m_y_cell(icell) = m_dc * (irow + 1) * sqrt(3) / 2;
          m_z_cell(icell) = 0;
        }

        m_x_edge(m_edges_on_cell(icell, 0)) = m_x_cell(icell) - m_dc / 2;
        m_y_edge(m_edges_on_cell(icell, 0)) = m_y_cell(icell);
        m_z_edge(m_edges_on_cell(icell, 0)) = 0;

        m_x_edge(m_edges_on_cell(icell, 1)) =
            m_x_cell(icell) - m_dc / 2 * cos(pi / 3);
        m_y_edge(m_edges_on_cell(icell, 1)) =
            m_y_cell(icell) - m_dc / 2 * sin(pi / 3);
        m_z_edge(m_edges_on_cell(icell, 1)) = 0;

        m_x_edge(m_edges_on_cell(icell, 2)) =
            m_x_cell(icell) + m_dc / 2 * cos(pi / 3);
        m_y_edge(m_edges_on_cell(icell, 2)) =
            m_y_cell(icell) - m_dc / 2 * sin(pi / 3);
        m_z_edge(m_edges_on_cell(icell, 2)) = 0;

        m_angle_edge(m_edges_on_cell(icell, 0)) = 0;
        m_angle_edge(m_edges_on_cell(icell, 1)) = pi / 3;
        m_angle_edge(m_edges_on_cell(icell, 2)) = 2 * pi / 3;

        m_x_vertex(m_vertices_on_cell(icell, 0)) = m_x_cell(icell) - m_dc / 2;
        m_y_vertex(m_vertices_on_cell(icell, 0)) =
            m_y_cell(icell) + m_dc * sqrt(3) / 6;
        m_z_vertex(m_vertices_on_cell(icell, 0)) = 0;

        m_x_vertex(m_vertices_on_cell(icell, 1)) = m_x_cell(icell) - m_dc / 2;
        m_y_vertex(m_vertices_on_cell(icell, 1)) =
            m_y_cell(icell) - m_dc * sqrt(3) / 6;
        m_z_vertex(m_vertices_on_cell(icell, 1)) = 0;
      });

  parallel_for(
      "compute_edge_arrays", RangePolicy(0, m_nedges),
      KOKKOS_CLASS_LAMBDA(Int iedge) {
        m_nedges_on_edge(iedge) = 10;
        m_dc_edge(iedge) = m_dc;
        m_dv_edge(iedge) = m_dc_edge(iedge) * sqrt(3) / 3;
        m_lat_edge(iedge) = 0;
        m_lon_edge(iedge) = 0;
      });

  parallel_for(
      "compute_vertex_arrays", RangePolicy(0, m_nvertices),
      KOKKOS_CLASS_LAMBDA(Int ivertex) {
        m_lat_vertex(ivertex) = 0;
        m_lon_vertex(ivertex) = 0;
        m_area_triangle(ivertex) = m_dc * m_dc * sqrt(3) / 4;
        for (Int j = 0; j < 3; ++j) {
          m_kiteareas_on_vertex(ivertex, j) = m_dc * m_dc * sqrt(3) / 12;
        }
      });

  finalize_mesh();
}
} // namespace omega
