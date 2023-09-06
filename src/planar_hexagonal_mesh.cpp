#include <planar_hexagonal_mesh.hpp>

namespace omega {

PlanarHexagonalMesh::PlanarHexagonalMesh(Int nx, Int ny, Int nlayers)
    : PlanarHexagonalMesh(nx, ny, 1. / nx, nlayers) {}

PlanarHexagonalMesh::PlanarHexagonalMesh(Int nx, Int ny, Real dc, Int nlayers)
    : nx(nx), ny(ny), dc(dc), nlayers(nlayers) {

  this->ncells = nx * ny;
  this->nedges = 3 * ncells;
  this->nvertices = 2 * ncells;
  this->period_x = nx * dc;
  this->period_y = ny * dc * sqrt(3) / 2;

  // cell properties
  this->nedges_on_cell = Int1d("nedges_on_cell", ncells);
  this->max_level_cell = Int1d("max_level_cell", ncells);
  this->cells_on_cell = Int2d("cells_on_cell", ncells, maxedges);
  this->edges_on_cell = Int2d("edges_on_cell", ncells, maxedges);
  this->vertices_on_cell = Int2d("vertices_on_cell", ncells, maxedges);
  this->edge_sign_on_cell = Int2d("edge_sign_on_cell", ncells, maxedges);
  this->kite_index_on_cell = Int2d("kite_index_on_cell", ncells, maxedges);

  this->area_cell = Real1d("area_cell", ncells);
  this->lat_cell = Real1d("lat_cell", ncells);
  this->lon_cell = Real1d("lon_cell", ncells);
  this->x_cell = Real1d("x_cell", ncells);
  this->y_cell = Real1d("y_cell", ncells);
  this->z_cell = Real1d("z_cell", ncells);
  this->mesh_density = Real1d("mesh_density", ncells);

  // edge properties
  this->nedges_on_edge = Int1d("nedges_on_edge", nedges);
  this->max_level_edge_bot = Int1d("max_level_edge_bot", nedges);
  this->max_level_edge_top = Int1d("max_level_edge_top", nedges);
  this->cells_on_edge = Int2d("cells_on_edge", nedges, 2);
  this->vertices_on_edge = Int2d("vertices_on_edge", nedges, 2);
  this->edges_on_edge = Int2d("edges_on_edge", nedges, 2 * maxedges);

  this->dc_edge = Real1d("dc_edge", nedges);
  this->dv_edge = Real1d("dv_edge", nedges);
  this->angle_edge = Real1d("angle_edge", nedges);
  this->lat_edge = Real1d("lat_edge", nedges);
  this->lon_edge = Real1d("lon_edge", nedges);
  this->x_edge = Real1d("x_edge", nedges);
  this->y_edge = Real1d("y_edge", nedges);
  this->z_edge = Real1d("z_edge", nedges);
  this->weights_on_edge = Real2d("weights_on_edge", nedges, 2 * maxedges);

  // vertex properties
  this->max_level_vertex_bot = Int1d("max_level_vertex_bot", nvertices);
  this->max_level_vertex_top = Int1d("max_level_vertex_top", nvertices);
  this->edges_on_vertex = Int2d("edges_on_vertex", nvertices, 3);
  this->cells_on_vertex = Int2d("cells_on_vertex", nvertices, 3);
  this->edge_sign_on_vertex = Int2d("edge_sign_on_vertex", nvertices, 3);

  this->area_triangle = Real1d("area_triangle", nvertices);
  this->lat_vertex = Real1d("lat_vertex", nvertices);
  this->lon_vertex = Real1d("lon_vertex", nvertices);
  this->x_vertex = Real1d("x_vertex", nvertices);
  this->y_vertex = Real1d("y_vertex", nvertices);
  this->z_vertex = Real1d("z_vertex", nvertices);
  this->kiteareas_on_vertex = Real2d("kiteareas_on_vertex", nvertices, 3);

  this->compute_mesh_arrays();
}

YAKL_INLINE Int PlanarHexagonalMesh::cellidx(Int icol, Int irow) const {
  return irow * nx + icol;
}

YAKL_INLINE Int PlanarHexagonalMesh::cell_on_cell(Int icol, Int irow,
                                                  Int nb) const {
  Int mx = icol == 0 ? nx - 1 : icol - 1;
  Int px = icol == (nx - 1) ? 0 : icol + 1;

  Int my = irow == 0 ? ny - 1 : irow - 1;
  Int py = irow == (ny - 1) ? 0 : irow + 1;

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

YAKL_INLINE Int PlanarHexagonalMesh::edge_on_cell(Int icell, Int icol, Int irow,
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

YAKL_INLINE Int PlanarHexagonalMesh::vertex_on_cell(Int icell, Int icol,
                                                    Int irow, Int nb) const {
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
      "compute_mesh_arrays", SimpleBounds<2>(ny, nx),
      YAKL_CLASS_LAMBDA(Int irow, Int icol) {
        Int icell = cellidx(icol, irow);
        nedges_on_cell(icell) = 6;

        for (Int j = 0; j < maxedges; ++j) {
          cells_on_cell(icell, j) = cell_on_cell(icol, irow, j);
          edges_on_cell(icell, j) = edge_on_cell(icell, icol, irow, j);
          vertices_on_cell(icell, j) = vertex_on_cell(icell, icol, irow, j);
        }

        for (Int j = 0; j < 3; ++j) {
          cells_on_edge(edges_on_cell(icell, j), 1) = icell;
        }
        for (Int j = 3; j < maxedges; ++j) {
          cells_on_edge(edges_on_cell(icell, j), 0) = icell;
        }
        vertices_on_edge(edges_on_cell(icell, 0), 0) =
            vertices_on_cell(icell, 1);
        vertices_on_edge(edges_on_cell(icell, 0), 1) =
            vertices_on_cell(icell, 0);
        vertices_on_edge(edges_on_cell(icell, 1), 0) =
            vertices_on_cell(icell, 2);
        vertices_on_edge(edges_on_cell(icell, 1), 1) =
            vertices_on_cell(icell, 1);
        vertices_on_edge(edges_on_cell(icell, 2), 0) =
            vertices_on_cell(icell, 3);
        vertices_on_edge(edges_on_cell(icell, 2), 1) =
            vertices_on_cell(icell, 2);

        edges_on_edge(edges_on_cell(icell, 3), 0) = edges_on_cell(icell, 4);
        edges_on_edge(edges_on_cell(icell, 3), 1) = edges_on_cell(icell, 5);
        edges_on_edge(edges_on_cell(icell, 3), 2) = edges_on_cell(icell, 0);
        edges_on_edge(edges_on_cell(icell, 3), 3) = edges_on_cell(icell, 1);
        edges_on_edge(edges_on_cell(icell, 3), 4) = edges_on_cell(icell, 2);

        edges_on_edge(edges_on_cell(icell, 4), 0) = edges_on_cell(icell, 5);
        edges_on_edge(edges_on_cell(icell, 4), 1) = edges_on_cell(icell, 0);
        edges_on_edge(edges_on_cell(icell, 4), 2) = edges_on_cell(icell, 1);
        edges_on_edge(edges_on_cell(icell, 4), 3) = edges_on_cell(icell, 2);
        edges_on_edge(edges_on_cell(icell, 4), 4) = edges_on_cell(icell, 3);

        edges_on_edge(edges_on_cell(icell, 5), 0) = edges_on_cell(icell, 0);
        edges_on_edge(edges_on_cell(icell, 5), 1) = edges_on_cell(icell, 1);
        edges_on_edge(edges_on_cell(icell, 5), 2) = edges_on_cell(icell, 2);
        edges_on_edge(edges_on_cell(icell, 5), 3) = edges_on_cell(icell, 3);
        edges_on_edge(edges_on_cell(icell, 5), 4) = edges_on_cell(icell, 4);

        edges_on_edge(edges_on_cell(icell, 0), 5) = edges_on_cell(icell, 1);
        edges_on_edge(edges_on_cell(icell, 0), 6) = edges_on_cell(icell, 2);
        edges_on_edge(edges_on_cell(icell, 0), 7) = edges_on_cell(icell, 3);
        edges_on_edge(edges_on_cell(icell, 0), 8) = edges_on_cell(icell, 4);
        edges_on_edge(edges_on_cell(icell, 0), 9) = edges_on_cell(icell, 5);

        edges_on_edge(edges_on_cell(icell, 1), 5) = edges_on_cell(icell, 2);
        edges_on_edge(edges_on_cell(icell, 1), 6) = edges_on_cell(icell, 3);
        edges_on_edge(edges_on_cell(icell, 1), 7) = edges_on_cell(icell, 4);
        edges_on_edge(edges_on_cell(icell, 1), 8) = edges_on_cell(icell, 5);
        edges_on_edge(edges_on_cell(icell, 1), 9) = edges_on_cell(icell, 0);

        edges_on_edge(edges_on_cell(icell, 2), 5) = edges_on_cell(icell, 3);
        edges_on_edge(edges_on_cell(icell, 2), 6) = edges_on_cell(icell, 4);
        edges_on_edge(edges_on_cell(icell, 2), 7) = edges_on_cell(icell, 5);
        edges_on_edge(edges_on_cell(icell, 2), 8) = edges_on_cell(icell, 0);
        edges_on_edge(edges_on_cell(icell, 2), 9) = edges_on_cell(icell, 1);

        weights_on_edge(edges_on_cell(icell, 3), 0) = 1. / 3;
        weights_on_edge(edges_on_cell(icell, 3), 1) = 1. / 6;
        weights_on_edge(edges_on_cell(icell, 3), 2) = 0;
        weights_on_edge(edges_on_cell(icell, 3), 3) = 1. / 6;
        weights_on_edge(edges_on_cell(icell, 3), 4) = 1. / 3;

        weights_on_edge(edges_on_cell(icell, 4), 0) = 1. / 3;
        weights_on_edge(edges_on_cell(icell, 4), 1) = -1. / 6;
        weights_on_edge(edges_on_cell(icell, 4), 2) = 0;
        weights_on_edge(edges_on_cell(icell, 4), 3) = 1. / 6;
        weights_on_edge(edges_on_cell(icell, 4), 4) = -1. / 3;

        weights_on_edge(edges_on_cell(icell, 5), 0) = -1. / 3;
        weights_on_edge(edges_on_cell(icell, 5), 1) = -1. / 6;
        weights_on_edge(edges_on_cell(icell, 5), 2) = 0;
        weights_on_edge(edges_on_cell(icell, 5), 3) = -1. / 6;
        weights_on_edge(edges_on_cell(icell, 5), 4) = -1. / 3;

        weights_on_edge(edges_on_cell(icell, 0), 5) = 1. / 3;
        weights_on_edge(edges_on_cell(icell, 0), 6) = 1. / 6;
        weights_on_edge(edges_on_cell(icell, 0), 7) = 0;
        weights_on_edge(edges_on_cell(icell, 0), 8) = 1. / 6;
        weights_on_edge(edges_on_cell(icell, 0), 9) = 1. / 3;

        weights_on_edge(edges_on_cell(icell, 1), 5) = 1. / 3;
        weights_on_edge(edges_on_cell(icell, 1), 6) = -1. / 6;
        weights_on_edge(edges_on_cell(icell, 1), 7) = 0;
        weights_on_edge(edges_on_cell(icell, 1), 8) = 1. / 6;
        weights_on_edge(edges_on_cell(icell, 1), 9) = -1. / 3;

        weights_on_edge(edges_on_cell(icell, 2), 5) = -1. / 3;
        weights_on_edge(edges_on_cell(icell, 2), 6) = -1. / 6;
        weights_on_edge(edges_on_cell(icell, 2), 7) = 0;
        weights_on_edge(edges_on_cell(icell, 2), 8) = -1. / 6;
        weights_on_edge(edges_on_cell(icell, 2), 9) = -1. / 3;

        cells_on_vertex(vertices_on_cell(icell, 1), 2) = icell;
        cells_on_vertex(vertices_on_cell(icell, 3), 0) = icell;
        cells_on_vertex(vertices_on_cell(icell, 5), 1) = icell;
        cells_on_vertex(vertices_on_cell(icell, 0), 0) = icell;
        cells_on_vertex(vertices_on_cell(icell, 2), 1) = icell;
        cells_on_vertex(vertices_on_cell(icell, 4), 2) = icell;

        edges_on_vertex(vertices_on_cell(icell, 0), 0) =
            edges_on_cell(icell, 0);
        edges_on_vertex(vertices_on_cell(icell, 1), 0) =
            edges_on_cell(icell, 0);
        edges_on_vertex(vertices_on_cell(icell, 2), 2) =
            edges_on_cell(icell, 1);
        edges_on_vertex(vertices_on_cell(icell, 1), 2) =
            edges_on_cell(icell, 1);
        edges_on_vertex(vertices_on_cell(icell, 2), 1) =
            edges_on_cell(icell, 2);
        edges_on_vertex(vertices_on_cell(icell, 3), 1) =
            edges_on_cell(icell, 2);
      });

  // parallel_for("scale_weights",
  //     SimpleBounds<2>(nedges, 2 * maxedges), YAKL_CLASS_LAMBDA(Int iedge, Int
  //     j) { weights_on_edge(iedge, j) *= 1. / sqrt(3);
  // });

  parallel_for(
      "compute_cell_arrays", SimpleBounds<2>(ny, nx),
      YAKL_CLASS_LAMBDA(Int irow, Int icol) {
        Int icell = cellidx(icol, irow);
        area_cell(icell) = dc * dc * sqrt(3) / 2;
        lat_cell(icell) = 0;
        lon_cell(icell) = 0;

        if (irow % 2 == 0) {
          x_cell(icell) = dc * icol + dc / 2;
          y_cell(icell) = dc * (irow + 1) * sqrt(3) / 2;
          z_cell(icell) = 0;
        } else {
          x_cell(icell) = dc * (icol + 1);
          y_cell(icell) = dc * (irow + 1) * sqrt(3) / 2;
          z_cell(icell) = 0;
        }

        x_edge(edges_on_cell(icell, 0)) = x_cell(icell) - dc / 2;
        y_edge(edges_on_cell(icell, 0)) = y_cell(icell);
        z_edge(edges_on_cell(icell, 0)) = 0;

        x_edge(edges_on_cell(icell, 1)) = x_cell(icell) - dc / 2 * cos(pi / 3);
        y_edge(edges_on_cell(icell, 1)) = y_cell(icell) - dc / 2 * sin(pi / 3);
        z_edge(edges_on_cell(icell, 1)) = 0;

        x_edge(edges_on_cell(icell, 2)) = x_cell(icell) + dc / 2 * cos(pi / 3);
        y_edge(edges_on_cell(icell, 2)) = y_cell(icell) - dc / 2 * sin(pi / 3);
        z_edge(edges_on_cell(icell, 2)) = 0;

        angle_edge(edges_on_cell(icell, 0)) = 0;
        angle_edge(edges_on_cell(icell, 1)) = pi / 3;
        angle_edge(edges_on_cell(icell, 2)) = 2 * pi / 3;

        x_vertex(vertices_on_cell(icell, 0)) = x_cell(icell) - dc / 2;
        y_vertex(vertices_on_cell(icell, 0)) = y_cell(icell) + dc * sqrt(3) / 6;
        z_vertex(vertices_on_cell(icell, 0)) = 0;

        x_vertex(vertices_on_cell(icell, 1)) = x_cell(icell) - dc / 2;
        y_vertex(vertices_on_cell(icell, 1)) = y_cell(icell) - dc * sqrt(3) / 6;
        z_vertex(vertices_on_cell(icell, 1)) = 0;

        for (Int j = 0; j < maxedges; ++j) {
          edge_sign_on_cell(icell, j) =
              cells_on_edge(edges_on_cell(icell, j), 0) == icell ? 1 : -1;
        }

        for (Int j = 0; j < nedges_on_cell(icell); ++j) {
          Int jvertex = vertices_on_cell(icell, j);
          for (Int l = 0; l < 3; ++l) {
            if (cells_on_vertex(jvertex, l) == icell) {
              kite_index_on_cell(icell, j) = l;
            }
          }
        }
      });

  parallel_for(
      "compute_edge_arrays", nedges, YAKL_CLASS_LAMBDA(Int iedge) {
        nedges_on_edge(iedge) = 10;
        dc_edge(iedge) = dc;
        dv_edge(iedge) = dc_edge(iedge) * sqrt(3) / 3;
        lat_edge(iedge) = 0;
        lon_edge(iedge) = 0;
      });

  parallel_for(
      "compute_vertex_arrays", nvertices, YAKL_CLASS_LAMBDA(Int ivertex) {
        lat_vertex(ivertex) = 0;
        lon_vertex(ivertex) = 0;
        area_triangle(ivertex) = dc * dc * sqrt(3) / 4;
        for (Int j = 0; j < 3; ++j) {
          kiteareas_on_vertex(ivertex, j) = dc * dc * sqrt(3) / 12;
        }

        for (Int j = 0; j < 3; ++j) {
          edge_sign_on_vertex(ivertex, j) =
              vertices_on_edge(edges_on_vertex(ivertex, j), 0) == ivertex ? -1
                                                                          : 1;
        }
      });

  yakl::memset(max_level_cell, nlayers);
  yakl::memset(max_level_edge_bot, nlayers);
  yakl::memset(max_level_edge_top, nlayers);
  yakl::memset(max_level_vertex_bot, nlayers);
  yakl::memset(max_level_vertex_top, nlayers);
}
} // namespace omega
