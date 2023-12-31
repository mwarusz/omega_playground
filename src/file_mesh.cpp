#include "YAKL_netcdf.h"
#include <file_mesh.hpp>

namespace omega {

void FileMesh::convert_fortran_indices_to_cxx() const {
  parallel_for(
      "fix_cell_indices", m_ncells, YAKL_CLASS_LAMBDA(Int icell) {
        for (Int j = 0; j < maxedges; ++j) {
          m_edges_on_cell(icell, j) -= 1;
          m_cells_on_cell(icell, j) -= 1;
          m_vertices_on_cell(icell, j) -= 1;
        }
      });

  parallel_for(
      "fix_edge_indices", m_nedges, YAKL_CLASS_LAMBDA(Int iedge) {
        for (Int j = 0; j < 2 * maxedges; ++j) {
          m_edges_on_edge(iedge, j) -= 1;
        }
        for (Int j = 0; j < 2; ++j) {
          m_cells_on_edge(iedge, j) -= 1;
          m_vertices_on_edge(iedge, j) -= 1;
        }
      });

  parallel_for(
      "fix_vertex_indices", m_nvertices, YAKL_CLASS_LAMBDA(Int ivertex) {
        for (Int j = 0; j < 3; ++j) {
          m_edges_on_vertex(ivertex, j) -= 1;
          m_cells_on_vertex(ivertex, j) -= 1;
        }
      });
}

FileMesh::FileMesh(const std::string &filename, Int nlayers) {
  m_nlayers = std::ceil(Real(nlayers) / vector_length) * vector_length;

  yakl::SimpleNetCDF nc;

  nc.open(filename);

  nc.read(m_nedges_on_cell, "nEdgesOnCell");
  nc.read(m_edges_on_cell, "edgesOnCell");
  nc.read(m_cells_on_cell, "cellsOnCell");
  nc.read(m_vertices_on_cell, "verticesOnCell");

  nc.read(m_area_cell, "areaCell");
  nc.read(m_lat_cell, "latCell");
  nc.read(m_lon_cell, "lonCell");
  nc.read(m_x_cell, "xCell");
  nc.read(m_y_cell, "yCell");
  nc.read(m_z_cell, "zCell");
  nc.read(m_mesh_density, "meshDensity");

  m_ncells = m_area_cell.dimension[0];

  nc.read(m_nedges_on_edge, "nEdgesOnEdge");
  nc.read(m_edges_on_edge, "edgesOnEdge");
  nc.read(m_cells_on_edge, "cellsOnEdge");
  nc.read(m_vertices_on_edge, "verticesOnEdge");

  nc.read(m_dc_edge, "dcEdge");
  nc.read(m_dv_edge, "dvEdge");
  nc.read(m_angle_edge, "angleEdge");
  nc.read(m_lat_edge, "latEdge");
  nc.read(m_lon_edge, "lonEdge");
  nc.read(m_x_edge, "xEdge");
  nc.read(m_y_edge, "yEdge");
  nc.read(m_z_edge, "zEdge");
  nc.read(m_weights_on_edge, "weightsOnEdge");

  m_nedges = m_dc_edge.dimension[0];

  nc.read(m_edges_on_vertex, "edgesOnVertex");
  nc.read(m_cells_on_vertex, "cellsOnVertex");

  nc.read(m_area_triangle, "areaTriangle");
  nc.read(m_lat_vertex, "latVertex");
  nc.read(m_lon_vertex, "lonVertex");
  nc.read(m_x_vertex, "xVertex");
  nc.read(m_y_vertex, "yVertex");
  nc.read(m_z_vertex, "zVertex");
  nc.read(m_kiteareas_on_vertex, "kiteAreasOnVertex");

  m_nvertices = m_area_triangle.dimension[0];

  convert_fortran_indices_to_cxx();
  finalize_mesh();
}

void FileMesh::rescale_radius(Real radius) const {
  Real radius2 = radius * radius;
  parallel_for(
      "rescale_cell", m_ncells, YAKL_CLASS_LAMBDA(Int icell) {
        m_x_cell(icell) *= radius;
        m_y_cell(icell) *= radius;
        m_z_cell(icell) *= radius;
        m_area_cell(icell) *= radius2;
      });

  parallel_for(
      "rescale_vertex", m_nvertices, YAKL_CLASS_LAMBDA(Int ivertex) {
        m_x_vertex(ivertex) *= radius;
        m_y_vertex(ivertex) *= radius;
        m_z_vertex(ivertex) *= radius;
        m_area_triangle(ivertex) *= radius2;
        for (Int j = 0; j < 3; ++j) {
          m_kiteareas_on_vertex(ivertex, j) *= radius2;
        }
      });

  parallel_for(
      "rescale_edge", m_nedges, YAKL_CLASS_LAMBDA(Int iedge) {
        m_x_edge(iedge) *= radius;
        m_y_edge(iedge) *= radius;
        m_z_edge(iedge) *= radius;
        m_dc_edge(iedge) *= radius;
        m_dv_edge(iedge) *= radius;
      });
}

} // namespace omega
