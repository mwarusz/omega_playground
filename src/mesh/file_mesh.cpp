#include "file_mesh.hpp"

namespace omega {

FileMesh::FileMesh(const std::string &filename, Int nlayers)
    : FileMesh(netCDF::NcFile(filename, netCDF::NcFile::read), nlayers) {}

FileMesh::FileMesh(const netCDF::NcFile &mesh_file, Int nlayers)
    : MPASMesh(mesh_file.getDim("nCells").getSize(),
               mesh_file.getDim("nEdges").getSize(),
               mesh_file.getDim("nVertices").getSize(), nlayers) {

  mesh_file.getVar("nEdgesOnCell").getVar(m_nedges_on_cell.data());
  mesh_file.getVar("edgesOnCell").getVar(m_edges_on_cell.data());
  mesh_file.getVar("cellsOnCell").getVar(m_cells_on_cell.data());
  mesh_file.getVar("verticesOnCell").getVar(m_vertices_on_cell.data());

  mesh_file.getVar("areaCell").getVar(m_area_cell.data());
  mesh_file.getVar("latCell").getVar(m_lat_cell.data());
  mesh_file.getVar("lonCell").getVar(m_lon_cell.data());
  mesh_file.getVar("xCell").getVar(m_x_cell.data());
  mesh_file.getVar("yCell").getVar(m_y_cell.data());
  mesh_file.getVar("zCell").getVar(m_z_cell.data());
  mesh_file.getVar("meshDensity").getVar(m_mesh_density.data());

  mesh_file.getVar("nEdgesOnEdge").getVar(m_nedges_on_edge.data());
  mesh_file.getVar("edgesOnEdge").getVar(m_edges_on_edge.data());
  mesh_file.getVar("cellsOnEdge").getVar(m_cells_on_edge.data());
  mesh_file.getVar("verticesOnEdge").getVar(m_vertices_on_edge.data());

  mesh_file.getVar("dcEdge").getVar(m_dc_edge.data());
  mesh_file.getVar("dvEdge").getVar(m_dv_edge.data());
  mesh_file.getVar("angleEdge").getVar(m_angle_edge.data());
  mesh_file.getVar("latEdge").getVar(m_lat_edge.data());
  mesh_file.getVar("lonEdge").getVar(m_lon_edge.data());
  mesh_file.getVar("xEdge").getVar(m_x_edge.data());
  mesh_file.getVar("yEdge").getVar(m_y_edge.data());
  mesh_file.getVar("zEdge").getVar(m_z_edge.data());
  mesh_file.getVar("weightsOnEdge").getVar(m_weights_on_edge.data());

  mesh_file.getVar("edgesOnVertex").getVar(m_edges_on_vertex.data());
  mesh_file.getVar("cellsOnVertex").getVar(m_cells_on_vertex.data());

  mesh_file.getVar("areaTriangle").getVar(m_area_triangle.data());
  mesh_file.getVar("latVertex").getVar(m_lat_vertex.data());
  mesh_file.getVar("lonVertex").getVar(m_lon_vertex.data());
  mesh_file.getVar("xVertex").getVar(m_x_vertex.data());
  mesh_file.getVar("yVertex").getVar(m_y_vertex.data());
  mesh_file.getVar("zVertex").getVar(m_z_vertex.data());
  mesh_file.getVar("kiteAreasOnVertex").getVar(m_kiteareas_on_vertex.data());

  convert_fortran_indices_to_cxx();
  finalize_mesh();
}

void FileMesh::convert_fortran_indices_to_cxx() const {
  omega_parallel_for(
      "fix_cell_indices", {m_ncells}, KOKKOS_CLASS_LAMBDA(Int icell) {
        for (Int j = 0; j < maxedges; ++j) {
          m_edges_on_cell(icell, j) -= 1;
          m_cells_on_cell(icell, j) -= 1;
          m_vertices_on_cell(icell, j) -= 1;
        }
      });

  omega_parallel_for(
      "fix_edge_indices", {m_nedges}, KOKKOS_CLASS_LAMBDA(Int iedge) {
        for (Int j = 0; j < 2 * maxedges; ++j) {
          m_edges_on_edge(iedge, j) -= 1;
        }
        for (Int j = 0; j < 2; ++j) {
          m_cells_on_edge(iedge, j) -= 1;
          m_vertices_on_edge(iedge, j) -= 1;
        }
      });

  omega_parallel_for(
      "fix_vertex_indices", {m_nvertices}, KOKKOS_CLASS_LAMBDA(Int ivertex) {
        for (Int j = 0; j < 3; ++j) {
          m_edges_on_vertex(ivertex, j) -= 1;
          m_cells_on_vertex(ivertex, j) -= 1;
        }
      });
}

void FileMesh::rescale_radius(Real radius) const {
  Real radius2 = radius * radius;
  omega_parallel_for(
      "rescale_cell", {m_ncells}, KOKKOS_CLASS_LAMBDA(Int icell) {
        m_x_cell(icell) *= radius;
        m_y_cell(icell) *= radius;
        m_z_cell(icell) *= radius;
        m_area_cell(icell) *= radius2;
      });

  omega_parallel_for(
      "rescale_vertex", {m_nvertices}, KOKKOS_CLASS_LAMBDA(Int ivertex) {
        m_x_vertex(ivertex) *= radius;
        m_y_vertex(ivertex) *= radius;
        m_z_vertex(ivertex) *= radius;
        m_area_triangle(ivertex) *= radius2;
        for (Int j = 0; j < 3; ++j) {
          m_kiteareas_on_vertex(ivertex, j) *= radius2;
        }
      });

  omega_parallel_for(
      "rescale_edge", {m_nedges}, KOKKOS_CLASS_LAMBDA(Int iedge) {
        m_x_edge(iedge) *= radius;
        m_y_edge(iedge) *= radius;
        m_z_edge(iedge) *= radius;
        m_dc_edge(iedge) *= radius;
        m_dv_edge(iedge) *= radius;
      });
}

} // namespace omega
