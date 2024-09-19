#include "file_mesh.hpp"

namespace omega {

template<class DeviceView>
static void get_from_file(const std::string &name,
                  const netCDF::NcFile &mesh_file,
                  DeviceView view) {
  auto host_view = create_mirror_view(HostMemSpace(), view);
  mesh_file.getVar(name).getVar(host_view.data());
  deep_copy(view, host_view);
}

FileMesh::FileMesh(const std::string &filename, Int nlayers)
    : FileMesh(netCDF::NcFile(filename, netCDF::NcFile::read), nlayers) {}

FileMesh::FileMesh(const netCDF::NcFile &mesh_file, Int nlayers)
    : MPASMesh(mesh_file.getDim("nCells").getSize(),
               mesh_file.getDim("nEdges").getSize(),
               mesh_file.getDim("nVertices").getSize(), nlayers) {

  get_from_file("nEdgesOnCell", mesh_file, m_nedges_on_cell);
  get_from_file("edgesOnCell", mesh_file, m_edges_on_cell);
  get_from_file("cellsOnCell", mesh_file, m_cells_on_cell);
  get_from_file("verticesOnCell", mesh_file, m_vertices_on_cell);
  
  get_from_file("areaCell", mesh_file, m_area_cell);
  get_from_file("latCell", mesh_file, m_lat_cell);
  get_from_file("lonCell", mesh_file, m_lon_cell);
  get_from_file("xCell", mesh_file, m_x_cell);
  get_from_file("yCell", mesh_file, m_y_cell);
  get_from_file("zCell", mesh_file, m_z_cell);
  get_from_file("meshDensity", mesh_file, m_mesh_density);
  
  get_from_file("nEdgesOnEdge", mesh_file, m_nedges_on_edge);
  get_from_file("edgesOnEdge", mesh_file, m_edges_on_edge);
  get_from_file("cellsOnEdge", mesh_file, m_cells_on_edge);
  get_from_file("verticesOnEdge", mesh_file, m_vertices_on_edge);
  
  get_from_file("dcEdge", mesh_file, m_dc_edge);
  get_from_file("dvEdge", mesh_file, m_dv_edge);
  get_from_file("angleEdge", mesh_file, m_angle_edge);
  get_from_file("latEdge", mesh_file, m_lat_edge);
  get_from_file("lonEdge", mesh_file, m_lon_edge);
  get_from_file("xEdge", mesh_file, m_x_edge);
  get_from_file("yEdge", mesh_file, m_y_edge);
  get_from_file("zEdge", mesh_file, m_z_edge);
  get_from_file("weightsOnEdge", mesh_file, m_weights_on_edge);

  get_from_file("edgesOnVertex", mesh_file, m_edges_on_vertex);
  get_from_file("cellsOnVertex", mesh_file, m_cells_on_vertex);

  get_from_file("areaTriangle", mesh_file, m_area_triangle);
  get_from_file("latVertex", mesh_file, m_lat_vertex);
  get_from_file("lonVertex", mesh_file, m_lon_vertex);
  get_from_file("xVertex", mesh_file, m_x_vertex);
  get_from_file("yVertex", mesh_file, m_y_vertex);
  get_from_file("zVertex", mesh_file, m_z_vertex);
  get_from_file("kiteAreasOnVertex", mesh_file, m_kiteareas_on_vertex);

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
