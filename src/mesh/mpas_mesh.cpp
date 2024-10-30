#include "mpas_mesh.hpp"
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/cuthill_mckee_ordering.hpp>
#include <boost/graph/properties.hpp>
#include <boost/graph/bandwidth.hpp>
#include <numeric>
#include <algorithm>
#include <tuple>

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

static void permute_to(const Int2d &a, const std::vector<Int> &index_to) {
  auto a_perm = create_mirror(a);
  for (Int i = 0; i < a.extent_int(0); ++i) {
    for (Int j = 0; j < a.extent_int(1); ++j) {
      a_perm(i, j) = index_to[a(i, j)];
    }
  }
  deep_copy(a, a_perm);
}

template<class T>
static void permute_from(const Kokkos::View<T*, Layout, MemSpace> &a, const std::vector<Int> &index_to) {
  auto a_perm = create_mirror(a);
  for (Int i = 0; i < a.extent_int(0); ++i) {
    a_perm(index_to[i]) = a(i);
  }
  deep_copy(a, a_perm);
}

template<class T>
static void permute_from(const Kokkos::View<T**, Layout, MemSpace> &a, const std::vector<Int> &index_to) {
  auto a_perm = create_mirror(a);
  for (Int i = 0; i < a.extent_int(0); ++i) {
    for (Int j = 0; j < a.extent_int(1); ++j) {
      a_perm(index_to[i], j) = a(i, j);
    }
  }
  deep_copy(a, a_perm);
}

void MPASMesh::reorder_mesh_cell() {
  using namespace boost;
  using Graph = adjacency_list< vecS, vecS, undirectedS,
        property<vertex_color_t, default_color_type,property<vertex_degree_t,int>> >;

  using Vertex = graph_traits<Graph>::vertex_descriptor;
  using SizeType = graph_traits<Graph>::vertices_size_type;
  
  Graph graph(m_ncells);
  for (Int iedge = 0; iedge < m_nedges; ++iedge) {
    add_edge(m_cells_on_edge(iedge, 0), m_cells_on_edge(iedge, 1), graph);
  }

  std::cout << "original bandwidth: " << bandwidth(graph) << std::endl;

  std::vector<Vertex> inv_perm(num_vertices(graph));
  std::vector<Int> perm(num_vertices(graph));

  using IndexMap = property_map<Graph, vertex_index_t>::type;
  IndexMap index_map = get(vertex_index, graph);

  cuthill_mckee_ordering(graph, inv_perm.rbegin(), get(vertex_color, graph), make_degree_map(graph));

  for (SizeType c = 0; c != inv_perm.size(); ++c) {
      perm[index_map[inv_perm[c]]] = c;
      //perm[c] = m_ncells - 1 - c;
  }

  std::cout << "final bandwidth: "
            << bandwidth(graph,
                   make_iterator_property_map(
                       &perm[0], index_map, perm[0]))
            << std::endl;

  permute_to(m_cells_on_cell, perm);
  permute_to(m_cells_on_edge, perm);
  permute_to(m_cells_on_vertex, perm);
  
  permute_from(m_nedges_on_cell, perm);

  permute_from(m_vertices_on_cell, perm);
  permute_from(m_edges_on_cell, perm);
  permute_from(m_cells_on_cell, perm);
  permute_from(m_kite_index_on_cell, perm);

  permute_from(m_x_cell, perm);
  permute_from(m_y_cell, perm);
  permute_from(m_z_cell, perm);
  permute_from(m_area_cell, perm);
  permute_from(m_lon_cell, perm);
  permute_from(m_lat_cell, perm);

  permute_from(m_edge_sign_on_cell, perm);

  std::vector<Int> inv_edge_perm(m_nedges);
  std::vector<Int> edge_perm(m_nedges);

  std::iota(inv_edge_perm.begin(), inv_edge_perm.end(), 0);
  std::sort(inv_edge_perm.begin(), inv_edge_perm.end(), [&](const Int &a, const Int &b) {
      return std::pair(m_cells_on_edge(a, 0), m_cells_on_edge(a, 1)) < std::pair(m_cells_on_edge(b, 0), m_cells_on_edge(b, 1));
  });
  
  for (SizeType c = 0; c != inv_edge_perm.size(); ++c) {
      edge_perm[inv_edge_perm[c]] = c;
  }
  
  permute_to(m_edges_on_cell, edge_perm);
  permute_to(m_edges_on_edge, edge_perm);
  permute_to(m_edges_on_vertex, edge_perm);
  
  permute_from(m_nedges_on_edge, edge_perm);
  permute_from(m_edges_on_edge, edge_perm);
  permute_from(m_cells_on_edge, edge_perm);
  permute_from(m_vertices_on_edge, edge_perm);

  permute_from(m_dc_edge, edge_perm);
  permute_from(m_dv_edge, edge_perm);
  permute_from(m_angle_edge, edge_perm);
  permute_from(m_lat_edge, edge_perm);
  permute_from(m_lon_edge, edge_perm);
  permute_from(m_x_edge, edge_perm);
  permute_from(m_y_edge, edge_perm);
  permute_from(m_z_edge, edge_perm);
  permute_from(m_weights_on_edge, edge_perm);
  
  std::vector<Int> inv_vertex_perm(m_nvertices);
  std::vector<Int> vertex_perm(m_nvertices);

  std::iota(inv_vertex_perm.begin(), inv_vertex_perm.end(), 0);
  std::sort(inv_vertex_perm.begin(), inv_vertex_perm.end(), [&](const Int &a, const Int &b) {
      return std::tuple(m_cells_on_vertex(a, 0), m_cells_on_vertex(a, 1), m_cells_on_vertex(a, 2)) 
              < std::tuple(m_cells_on_vertex(b, 0), m_cells_on_vertex(b, 1), m_cells_on_vertex(b, 2));
  });
  
  for (SizeType c = 0; c != inv_vertex_perm.size(); ++c) {
      vertex_perm[inv_vertex_perm[c]] = c;
  }
  
  permute_to(m_vertices_on_edge, vertex_perm);
  permute_to(m_vertices_on_cell, vertex_perm);
  
  permute_from(m_edges_on_vertex, vertex_perm);
  permute_from(m_cells_on_vertex, vertex_perm);
  permute_from(m_area_triangle, vertex_perm);
  permute_from(m_lat_vertex, vertex_perm);
  permute_from(m_lon_vertex, vertex_perm);
  permute_from(m_x_vertex, vertex_perm);
  permute_from(m_y_vertex, vertex_perm);
  permute_from(m_z_vertex, vertex_perm);
  permute_from(m_kiteareas_on_vertex, vertex_perm);
  permute_from(m_edge_sign_on_vertex, vertex_perm);
}

void MPASMesh::reorder_mesh_vertex() {
  using namespace boost;
  using Graph = adjacency_list< vecS, vecS, undirectedS,
        property<vertex_color_t, default_color_type,property<vertex_degree_t,int>> >;

  using Vertex = graph_traits<Graph>::vertex_descriptor;
  using SizeType = graph_traits<Graph>::vertices_size_type;
  
  Graph graph(m_nvertices);
  for (Int iedge = 0; iedge < m_nedges; ++iedge) {
    add_edge(m_vertices_on_edge(iedge, 0), m_vertices_on_edge(iedge, 1), graph);
  }

  std::cout << "original bandwidth: " << bandwidth(graph) << std::endl;

  std::vector<Vertex> inv_perm(num_vertices(graph));
  std::vector<Int> perm(num_vertices(graph));

  using IndexMap = property_map<Graph, vertex_index_t>::type;
  IndexMap index_map = get(vertex_index, graph);

  cuthill_mckee_ordering(graph, inv_perm.rbegin(), get(vertex_color, graph), make_degree_map(graph));

  for (SizeType c = 0; c != inv_perm.size(); ++c) {
      perm[index_map[inv_perm[c]]] = c;
  }

  std::cout << "final bandwidth: "
            << bandwidth(graph,
                   make_iterator_property_map(
                       &perm[0], index_map, perm[0]))
            << std::endl;
  
  permute_to(m_vertices_on_edge, perm);
  permute_to(m_vertices_on_cell, perm);
  
  permute_from(m_edges_on_vertex, perm);
  permute_from(m_cells_on_vertex, perm);
  permute_from(m_area_triangle, perm);
  permute_from(m_lat_vertex, perm);
  permute_from(m_lon_vertex, perm);
  permute_from(m_x_vertex, perm);
  permute_from(m_y_vertex, perm);
  permute_from(m_z_vertex, perm);
  permute_from(m_kiteareas_on_vertex, perm);
  permute_from(m_edge_sign_on_vertex, perm);
  

  std::vector<Int> inv_edge_perm(m_nedges);
  std::vector<Int> edge_perm(m_nedges);

  std::iota(inv_edge_perm.begin(), inv_edge_perm.end(), 0);
  std::sort(inv_edge_perm.begin(), inv_edge_perm.end(), [&](const Int &a, const Int &b) {
      return std::pair(m_vertices_on_edge(a, 0), m_vertices_on_edge(a, 1)) < std::pair(m_vertices_on_edge(b, 0), m_vertices_on_edge(b, 1));
  });
  
  for (SizeType c = 0; c != inv_edge_perm.size(); ++c) {
      edge_perm[inv_edge_perm[c]] = c;
  }
  
  permute_to(m_edges_on_cell, edge_perm);
  permute_to(m_edges_on_edge, edge_perm);
  permute_to(m_edges_on_vertex, edge_perm);
  
  permute_from(m_nedges_on_edge, edge_perm);
  permute_from(m_edges_on_edge, edge_perm);
  permute_from(m_cells_on_edge, edge_perm);
  permute_from(m_vertices_on_edge, edge_perm);

  permute_from(m_dc_edge, edge_perm);
  permute_from(m_dv_edge, edge_perm);
  permute_from(m_angle_edge, edge_perm);
  permute_from(m_lat_edge, edge_perm);
  permute_from(m_lon_edge, edge_perm);
  permute_from(m_x_edge, edge_perm);
  permute_from(m_y_edge, edge_perm);
  permute_from(m_z_edge, edge_perm);
  permute_from(m_weights_on_edge, edge_perm);
  
  std::vector<Int> inv_cell_perm(m_ncells);
  std::vector<Int> cell_perm(m_ncells);

  std::iota(inv_cell_perm.begin(), inv_cell_perm.end(), 0);
  std::sort(inv_cell_perm.begin(), inv_cell_perm.end(), [&](const Int &a, const Int &b) {
      std::array<int, 6> va = {0};
      std::array<int, 6> vb = {0};
       
      for (int j = 0; j < m_nedges_on_cell(a); ++j) {
        va[j] = m_vertices_on_cell(a, j);
      }
      for (int j = 0; j < m_nedges_on_cell(b); ++j) {
        vb[j] = m_vertices_on_cell(b, j);
      }

      return std::lexicographical_compare(va.begin(), va.end(), vb.begin(), vb.end());
  });
  
  for (SizeType c = 0; c != inv_cell_perm.size(); ++c) {
      cell_perm[inv_cell_perm[c]] = c;
  }
  
  permute_to(m_cells_on_cell, cell_perm);
  permute_to(m_cells_on_edge, cell_perm);
  permute_to(m_cells_on_vertex, cell_perm);
  
  permute_from(m_nedges_on_cell, cell_perm);
  permute_from(m_vertices_on_cell, cell_perm);
  permute_from(m_edges_on_cell, cell_perm);
  permute_from(m_cells_on_cell, cell_perm);
  permute_from(m_kite_index_on_cell, cell_perm);

  permute_from(m_x_cell, cell_perm);
  permute_from(m_y_cell, cell_perm);
  permute_from(m_z_cell, cell_perm);
  permute_from(m_area_cell, cell_perm);
  permute_from(m_lon_cell, cell_perm);
  permute_from(m_lat_cell, cell_perm);
  permute_from(m_edge_sign_on_cell, cell_perm);
  
}

void MPASMesh::reorder_mesh_edge() {
  using namespace boost;
  using Graph = adjacency_list< vecS, vecS, undirectedS,
        property<vertex_color_t, default_color_type,property<vertex_degree_t,int>> >;

  using Vertex = graph_traits<Graph>::vertex_descriptor;
  using SizeType = graph_traits<Graph>::vertices_size_type;
  
  Graph graph(m_nedges);
  for (Int iedge = 0; iedge < m_nedges; ++iedge) {
    for (Int j = 0; j < m_nedges_on_edge(iedge); ++j) {
      add_edge(iedge, m_edges_on_edge(iedge, j), graph);
    }
  }

  std::cout << "original bandwidth: " << bandwidth(graph) << std::endl;

  std::vector<Vertex> inv_perm(num_vertices(graph));
  std::vector<Int> perm(num_vertices(graph));

  using IndexMap = property_map<Graph, vertex_index_t>::type;
  IndexMap index_map = get(vertex_index, graph);

  cuthill_mckee_ordering(graph, inv_perm.rbegin(), get(vertex_color, graph), make_degree_map(graph));

  for (SizeType c = 0; c != inv_perm.size(); ++c) {
      perm[index_map[inv_perm[c]]] = c;
  }

  std::cout << "final bandwidth: "
            << bandwidth(graph,
                   make_iterator_property_map(
                       &perm[0], index_map, perm[0]))
            << std::endl;
  
  permute_to(m_edges_on_cell, perm);
  permute_to(m_edges_on_edge, perm);
  permute_to(m_edges_on_vertex, perm);
  
  permute_from(m_nedges_on_edge, perm);
  permute_from(m_edges_on_edge, perm);
  permute_from(m_cells_on_edge, perm);
  permute_from(m_vertices_on_edge, perm);

  permute_from(m_dc_edge, perm);
  permute_from(m_dv_edge, perm);
  permute_from(m_angle_edge, perm);
  permute_from(m_lat_edge, perm);
  permute_from(m_lon_edge, perm);
  permute_from(m_x_edge, perm);
  permute_from(m_y_edge, perm);
  permute_from(m_z_edge, perm);
  permute_from(m_weights_on_edge, perm);
  
  
  std::vector<Int> inv_cell_perm(m_ncells);
  std::vector<Int> cell_perm(m_ncells);

  std::iota(inv_cell_perm.begin(), inv_cell_perm.end(), 0);
  std::sort(inv_cell_perm.begin(), inv_cell_perm.end(), [&](const Int &a, const Int &b) {
      std::array<int, 6> va = {0};
      std::array<int, 6> vb = {0};
       
      for (int j = 0; j < m_nedges_on_cell(a); ++j) {
        va[j] = m_edges_on_cell(a, j);
      }
      for (int j = 0; j < m_nedges_on_cell(b); ++j) {
        vb[j] = m_edges_on_cell(b, j);
      }

      return std::lexicographical_compare(va.begin(), va.end(), vb.begin(), vb.end());
  });
  
  for (SizeType c = 0; c != inv_cell_perm.size(); ++c) {
      cell_perm[inv_cell_perm[c]] = c;
  }
  
  permute_to(m_cells_on_cell, cell_perm);
  permute_to(m_cells_on_edge, cell_perm);
  permute_to(m_cells_on_vertex, cell_perm);
  
  permute_from(m_nedges_on_cell, cell_perm);
  permute_from(m_vertices_on_cell, cell_perm);
  permute_from(m_edges_on_cell, cell_perm);
  permute_from(m_cells_on_cell, cell_perm);
  permute_from(m_kite_index_on_cell, cell_perm);

  permute_from(m_x_cell, cell_perm);
  permute_from(m_y_cell, cell_perm);
  permute_from(m_z_cell, cell_perm);
  permute_from(m_area_cell, cell_perm);
  permute_from(m_lon_cell, cell_perm);
  permute_from(m_lat_cell, cell_perm);
  permute_from(m_edge_sign_on_cell, cell_perm);
  
  std::vector<Int> inv_vertex_perm(m_nvertices);
  std::vector<Int> vertex_perm(m_nvertices);

  std::iota(inv_vertex_perm.begin(), inv_vertex_perm.end(), 0);
  std::sort(inv_vertex_perm.begin(), inv_vertex_perm.end(), [&](const Int &a, const Int &b) {
      std::array<int, 6> va = {0};
      std::array<int, 6> vb = {0};
       
      for (int j = 0; j < 3; ++j) {
        va[j] = m_edges_on_vertex(a, j);
      }
      for (int j = 0; j < 3; ++j) {
        vb[j] = m_edges_on_vertex(b, j);
      }

      return std::lexicographical_compare(va.begin(), va.end(), vb.begin(), vb.end());


  });
  
  for (SizeType c = 0; c != inv_vertex_perm.size(); ++c) {
      vertex_perm[inv_vertex_perm[c]] = c;
  }
  
  permute_to(m_vertices_on_edge, vertex_perm);
  permute_to(m_vertices_on_cell, vertex_perm);
  
  permute_from(m_edges_on_vertex, vertex_perm);
  permute_from(m_cells_on_vertex, vertex_perm);
  permute_from(m_area_triangle, vertex_perm);
  permute_from(m_lat_vertex, vertex_perm);
  permute_from(m_lon_vertex, vertex_perm);
  permute_from(m_x_vertex, vertex_perm);
  permute_from(m_y_vertex, vertex_perm);
  permute_from(m_z_vertex, vertex_perm);
  permute_from(m_kiteareas_on_vertex, vertex_perm);
  permute_from(m_edge_sign_on_vertex, vertex_perm);
}

} // namespace omega
