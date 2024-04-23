#pragma once

#include <common.hpp>
#include <mesh/mpas_mesh.hpp>

namespace omega {

struct ShallowWaterAuxiliaryState;

struct RelVortOnVertex {
  bool m_enabled = false;
  Real2d m_array;

  Int2d m_edges_on_vertex;
  Real2d m_edge_sign_on_vertex;
  Real1d m_dc_edge;
  Real1d m_area_triangle;

  void enable(ShallowWaterAuxiliaryState &aux_state);
  void allocate(const MPASMesh *mesh);
  RealConst2d const_array() const { return m_array; }

  KOKKOS_FUNCTION Real operator()(Int ivertex, Int k,
                                  const RealConst2d &vn_edge) const {
    Real rvort = -0;
    for (Int j = 0; j < 3; ++j) {
      Int jedge = m_edges_on_vertex(ivertex, j);
      rvort += m_dc_edge(jedge) * m_edge_sign_on_vertex(ivertex, j) *
               vn_edge(jedge, k);
    }
    Real inv_area_triangle = 1._fp / m_area_triangle(ivertex);
    rvort *= inv_area_triangle;

    return rvort;
  }

  RelVortOnVertex(const MPASMesh *mesh)
      : m_edges_on_vertex(mesh->m_edges_on_vertex),
        m_edge_sign_on_vertex(mesh->m_edge_sign_on_vertex),
        m_dc_edge(mesh->m_dc_edge), m_area_triangle(mesh->m_area_triangle) {}
};
} // namespace omega
