#pragma once

#include <common.hpp>
#include <mesh/mpas_mesh.hpp>

namespace omega {

struct ShallowWaterAuxiliaryState;

struct NormCoriolisOnEdge {
  bool m_enabled = false;
  Real2d m_array;

  Int2d m_vertices_on_edge;

  void enable(ShallowWaterAuxiliaryState &aux_state);
  void allocate(const MPASMesh *mesh);
  RealConst2d const_array() const { return m_array; }

  KOKKOS_FUNCTION Real operator()(Int iedge, Int k,
                                  const RealConst2d &norm_f_vertex) const {
    Int jvertex0 = m_vertices_on_edge(iedge, 0);
    Int jvertex1 = m_vertices_on_edge(iedge, 1);
    return 0.5_fp * (norm_f_vertex(jvertex0, k) + norm_f_vertex(jvertex1, k));
  }

  NormCoriolisOnEdge(const MPASMesh *mesh)
      : m_vertices_on_edge(mesh->m_vertices_on_edge) {}
};
} // namespace omega
