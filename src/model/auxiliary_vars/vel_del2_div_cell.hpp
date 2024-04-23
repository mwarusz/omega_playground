#pragma once

#include <common.hpp>
#include <mesh/mpas_mesh.hpp>

namespace omega {
struct ShallowWaterAuxiliaryState;

struct VelDel2DivOnCell {
  bool m_enabled = false;
  Real2d m_array;

  Int1d m_nedges_on_cell;
  Int2d m_edges_on_cell;
  Real2d m_edge_sign_on_cell;
  Real1d m_dv_edge;
  Real1d m_area_cell;

  void enable(ShallowWaterAuxiliaryState &aux_state);
  void allocate(const MPASMesh *mesh);
  RealConst2d const_array() const { return m_array; }

  KOKKOS_FUNCTION Real operator()(Int icell, Int k,
                                  const RealConst2d &v_edge) const {
    Real accum = 0;
    for (Int j = 0; j < m_nedges_on_cell(icell); ++j) {
      Int jedge = m_edges_on_cell(icell, j);
      accum +=
          m_dv_edge(jedge) * m_edge_sign_on_cell(icell, j) * v_edge(jedge, k);
    }
    Real inv_area_cell = 1._fp / m_area_cell(icell);
    return accum * inv_area_cell;
  }

  VelDel2DivOnCell(const MPASMesh *mesh)
      : m_nedges_on_cell(mesh->m_nedges_on_cell),
        m_edges_on_cell(mesh->m_edges_on_cell),
        m_edge_sign_on_cell(mesh->m_edge_sign_on_cell),
        m_dv_edge(mesh->m_dv_edge), m_area_cell(mesh->m_area_cell) {}
};
} // namespace omega
