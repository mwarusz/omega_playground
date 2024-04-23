#pragma once

#include <common.hpp>
#include <mesh/mpas_mesh.hpp>

namespace omega {

struct ShallowWaterAuxiliaryState;

struct KineticEnergyOnCell {
  bool m_enabled = false;
  Real2d m_array;

  Int1d m_nedges_on_cell;
  Int2d m_edges_on_cell;
  Real1d m_dc_edge;
  Real1d m_dv_edge;
  Real1d m_area_cell;

  void enable(ShallowWaterAuxiliaryState &aux_state);
  void allocate(const MPASMesh *mesh);
  RealConst2d const_array() const { return m_array; }

  KOKKOS_FUNCTION Real operator()(Int icell, Int k,
                                  const RealConst2d &vn_edge) const {
    Real ke = -0;
    for (Int j = 0; j < m_nedges_on_cell(icell); ++j) {
      Int jedge = m_edges_on_cell(icell, j);
      Real area_edge = m_dv_edge(jedge) * m_dc_edge(jedge);
      ke += area_edge * vn_edge(jedge, k) * vn_edge(jedge, k) * 0.25_fp;
    }
    Real inv_area_cell = 1._fp / m_area_cell(icell);
    ke *= inv_area_cell;
    return ke;
  }

  KineticEnergyOnCell(const MPASMesh *mesh)
      : m_nedges_on_cell(mesh->m_nedges_on_cell),
        m_edges_on_cell(mesh->m_edges_on_cell), m_dc_edge(mesh->m_dc_edge),
        m_dv_edge(mesh->m_dv_edge), m_area_cell(mesh->m_area_cell) {}
};
} // namespace omega
