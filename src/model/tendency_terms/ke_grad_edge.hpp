#pragma once

#include <common.hpp>
#include <mesh/mpas_mesh.hpp>
#include <model/shallow_water_auxstate.hpp>

namespace omega {

struct KineticEnergyGradOnEdge {
  bool m_enabled = false;

  Int2d m_cells_on_edge;
  Real1d m_dc_edge;

  void enable(ShallowWaterAuxiliaryState &aux_state) {
    m_enabled = true;
    aux_state.m_ke_cell.enable(aux_state);
  }

  KOKKOS_FUNCTION Real operator()(Int iedge, Int k,
                                  const RealConst2d &ke_cell) const {
    Int icell0 = m_cells_on_edge(iedge, 0);
    Int icell1 = m_cells_on_edge(iedge, 1);
    Real inv_dc_edge = 1._fp / m_dc_edge(iedge);
    return (ke_cell(icell1, k) - ke_cell(icell0, k)) * inv_dc_edge;
  }

  KineticEnergyGradOnEdge(const MPASMesh *mesh)
      : m_cells_on_edge(mesh->m_cells_on_edge), m_dc_edge(mesh->m_dc_edge) {}
};

} // namespace omega
