#pragma once

#include <common.hpp>
#include <mesh/mpas_mesh.hpp>
#include <model/shallow_water_auxstate.hpp>

namespace omega {

struct SSHGradOnEdge {
  bool m_enabled = false;

  Real m_grav;
  Int2d m_cells_on_edge;
  Real1d m_dc_edge;

  void enable(ShallowWaterAuxiliaryState &aux_state) { m_enabled = true; }

  KOKKOS_FUNCTION Real operator()(Int iedge, Int k,
                                  const RealConst2d &h_cell) const {
    const Int icell0 = m_cells_on_edge(iedge, 0);
    const Int icell1 = m_cells_on_edge(iedge, 1);
    const Real inv_dc_edge = 1._fp / m_dc_edge(iedge);
    return -m_grav * (h_cell(icell1, k) - h_cell(icell0, k)) * inv_dc_edge;
  }

  SSHGradOnEdge(const MPASMesh *mesh, const Real grav)
      : m_grav(grav), m_cells_on_edge(mesh->m_cells_on_edge),
        m_dc_edge(mesh->m_dc_edge) {}
};

} // namespace omega
