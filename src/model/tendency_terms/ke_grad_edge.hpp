#pragma once

#include <common.hpp>
#include <mesh/mpas_mesh.hpp>
#include <model/shallow_water_auxstate.hpp>

namespace omega {

struct KineticEnergyGradOnEdge {
  bool m_enabled = false;

  Int2d m_cells_on_edge;
  Real1d m_dc_edge;

  void enable(ShallowWaterAuxiliaryState &aux_state) { m_enabled = true; }

  KOKKOS_FUNCTION void operator()(const Real2d &vn_tend_edge, Int iedge, Int kchunk,
                                  const RealConst2d &ke_cell) const {
    const Int kstart = kchunk * vector_length;
    const Int icell0 = m_cells_on_edge(iedge, 0);
    const Int icell1 = m_cells_on_edge(iedge, 1);
    const Real inv_dc_edge = 1._fp / m_dc_edge(iedge);

    for (Int kvec = 0; kvec < vector_length; ++kvec) {
      const Int k = kstart + kvec;
      vn_tend_edge(iedge, k) -=
          (ke_cell(icell1, k) - ke_cell(icell0, k)) * inv_dc_edge;
    }
  }

  KineticEnergyGradOnEdge(const MPASMesh *mesh)
      : m_cells_on_edge(mesh->m_cells_on_edge), m_dc_edge(mesh->m_dc_edge) {}
};

} // namespace omega
