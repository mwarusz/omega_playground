#pragma once

#include <common.hpp>
#include <mesh/mpas_mesh.hpp>

namespace omega {

struct ShallowWaterAuxiliaryState;

struct MeanThicknessOnEdge {
  bool m_enabled = false;
  Real2d m_array;

  Int2d m_cells_on_edge;

  void enable(ShallowWaterAuxiliaryState &aux_state);
  void allocate(const MPASMesh *mesh);
  RealConst2d const_array() const { return m_array; }

  KOKKOS_FUNCTION Real operator()(Int iedge, Int k,
                                  const RealConst2d &h_cell) const {
    Int jcell0 = m_cells_on_edge(iedge, 0);
    Int jcell1 = m_cells_on_edge(iedge, 1);
    return 0.5_fp * (h_cell(jcell0, k) + h_cell(jcell1, k));
  }

  MeanThicknessOnEdge(const MPASMesh *mesh)
      : m_cells_on_edge(mesh->m_cells_on_edge) {}
};
} // namespace omega
