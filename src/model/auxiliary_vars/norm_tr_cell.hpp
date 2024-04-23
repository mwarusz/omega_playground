#pragma once

#include <common.hpp>
#include <mesh/mpas_mesh.hpp>

namespace omega {

struct ShallowWaterAuxiliaryState;

struct NormTracerOnCell {
  bool m_enabled = false;
  Real3d m_array;

  Real1d m_area_cell;

  void enable(ShallowWaterAuxiliaryState &aux_state);
  void allocate(const MPASMesh *mesh, Int ntracers);
  RealConst3d const_array() const { return m_array; }

  KOKKOS_FUNCTION Real operator()(Int l, Int icell, Int k,
                                  const RealConst3d &tr_cell,
                                  const RealConst2d &h_cell) const {
    const Real inv_h_cell = 1._fp / h_cell(icell, k);
    return tr_cell(l, icell, k) * inv_h_cell;
  }

  NormTracerOnCell(const MPASMesh *mesh) : m_area_cell(mesh->m_area_cell) {}
};
} // namespace omega
