#pragma once

#include <common.hpp>
#include <mesh/mpas_mesh.hpp>

namespace omega {

struct TracerAuxVars {
  Real3d m_norm_tr_cell;
  Real1d m_area_cell;

  KOKKOS_FUNCTION void
  compute_norm_tracer_cell(Int l, Int icell, Int kchunk, const RealConst3d &tr_cell,
                           const RealConst2d &h_cell) const {
    const Int kstart = vector_length * kchunk;

    Real norm_tr_cell[vector_length];

    for (Int kvec = 0; kvec < vector_length; ++kvec) {
      const Int k = kstart + kvec;
      const Real inv_h_cell = 1._fp / h_cell(icell, k);
      norm_tr_cell[kvec] = tr_cell(l, icell, k) * inv_h_cell;
    }
    
    for (Int kvec = 0; kvec < vector_length; ++kvec) {
      const Int k = kstart + kvec;
      m_norm_tr_cell(l, icell, k) = norm_tr_cell[kvec];
    }
  }

  TracerAuxVars(const MPASMesh *mesh, Int ntracers)
      : m_norm_tr_cell("norm_tr_cell", ntracers, mesh->m_ncells,
                       mesh->m_nlayers),
        m_area_cell(mesh->m_area_cell) {}
};
} // namespace omega
