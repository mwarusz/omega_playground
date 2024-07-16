#pragma once

#include <common.hpp>
#include <mesh/mpas_mesh.hpp>

namespace omega {

struct TracerAuxVars {
  Real3d m_norm_tr_cell;
  Real1d m_area_cell;

#ifdef OMEGA_KOKKOS_SIMD
  KOKKOS_FUNCTION void
  compute_norm_tracer_cell(Int l, Int icell, Int kchunk,
                           const RealConst3d &tr_cell,
                           const RealConst2d &h_cell) const {
    const Int kstart = vector_length * kchunk;

    Vec tr_cell_tmp;
    tr_cell_tmp.copy_from(&tr_cell(l, icell, kstart), VecTag());

    Vec h_cell_tmp;
    h_cell_tmp.copy_from(&h_cell(icell, kstart), VecTag());
    Vec inv_h_cell = 1._fp / h_cell_tmp;

    Vec norm_tr_cell = tr_cell_tmp * inv_h_cell;

    norm_tr_cell.copy_to(&m_norm_tr_cell(l, icell, kstart), VecTag());
  }
#else
  KOKKOS_FUNCTION void
  compute_norm_tracer_cell(Int l, Int icell, Int kchunk,
                           const RealConst3d &tr_cell,
                           const RealConst2d &h_cell) const {
    const Int kstart = vector_length * kchunk;

    Real norm_tr_cell[vector_length];

    OMEGA_SIMD_PRAGMA
    for (Int kvec = 0; kvec < vector_length; ++kvec) {
      const Int k = kstart + kvec;
      const Real inv_h_cell = 1._fp / h_cell(icell, k);
      norm_tr_cell[kvec] = tr_cell(l, icell, k) * inv_h_cell;
    }

    OMEGA_SIMD_PRAGMA
    for (Int kvec = 0; kvec < vector_length; ++kvec) {
      const Int k = kstart + kvec;
      m_norm_tr_cell(l, icell, k) = norm_tr_cell[kvec];
    }
  }
#endif

  TracerAuxVars(const MPASMesh *mesh, Int ntracers)
      : m_norm_tr_cell("norm_tr_cell", ntracers, mesh->m_ncells,
                       mesh->m_nlayers),
        m_area_cell(mesh->m_area_cell) {}
};
} // namespace omega
