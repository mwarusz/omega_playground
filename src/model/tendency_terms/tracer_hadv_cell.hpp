#pragma once

#include <common.hpp>
#include <mesh/mpas_mesh.hpp>
#include <model/shallow_water_auxstate.hpp>

namespace omega {

struct TracerHorzAdvOnCell {
  bool m_enabled = false;

  Int1d m_nedges_on_cell;
  Int2d m_edges_on_cell;
  Int2d m_cells_on_edge;
  Real2d m_edge_sign_on_cell;
  Real1d m_dv_edge;
  Real1d m_area_cell;

  void enable(ShallowWaterAuxiliaryState &aux_state) { m_enabled = true; }

#ifdef OMEGA_KOKKOS_SIMD
  KOKKOS_FUNCTION void operator()(const Real3d &tr_tend_cell, Int l, Int icell,
                                  Int kchunk, const RealConst2d &vn_edge,
                                  const RealConst3d &norm_tr_cell,
                                  const RealConst2d &h_flux_edge) const {
    const Int kstart = kchunk * vector_length;
    const Real inv_area_cell = 1._fp / m_area_cell(icell);

    Vec accum = 0;
    for (Int j = 0; j < m_nedges_on_cell(icell); ++j) {
      const Int jedge = m_edges_on_cell(icell, j);

      const Int jcell0 = m_cells_on_edge(jedge, 0);
      const Int jcell1 = m_cells_on_edge(jedge, 1);

      Vec norm_tr_jcell0;
      norm_tr_jcell0.copy_from(&norm_tr_cell(l, jcell0, kstart), VecTag());
      Vec norm_tr_jcell1;
      norm_tr_jcell1.copy_from(&norm_tr_cell(l, jcell1, kstart), VecTag());

      const Vec norm_tr_jedge = 0.5_fp * (norm_tr_jcell0 + norm_tr_jcell1);

      Vec h_flux_jedge;
      h_flux_jedge.copy_from(&h_flux_edge(jedge, kstart), VecTag());
      
      Vec vn_jedge;
      vn_jedge.copy_from(&vn_edge(jedge, kstart), VecTag());

      accum -= m_dv_edge(jedge) * inv_area_cell * m_edge_sign_on_cell(icell, j) *
               h_flux_jedge * norm_tr_jedge * vn_jedge;
    }

    Vec tr_tend_icell;
    tr_tend_icell.copy_from(&tr_tend_cell(l, icell, kstart), VecTag());
    tr_tend_icell += accum;
    tr_tend_icell.copy_to(&tr_tend_cell(l, icell, kstart), VecTag());
  }
#else
  KOKKOS_FUNCTION void operator()(const Real3d &tr_tend_cell, Int l, Int icell,
                                  Int kchunk, const RealConst2d &v_edge,
                                  const RealConst3d &norm_tr_cell,
                                  const RealConst2d &h_flux_edge) const {
    const Int kstart = kchunk * vector_length;
    const Real inv_area_cell = 1._fp / m_area_cell(icell);

    Real accum[vector_length] = {0};
    for (Int j = 0; j < m_nedges_on_cell(icell); ++j) {
      const Int jedge = m_edges_on_cell(icell, j);

      const Int jcell0 = m_cells_on_edge(jedge, 0);
      const Int jcell1 = m_cells_on_edge(jedge, 1);

      for (Int kvec = 0; kvec < vector_length; ++kvec) {
        const Int k = kstart + kvec;
        const Real norm_tr_edge =
            (norm_tr_cell(l, jcell0, k) + norm_tr_cell(l, jcell1, k)) * 0.5_fp;

        accum[kvec] -= m_dv_edge(jedge) * inv_area_cell * m_edge_sign_on_cell(icell, j) *
                 h_flux_edge(jedge, k) * norm_tr_edge * v_edge(jedge, k);
      }
    }
    for (Int kvec = 0; kvec < vector_length; ++kvec) {
      const Int k = kstart + kvec;
      tr_tend_cell(l, icell, k) += accum[kvec];
    }
  }
#endif

  TracerHorzAdvOnCell(const MPASMesh *mesh)
      : m_nedges_on_cell(mesh->m_nedges_on_cell),
        m_edges_on_cell(mesh->m_edges_on_cell),
        m_cells_on_edge(mesh->m_cells_on_edge),
        m_edge_sign_on_cell(mesh->m_edge_sign_on_cell),
        m_dv_edge(mesh->m_dv_edge), m_area_cell(mesh->m_area_cell) {}
};

} // namespace omega
