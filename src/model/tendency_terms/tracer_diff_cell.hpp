#pragma once

#include <common.hpp>
#include <mesh/mpas_mesh.hpp>
#include <model/shallow_water_auxstate.hpp>

namespace omega {

struct TracerDiffusionOnCell {
  bool m_enabled = false;

  Int1d m_nedges_on_cell;
  Int2d m_edges_on_cell;
  Int2d m_cells_on_edge;
  Real2d m_edge_sign_on_cell;
  Real1d m_dv_edge;
  Real1d m_dc_edge;
  Real1d m_area_cell;
  Real1d m_mesh_scaling_del2;

  Real m_eddy_diff2;

  void enable(ShallowWaterAuxiliaryState &aux_state) { m_enabled = true; }

#ifdef OMEGA_KOKKOS_SIMD
  KOKKOS_FUNCTION void operator()(const Real3d &tr_tend_cell, Int l, Int icell,
                                  Int kchunk, const RealConst3d &norm_tr_cell,
                                  const RealConst2d &h_mean_edge) const {
    const Int kstart = kchunk * vector_length;
    const Real inv_area_cell = 1._fp / m_area_cell(icell);

    Vec accum = 0;
    for (Int j = 0; j < m_nedges_on_cell(icell); ++j) {
      const Int jedge = m_edges_on_cell(icell, j);
      const Real inv_dc_edge = 1._fp / m_dc_edge(jedge);

      const Int jcell0 = m_cells_on_edge(jedge, 0);
      const Int jcell1 = m_cells_on_edge(jedge, 1);

      Vec norm_tr_jcell0;
      norm_tr_jcell0.copy_from(&norm_tr_cell(l, jcell0, kstart), VecTag());
      Vec norm_tr_jcell1;
      norm_tr_jcell1.copy_from(&norm_tr_cell(l, jcell1, kstart), VecTag());

      Vec h_mean_jedge;
      h_mean_jedge.copy_from(&h_mean_edge(jedge, kstart), VecTag());

      const Vec grad_tr_jedge = (norm_tr_jcell1 - norm_tr_jcell0) * inv_dc_edge;

      accum += m_eddy_diff2 * inv_area_cell * m_dv_edge(jedge) *
               m_edge_sign_on_cell(icell, j) * m_mesh_scaling_del2(jedge) *
               h_mean_jedge * grad_tr_jedge;
    }

    Vec tr_tend_icell;
    tr_tend_icell.copy_from(&tr_tend_cell(l, icell, kstart), VecTag());
    tr_tend_icell += accum;
    tr_tend_icell.copy_to(&tr_tend_cell(l, icell, kstart), VecTag());
  }
#else
  KOKKOS_FUNCTION void operator()(const Real3d &tr_tend_cell, Int l, Int icell,
                                  Int kchunk, const RealConst3d &norm_tr_cell,
                                  const RealConst2d &h_mean_edge) const {
    const Int kstart = kchunk * vector_length;
    const Real inv_area_cell = 1._fp / m_area_cell(icell);

    Real accum[vector_length] = {0};
    for (Int j = 0; j < m_nedges_on_cell(icell); ++j) {
      const Int jedge = m_edges_on_cell(icell, j);
      const Real inv_dc_edge = 1._fp / m_dc_edge(jedge);

      const Int jcell0 = m_cells_on_edge(jedge, 0);
      const Int jcell1 = m_cells_on_edge(jedge, 1);

      OMEGA_SIMD_PRAGMA
      for (Int kvec = 0; kvec < vector_length; ++kvec) {
        const Int k = kstart + kvec;
        const Real grad_tr_edge =
            (norm_tr_cell(l, jcell1, k) - norm_tr_cell(l, jcell0, k)) *
            inv_dc_edge;

        accum[kvec] += m_eddy_diff2 * inv_area_cell * m_dv_edge(jedge) *
                       m_edge_sign_on_cell(icell, j) * h_mean_edge(jedge, k) *
                       m_mesh_scaling_del2(jedge) * grad_tr_edge;
      }
    }
    OMEGA_SIMD_PRAGMA
    for (Int kvec = 0; kvec < vector_length; ++kvec) {
      const Int k = kstart + kvec;
      tr_tend_cell(l, icell, k) += accum[kvec];
    }
  }
#endif

  TracerDiffusionOnCell(const MPASMesh *mesh, Real eddy_diff2)
      : m_nedges_on_cell(mesh->m_nedges_on_cell),
        m_edges_on_cell(mesh->m_edges_on_cell),
        m_cells_on_edge(mesh->m_cells_on_edge),
        m_edge_sign_on_cell(mesh->m_edge_sign_on_cell),
        m_dv_edge(mesh->m_dv_edge), m_dc_edge(mesh->m_dc_edge),
        m_area_cell(mesh->m_area_cell),
        m_mesh_scaling_del2(mesh->m_mesh_scaling_del2),
        m_eddy_diff2(eddy_diff2) {}
};

} // namespace omega
