#pragma once

#include <common.hpp>
#include <mesh/mpas_mesh.hpp>
#include <model/shallow_water_auxstate.hpp>

namespace omega {

struct ThicknessHorzAdvOnCell {
  bool m_enabled = false;

  Int1d m_nedges_on_cell;
  Int2d m_edges_on_cell;
  Real2d m_edge_sign_on_cell;
  Real1d m_dv_edge;
  Real1d m_area_cell;

  void enable(ShallowWaterAuxiliaryState &aux_state) { m_enabled = true; }

#ifdef OMEGA_KOKKOS_SIMD
  KOKKOS_FUNCTION void operator()(const Real2d &h_tend_cell, Int icell, Int kchunk,
                                  const RealConst2d &vn_edge,
                                  const RealConst2d &h_flux_edge) const {

    const Int kstart = kchunk * vector_length;
    const Real inv_area_cell = 1._fp / m_area_cell(icell);
    
    Vec accum = 0;
    for (Int j = 0; j < m_nedges_on_cell(icell); ++j) {
      const Int jedge = m_edges_on_cell(icell, j);

      Vec vn_jedge;
      vn_jedge.copy_from(&vn_edge(jedge, kstart), VecTag());
      
      Vec h_flux_jedge;
      h_flux_jedge.copy_from(&h_flux_edge(jedge, kstart), VecTag());

      accum -= m_dv_edge(jedge) * inv_area_cell * m_edge_sign_on_cell(icell, j) *
               h_flux_jedge * vn_jedge;
    }

    Vec h_tend_icell;
    h_tend_icell.copy_from(&h_tend_cell(icell, kstart), VecTag());
    h_tend_icell += accum;
    h_tend_icell.copy_to(&h_tend_cell(icell, kstart), VecTag());
  }
#else
  KOKKOS_FUNCTION void operator()(const Real2d &h_tend_cell, Int icell, Int kchunk,
                                  const RealConst2d &v_edge,
                                  const RealConst2d &h_edge) const {

    const Int kstart = kchunk * vector_length;
    const Real inv_area_cell = 1._fp / m_area_cell(icell);
    
    Real accum[vector_length] = {0};
    for (Int j = 0; j < m_nedges_on_cell(icell); ++j) {
      const Int jedge = m_edges_on_cell(icell, j);

      for (Int kvec = 0; kvec < vector_length; ++kvec) {
        const Int k = kstart + kvec;
        accum[kvec] -= m_dv_edge(jedge) * inv_area_cell * m_edge_sign_on_cell(icell, j) *
                 h_edge(jedge, k) * v_edge(jedge, k);
      }
    }
    for (Int kvec = 0; kvec < vector_length; ++kvec) {
      const Int k = kstart + kvec;
      h_tend_cell(icell, k) += accum[kvec];
    }
  }
#endif

  ThicknessHorzAdvOnCell(const MPASMesh *mesh)
      : m_nedges_on_cell(mesh->m_nedges_on_cell),
        m_edges_on_cell(mesh->m_edges_on_cell),
        m_edge_sign_on_cell(mesh->m_edge_sign_on_cell),
        m_dv_edge(mesh->m_dv_edge), m_area_cell(mesh->m_area_cell) {}
};

} // namespace omega
