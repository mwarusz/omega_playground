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

#ifdef OMEGA_KOKKOS_SIMD
  KOKKOS_FUNCTION void operator()(const Real2d &vn_tend_edge, Int iedge,
                                  Int kchunk, const RealConst2d &h_cell) const {
    const Int kstart = kchunk * vector_length;
    const Int icell0 = m_cells_on_edge(iedge, 0);
    const Int icell1 = m_cells_on_edge(iedge, 1);
    const Real inv_dc_edge = 1._fp / m_dc_edge(iedge);

    Vec h_icell0;
    h_icell0.copy_from(&h_cell(icell0, kstart), VecTag());
    Vec h_icell1;
    h_icell1.copy_from(&h_cell(icell1, kstart), VecTag());

    Vec vn_tend_iedge;
    vn_tend_iedge.copy_from(&vn_tend_edge(iedge, kstart), VecTag());
    vn_tend_iedge -= m_grav * (h_icell1 - h_icell0) * inv_dc_edge;
    vn_tend_iedge.copy_to(&vn_tend_edge(iedge, kstart), VecTag());
  }
  
  KOKKOS_FUNCTION void operator()(Vec &vn_tend_edge, Int iedge,
                                  Int kchunk, const RealConst2d &h_cell) const {
    const Int kstart = kchunk * vector_length;
    const Int icell0 = m_cells_on_edge(iedge, 0);
    const Int icell1 = m_cells_on_edge(iedge, 1);
    const Real inv_dc_edge = 1._fp / m_dc_edge(iedge);

    Vec h_icell0;
    h_icell0.copy_from(&h_cell(icell0, kstart), VecTag());
    Vec h_icell1;
    h_icell1.copy_from(&h_cell(icell1, kstart), VecTag());

    vn_tend_edge -= m_grav * (h_icell1 - h_icell0) * inv_dc_edge;
  }
#else
  KOKKOS_FUNCTION void operator()(const Real2d &vn_tend_edge, Int iedge,
                                  Int kchunk, const RealConst2d &h_cell) const {
    const Int kstart = kchunk * vector_length;
    const Int icell0 = m_cells_on_edge(iedge, 0);
    const Int icell1 = m_cells_on_edge(iedge, 1);
    const Real inv_dc_edge = 1._fp / m_dc_edge(iedge);

    OMEGA_SIMD_PRAGMA
    for (Int kvec = 0; kvec < vector_length; ++kvec) {
      const Int k = kstart + kvec;
      vn_tend_edge(iedge, k) -=
          m_grav * (h_cell(icell1, k) - h_cell(icell0, k)) * inv_dc_edge;
    }
  }
  
  KOKKOS_FUNCTION void operator()(Vec &vn_tend_edge, Int iedge,
                                  Int kchunk, const RealConst2d &h_cell) const {
    const Int kstart = kchunk * vector_length;
    const Int icell0 = m_cells_on_edge(iedge, 0);
    const Int icell1 = m_cells_on_edge(iedge, 1);
    const Real inv_dc_edge = 1._fp / m_dc_edge(iedge);

    OMEGA_SIMD_PRAGMA
    for (Int kvec = 0; kvec < vector_length; ++kvec) {
      const Int k = kstart + kvec;
      vn_tend_edge[kvec] -=
          m_grav * (h_cell(icell1, k) - h_cell(icell0, k)) * inv_dc_edge;
    }
  }
#endif

  SSHGradOnEdge(const MPASMesh *mesh, const Real grav)
      : m_grav(grav), m_cells_on_edge(mesh->m_cells_on_edge),
        m_dc_edge(mesh->m_dc_edge) {}
};

} // namespace omega
