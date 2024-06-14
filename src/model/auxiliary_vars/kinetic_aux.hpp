#pragma once

#include <common.hpp>
#include <mesh/mpas_mesh.hpp>

namespace omega {

struct KineticAuxVars {
  Real2d m_ke_cell;
  Real2d m_vel_div_cell;

  Int1d m_nedges_on_cell;
  Int2d m_edges_on_cell;

  Real1d m_dc_edge;
  Real1d m_dv_edge;
  Real1d m_area_cell;
  Real2d m_edge_sign_on_cell;

#ifdef OMEGA_KOKKOS_SIMD
  KOKKOS_FUNCTION void compute_kinetic_cell(Int icell, Int kchunk,
                                            const RealConst2d &vn_edge) const {
    const Int kstart = kchunk * vector_length;
    const Real inv_area_cell = 1._fp / m_area_cell(icell);

    Vec ke_cell = 0;
    Vec vel_div_cell = 0;
    for (Int j = 0; j < m_nedges_on_cell(icell); ++j) {
      const Int jedge = m_edges_on_cell(icell, j);
      const Real area_edge = m_dv_edge(jedge) * m_dc_edge(jedge);

      Vec vn_edge_tmp;
      vn_edge_tmp.copy_from(&vn_edge(jedge, kstart), VecTag());
      ke_cell += inv_area_cell * area_edge * vn_edge_tmp * vn_edge_tmp * 0.25_fp;
      vel_div_cell +=
            inv_area_cell * m_dv_edge(jedge) * m_edge_sign_on_cell(icell, j) * vn_edge_tmp;
    }

    vel_div_cell.copy_to(&m_vel_div_cell(icell, kstart), VecTag());
    ke_cell.copy_to(&m_ke_cell(icell, kstart), VecTag());
  }
#else
  KOKKOS_FUNCTION void compute_kinetic_cell(Int icell, Int kchunk,
                                            const RealConst2d &vn_edge) const {
    const Int kstart = kchunk * vector_length;

    Real ke_cell[vector_length] = {0};
    Real vel_div_cell[vector_length] = {0};
    
    const Real inv_area_cell = 1._fp / m_area_cell(icell);

    for (Int j = 0; j < m_nedges_on_cell(icell); ++j) {
      const Int jedge = m_edges_on_cell(icell, j);
      const Real area_edge = m_dv_edge(jedge) * m_dc_edge(jedge);

      for (Int kvec = 0; kvec < vector_length; ++kvec) {
        const Int k = kstart + kvec;
        ke_cell[kvec] += inv_area_cell * area_edge * vn_edge(jedge, k) * vn_edge(jedge, k) * 0.25_fp;
        vel_div_cell[kvec] +=
            inv_area_cell * m_dv_edge(jedge) * m_edge_sign_on_cell(icell, j) * vn_edge(jedge, k);
      }
    }

    for (Int kvec = 0; kvec < vector_length; ++kvec) {
      const Int k = kstart + kvec;
      m_vel_div_cell(icell, k) = vel_div_cell[kvec];
      m_ke_cell(icell, k) = ke_cell[kvec];
    }
  }
#endif

  KineticAuxVars(const MPASMesh *mesh)
      : m_ke_cell("ke_cell", mesh->m_ncells, mesh->m_nlayers),
        m_vel_div_cell("vel_div_cell", mesh->m_ncells, mesh->m_nlayers),
        m_nedges_on_cell(mesh->m_nedges_on_cell),
        m_edges_on_cell(mesh->m_edges_on_cell), m_dc_edge(mesh->m_dc_edge),
        m_dv_edge(mesh->m_dv_edge), m_area_cell(mesh->m_area_cell),
        m_edge_sign_on_cell(mesh->m_edge_sign_on_cell) {}
};
} // namespace omega
