#pragma once

#include <common.hpp>
#include <mesh/mpas_mesh.hpp>

namespace omega {

struct ThicknessAuxVars {
  Real2d m_mean_h_edge;
  Real2d m_flux_h_edge;

  Int2d m_cells_on_edge;

  KOKKOS_FUNCTION void compute_thickness_edge(Int iedge, Int kchunk,
                                              const RealConst2d &h_cell) const {
    const Int kstart = kchunk * vector_length;
    const Int jcell0 = m_cells_on_edge(iedge, 0);
    const Int jcell1 = m_cells_on_edge(iedge, 1);

    Real mean_h_edge[vector_length];
    for (Int kvec = 0; kvec < vector_length; ++kvec) {
      const Int k = kstart + kvec;
      mean_h_edge[kvec] = 0.5_fp * (h_cell(jcell0, k) + h_cell(jcell1, k));
    }

    for (Int kvec = 0; kvec < vector_length; ++kvec) {
      const Int k = kstart + kvec;
      m_mean_h_edge(iedge, k) = mean_h_edge[kvec];
      m_flux_h_edge(iedge, k) = mean_h_edge[kvec];
    }
  }

  ThicknessAuxVars(const MPASMesh *mesh)
      : m_mean_h_edge("mean_h_edge", mesh->m_nedges, mesh->m_nlayers),
        m_flux_h_edge("flux_h_edge", mesh->m_nedges, mesh->m_nlayers),
        m_cells_on_edge(mesh->m_cells_on_edge) {}
};
} // namespace omega
