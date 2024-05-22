#pragma once

#include <common.hpp>
#include <mesh/mpas_mesh.hpp>

namespace omega {

struct ThicknessAuxVars {
  Real2d m_mean_h_edge;
  Real2d m_flux_h_edge;

  Int2d m_cells_on_edge;

  KOKKOS_FUNCTION void compute_thickness_edge(Int iedge, Int k,
                                              const RealConst2d &h_cell) const {
    const Int jcell0 = m_cells_on_edge(iedge, 0);
    const Int jcell1 = m_cells_on_edge(iedge, 1);
    const Real mean_h_edge = 0.5_fp * (h_cell(jcell0, k) + h_cell(jcell1, k));

    m_mean_h_edge(iedge, k) = mean_h_edge;
    m_flux_h_edge(iedge, k) = mean_h_edge;
  }

  ThicknessAuxVars(const MPASMesh *mesh)
      : m_mean_h_edge("mean_h_edge", mesh->m_nedges, mesh->m_nlayers),
        m_flux_h_edge("flux_h_edge", mesh->m_nedges, mesh->m_nlayers),
        m_cells_on_edge(mesh->m_cells_on_edge) {}
};
} // namespace omega
