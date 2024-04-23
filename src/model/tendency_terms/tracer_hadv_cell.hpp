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

  void enable(ShallowWaterAuxiliaryState &aux_state) {
    m_enabled = true;
    aux_state.m_h_flux_edge.enable(aux_state);
    aux_state.m_norm_tr_cell.enable(aux_state);
  }

  KOKKOS_FUNCTION Real operator()(Int l, Int icell, Int k,
                                  const RealConst2d &v_edge,
                                  const RealConst3d &norm_tr_cell,
                                  const RealConst2d &h_flux_edge) const {
    Real accum = 0;
    for (Int j = 0; j < m_nedges_on_cell(icell); ++j) {
      const Int jedge = m_edges_on_cell(icell, j);

      const Int jcell0 = m_cells_on_edge(jedge, 0);
      const Int jcell1 = m_cells_on_edge(jedge, 1);
      const Real norm_tr_edge =
          (norm_tr_cell(l, jcell0, k) + norm_tr_cell(l, jcell1, k)) * 0.5_fp;

      accum += m_dv_edge(jedge) * m_edge_sign_on_cell(icell, j) *
               h_flux_edge(jedge, k) * norm_tr_edge * v_edge(jedge, k);
    }
    Real inv_area_cell = 1._fp / m_area_cell(icell);
    return accum * inv_area_cell;
  }

  TracerHorzAdvOnCell(const MPASMesh *mesh)
      : m_nedges_on_cell(mesh->m_nedges_on_cell),
        m_edges_on_cell(mesh->m_edges_on_cell),
        m_cells_on_edge(mesh->m_cells_on_edge),
        m_edge_sign_on_cell(mesh->m_edge_sign_on_cell),
        m_dv_edge(mesh->m_dv_edge), m_area_cell(mesh->m_area_cell) {}
};

} // namespace omega
