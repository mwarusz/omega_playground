#pragma once

#include <common.hpp>
#include <mesh/mpas_mesh.hpp>
#include <model/shallow_water_auxstate.hpp>

namespace omega {

struct TracerHyperDiffusionOnCell {
  bool m_enabled = false;

  Int1d m_nedges_on_cell;
  Int2d m_edges_on_cell;
  Int2d m_cells_on_edge;
  Real2d m_edge_sign_on_cell;
  Real1d m_dv_edge;
  Real1d m_dc_edge;
  Real1d m_area_cell;
  Real1d m_mesh_scaling_del4;

  Real m_eddy_diff4;

  void enable(ShallowWaterAuxiliaryState &aux_state) {
    m_enabled = true;
    aux_state.m_tr_del2_cell.enable(aux_state);
  }

  KOKKOS_FUNCTION Real operator()(Int l, Int icell, Int k,
                                  const RealConst3d &tr_del2_cell) const {
    Real accum = 0;
    for (Int j = 0; j < m_nedges_on_cell(icell); ++j) {
      const Int jedge = m_edges_on_cell(icell, j);

      const Real inv_dc_edge = 1._fp / m_dc_edge(jedge);

      const Int jcell0 = m_cells_on_edge(jedge, 0);
      const Int jcell1 = m_cells_on_edge(jedge, 1);
      const Real grad_tr_del2_edge =
          (tr_del2_cell(l, jcell1, k) - tr_del2_cell(l, jcell0, k)) *
          inv_dc_edge;

      accum += m_eddy_diff4 * m_dv_edge(jedge) * m_edge_sign_on_cell(icell, j) *
               m_mesh_scaling_del4(jedge) * grad_tr_del2_edge;
    }
    const Real inv_area_cell = 1._fp / m_area_cell(icell);
    return -accum * inv_area_cell;
  }

  TracerHyperDiffusionOnCell(const MPASMesh *mesh, Real eddy_diff4)
      : m_nedges_on_cell(mesh->m_nedges_on_cell),
        m_edges_on_cell(mesh->m_edges_on_cell),
        m_cells_on_edge(mesh->m_cells_on_edge),
        m_edge_sign_on_cell(mesh->m_edge_sign_on_cell),
        m_dv_edge(mesh->m_dv_edge), m_dc_edge(mesh->m_dc_edge),
        m_area_cell(mesh->m_area_cell),
        m_mesh_scaling_del4(mesh->m_mesh_scaling_del4),
        m_eddy_diff4(eddy_diff4) {}
};

} // namespace omega
