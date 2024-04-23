#pragma once

#include <common.hpp>
#include <mesh/mpas_mesh.hpp>

namespace omega {

struct ShallowWaterAuxiliaryState;

struct TracerDel2OnCell {
  bool m_enabled = false;
  Real3d m_array;

  Int1d m_nedges_on_cell;
  Int2d m_edges_on_cell;
  Int2d m_cells_on_edge;
  Real2d m_edge_sign_on_cell;
  Real1d m_dv_edge;
  Real1d m_dc_edge;
  Real1d m_area_cell;

  void enable(ShallowWaterAuxiliaryState &aux_state);
  void allocate(const MPASMesh *mesh, Int ntracers);
  RealConst3d const_array() const { return m_array; }

  KOKKOS_FUNCTION Real operator()(Int l, Int icell, Int k,
                                  const RealConst3d &norm_tr_cell,
                                  const RealConst2d &h_mean_edge) const {
    Real accum = 0;
    for (Int j = 0; j < m_nedges_on_cell(icell); ++j) {
      const Int jedge = m_edges_on_cell(icell, j);

      const Real inv_dc_edge = 1._fp / m_dc_edge(jedge);

      const Int jcell0 = m_cells_on_edge(jedge, 0);
      const Int jcell1 = m_cells_on_edge(jedge, 1);
      const Real grad_tr_edge =
          (norm_tr_cell(l, jcell1, k) - norm_tr_cell(l, jcell0, k)) *
          inv_dc_edge;

      accum += m_dv_edge(jedge) * m_edge_sign_on_cell(icell, j) *
               h_mean_edge(jedge, k) * grad_tr_edge;
    }
    const Real inv_area_cell = 1._fp / m_area_cell(icell);
    return accum * inv_area_cell;
  }

  TracerDel2OnCell(const MPASMesh *mesh)
      : m_nedges_on_cell(mesh->m_nedges_on_cell),
        m_edges_on_cell(mesh->m_edges_on_cell),
        m_cells_on_edge(mesh->m_cells_on_edge),
        m_edge_sign_on_cell(mesh->m_edge_sign_on_cell),
        m_dv_edge(mesh->m_dv_edge), m_dc_edge(mesh->m_dc_edge),
        m_area_cell(mesh->m_area_cell) {}
};
} // namespace omega
