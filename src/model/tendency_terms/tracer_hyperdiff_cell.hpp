#pragma once

#include <common.hpp>
#include <mesh/mpas_mesh.hpp>
#include <model/shallow_water_auxstate.hpp>

namespace omega {

struct TracerHyperDiffusionOnCell {
  bool m_enabled = false;
  Real3d m_tracer_del2_cell;

  Int1d m_nedges_on_cell;
  Int2d m_edges_on_cell;
  Int2d m_cells_on_edge;
  Real2d m_edge_sign_on_cell;
  Real1d m_dv_edge;
  Real1d m_dc_edge;
  Real1d m_area_cell;
  Real1d m_mesh_scaling_del4;

  Real m_eddy_diff4;

  void enable(ShallowWaterAuxiliaryState &aux_state) { m_enabled = true; }

  KOKKOS_FUNCTION void
  compute_tracer_del2(Int l, Int icell, Int kchunk, const RealConst3d &norm_tr_cell,
                      const RealConst2d &h_mean_edge) const {
    const Int kstart = kchunk * vector_size;
    const Real inv_area_cell = 1._fp / m_area_cell(icell);

    Real tracer_del2_cell[vector_length] = {0};
    for (Int j = 0; j < m_nedges_on_cell(icell); ++j) {
      const Int jedge = m_edges_on_cell(icell, j);

      const Real inv_dc_edge = 1._fp / m_dc_edge(jedge);

      const Int jcell0 = m_cells_on_edge(jedge, 0);
      const Int jcell1 = m_cells_on_edge(jedge, 1);

      for (Int kvec = 0; kvec < vector_length; ++kvec) {
        const Int k = kstart + kvec;

        const Real grad_tr_edge =
            (norm_tr_cell(l, jcell1, k) - norm_tr_cell(l, jcell0, k)) *
            inv_dc_edge;

        tracer_del2_cell[kvec] += m_dv_edge(jedge) * inv_area_cell * m_edge_sign_on_cell(icell, j) *
                            h_mean_edge(jedge, k) * grad_tr_edge;
      }
    }

    for (Int kvec = 0; kvec < vector_length; ++kvec) {
      const Int k = kstart + kvec;
      m_tracer_del2_cell(l, icell, k) = tracer_del2_cell[kvec];
    }
  }

  KOKKOS_FUNCTION void operator()(const Real3d &tr_tend_cell, Int l, Int icell,
                                  Int kchunk,
                                  const RealConst3d &tr_del2_cell) const {
    const Int kstart = kchunk * vector_size;
    const Real inv_area_cell = 1._fp / m_area_cell(icell);

    Real accum[vector_length] = {0};
    for (Int j = 0; j < m_nedges_on_cell(icell); ++j) {
      const Int jedge = m_edges_on_cell(icell, j);

      const Real inv_dc_edge = 1._fp / m_dc_edge(jedge);

      const Int jcell0 = m_cells_on_edge(jedge, 0);
      const Int jcell1 = m_cells_on_edge(jedge, 1);

      for (Int kvec = 0; kvec < vector_length; ++kvec) {
        const Int k = kstart + kvec;
        const Real grad_tr_del2_edge =
            (tr_del2_cell(l, jcell1, k) - tr_del2_cell(l, jcell0, k)) *
            inv_dc_edge;

        accum[kvec] -= m_eddy_diff4 * inv_area_cell * m_dv_edge(jedge) * m_edge_sign_on_cell(icell, j) *
                 m_mesh_scaling_del4(jedge) * grad_tr_del2_edge;
      }
    }
    for (Int kvec = 0; kvec < vector_length; ++kvec) {
      const Int k = kstart + kvec;
      tr_tend_cell(l, icell, k) += accum[kvec];
    }
  }

  TracerHyperDiffusionOnCell(const MPASMesh *mesh, Int ntracers,
                             Real eddy_diff4)
      : m_tracer_del2_cell("tracer_del2_cell", ntracers, mesh->m_ncells,
                           mesh->m_nlayers),
        m_nedges_on_cell(mesh->m_nedges_on_cell),
        m_edges_on_cell(mesh->m_edges_on_cell),
        m_cells_on_edge(mesh->m_cells_on_edge),
        m_edge_sign_on_cell(mesh->m_edge_sign_on_cell),
        m_dv_edge(mesh->m_dv_edge), m_dc_edge(mesh->m_dc_edge),
        m_area_cell(mesh->m_area_cell),
        m_mesh_scaling_del4(mesh->m_mesh_scaling_del4),
        m_eddy_diff4(eddy_diff4) {}
};

} // namespace omega
