#pragma once

#include <common.hpp>
#include <mesh/mpas_mesh.hpp>
#include <model/shallow_water_auxstate.hpp>

namespace omega {

struct PotentialVortFluxOnEdge {
  bool m_enabled = false;

  Int1d m_nedges_on_edge;
  Int2d m_edges_on_edge;
  Real2d m_weights_on_edge;

  void enable(ShallowWaterAuxiliaryState &aux_state) {
    m_enabled = true;
    aux_state.m_norm_rvort_edge.enable(aux_state);
    aux_state.m_norm_f_edge.enable(aux_state);
    aux_state.m_h_flux_edge.enable(aux_state);
  }

  KOKKOS_FUNCTION Real operator()(Int iedge, Int k,
                                  const RealConst2d &norm_rvort_edge,
                                  const RealConst2d &norm_f_edge,
                                  const RealConst2d &h_flux_edge,
                                  const RealConst2d &vn_edge) const {
    Real qt = -0;
    for (Int j = 0; j < m_nedges_on_edge(iedge); ++j) {
      Int jedge = m_edges_on_edge(iedge, j);

      Real norm_vort = (norm_rvort_edge(iedge, k) + norm_f_edge(iedge, k) +
                        norm_rvort_edge(jedge, k) + norm_f_edge(jedge, k)) *
                       0.5_fp;

      qt += m_weights_on_edge(iedge, j) * h_flux_edge(jedge, k) *
            vn_edge(jedge, k) * norm_vort;
    }

    return qt;
  }

  PotentialVortFluxOnEdge(const MPASMesh *mesh)
      : m_nedges_on_edge(mesh->m_nedges_on_edge),
        m_edges_on_edge(mesh->m_edges_on_edge),
        m_weights_on_edge(mesh->m_weights_on_edge) {}
};

} // namespace omega
