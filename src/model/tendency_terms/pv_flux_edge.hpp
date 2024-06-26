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

  void enable(ShallowWaterAuxiliaryState &aux_state) { m_enabled = true; }

#ifdef OMEGA_KOKKOS_SIMD
  KOKKOS_FUNCTION void operator()(const Real2d &vn_tend_edge, Int iedge, Int kchunk,
                                  const RealConst2d &norm_rvort_edge,
                                  const RealConst2d &norm_f_edge,
                                  const RealConst2d &h_flux_edge,
                                  const RealConst2d &vn_edge) const {
    const Int kstart = kchunk * vector_length;

    Vec qt = 0;
    Vec norm_rvort_iedge;
    Vec norm_f_iedge;
    norm_rvort_iedge.copy_from(&norm_rvort_edge(iedge, kstart), VecTag());
    norm_f_iedge.copy_from(&norm_f_edge(iedge, kstart), VecTag());

    for (Int j = 0; j < m_nedges_on_edge(iedge); ++j) {
      const Int jedge = m_edges_on_edge(iedge, j);

      Vec norm_rvort_jedge;
      Vec norm_f_jedge;
      norm_rvort_jedge.copy_from(&norm_rvort_edge(jedge, kstart), VecTag());
      norm_f_jedge.copy_from(&norm_f_edge(jedge, kstart), VecTag());
      
      Vec vn_jedge;
      vn_jedge.copy_from(&vn_edge(jedge, kstart), VecTag());
      Vec h_flux_jedge;
      h_flux_jedge.copy_from(&h_flux_edge(jedge, kstart), VecTag());

      const Vec norm_vort = 0.5_fp * (norm_rvort_iedge + norm_f_iedge +
                                      norm_rvort_jedge + norm_f_jedge);

      qt += m_weights_on_edge(iedge, j) * h_flux_jedge *
              vn_jedge * norm_vort;
    }

    Vec vn_tend_iedge;
    vn_tend_iedge.copy_from(&vn_tend_edge(iedge, kstart), VecTag());
    vn_tend_iedge += qt;
    vn_tend_iedge.copy_to(&vn_tend_edge(iedge, kstart), VecTag());
  }
#else
  KOKKOS_FUNCTION void operator()(const Real2d &vn_tend_edge, Int iedge, Int kchunk,
                                  const RealConst2d &norm_rvort_edge,
                                  const RealConst2d &norm_f_edge,
                                  const RealConst2d &h_flux_edge,
                                  const RealConst2d &vn_edge) const {
    const Int kstart = kchunk * vector_length;

    Real qt[vector_length] = {0};
    for (Int j = 0; j < m_nedges_on_edge(iedge); ++j) {
      const Int jedge = m_edges_on_edge(iedge, j);

      for (Int kvec = 0; kvec < vector_length; ++kvec) {
        const Int k = kstart + kvec;
        const Real norm_vort =
            (norm_rvort_edge(iedge, k) + norm_f_edge(iedge, k) +
             norm_rvort_edge(jedge, k) + norm_f_edge(jedge, k)) *
            0.5_fp;

        qt[kvec] += m_weights_on_edge(iedge, j) * h_flux_edge(jedge, k) *
              vn_edge(jedge, k) * norm_vort;
      }
    }

    for (Int kvec = 0; kvec < vector_length; ++kvec) {
      const Int k = kstart + kvec;
      vn_tend_edge(iedge, k) += qt[kvec];
    }
  }
#endif

  PotentialVortFluxOnEdge(const MPASMesh *mesh)
      : m_nedges_on_edge(mesh->m_nedges_on_edge),
        m_edges_on_edge(mesh->m_edges_on_edge),
        m_weights_on_edge(mesh->m_weights_on_edge) {}
};

} // namespace omega
