#pragma once

#include <common.hpp>
#include <mesh/mpas_mesh.hpp>
#include <model/shallow_water_auxstate.hpp>

namespace omega {

struct VelocityDiffusionOnEdge {
  bool m_enabled = false;

  Int2d m_cells_on_edge;
  Int2d m_vertices_on_edge;
  Real1d m_dc_edge;
  Real1d m_dv_edge;
  Real1d m_mesh_scaling_del2;
  Real2d m_edge_mask;

  Real m_visc_del2;

  void enable(ShallowWaterAuxiliaryState &aux_state) { m_enabled = true; }

#ifdef OMEGA_KOKKOS_SIMD
  KOKKOS_FUNCTION void operator()(const Real2d &vn_tend_edge, Int iedge,
                                  Int kchunk, const RealConst2d &div_cell,
                                  const RealConst2d &rvort_vertex) const {
    const Int kstart = kchunk * vector_length;

    const Int icell0 = m_cells_on_edge(iedge, 0);
    const Int icell1 = m_cells_on_edge(iedge, 1);

    const Int ivertex0 = m_vertices_on_edge(iedge, 0);
    const Int ivertex1 = m_vertices_on_edge(iedge, 1);

    const Real dc_edge_inv = 1._fp / m_dc_edge(iedge);
    const Real dv_edge_inv = 1._fp / m_dv_edge(iedge);

    Vec div_icell0;
    div_icell0.copy_from(&div_cell(icell0, kstart), VecTag());
    Vec div_icell1;
    div_icell1.copy_from(&div_cell(icell1, kstart), VecTag());

    Vec rvort_ivertex0;
    rvort_ivertex0.copy_from(&rvort_vertex(ivertex0, kstart), VecTag());
    Vec rvort_ivertex1;
    rvort_ivertex1.copy_from(&rvort_vertex(ivertex1, kstart), VecTag());

    const Vec del2u = ((div_icell1 - div_icell0) * dc_edge_inv -
                       (rvort_ivertex1 - rvort_ivertex0) * dv_edge_inv);

    Vec edge_mask_iedge;
    edge_mask_iedge.copy_from(&m_edge_mask(iedge, kstart), VecTag());

    Vec vn_tend_iedge;
    vn_tend_iedge.copy_from(&vn_tend_edge(iedge, kstart), VecTag());
    vn_tend_iedge +=
        m_visc_del2 * m_mesh_scaling_del2(iedge) * edge_mask_iedge * del2u;
    vn_tend_iedge.copy_to(&vn_tend_edge(iedge, kstart), VecTag());
  }
  
  KOKKOS_FUNCTION void operator()(Vec &vn_tend_edge, Int iedge,
                                  Int kchunk, const RealConst2d &div_cell,
                                  const RealConst2d &rvort_vertex) const {
    const Int kstart = kchunk * vector_length;

    const Int icell0 = m_cells_on_edge(iedge, 0);
    const Int icell1 = m_cells_on_edge(iedge, 1);

    const Int ivertex0 = m_vertices_on_edge(iedge, 0);
    const Int ivertex1 = m_vertices_on_edge(iedge, 1);

    const Real dc_edge_inv = 1._fp / m_dc_edge(iedge);
    const Real dv_edge_inv = 1._fp / m_dv_edge(iedge);

    Vec div_icell0;
    div_icell0.copy_from(&div_cell(icell0, kstart), VecTag());
    Vec div_icell1;
    div_icell1.copy_from(&div_cell(icell1, kstart), VecTag());

    Vec rvort_ivertex0;
    rvort_ivertex0.copy_from(&rvort_vertex(ivertex0, kstart), VecTag());
    Vec rvort_ivertex1;
    rvort_ivertex1.copy_from(&rvort_vertex(ivertex1, kstart), VecTag());

    const Vec del2u = ((div_icell1 - div_icell0) * dc_edge_inv -
                       (rvort_ivertex1 - rvort_ivertex0) * dv_edge_inv);

    Vec edge_mask_iedge;
    edge_mask_iedge.copy_from(&m_edge_mask(iedge, kstart), VecTag());

    vn_tend_edge +=
        m_visc_del2 * m_mesh_scaling_del2(iedge) * edge_mask_iedge * del2u;
  }
#else
  KOKKOS_FUNCTION void operator()(const Real2d &vn_tend_edge, Int iedge,
                                  Int kchunk, const RealConst2d &div_cell,
                                  const RealConst2d &rvort_vertex) const {
    const Int kstart = kchunk * vector_length;

    const Int icell0 = m_cells_on_edge(iedge, 0);
    const Int icell1 = m_cells_on_edge(iedge, 1);

    const Int ivertex0 = m_vertices_on_edge(iedge, 0);
    const Int ivertex1 = m_vertices_on_edge(iedge, 1);

    const Real dc_edge_inv = 1._fp / m_dc_edge(iedge);
    const Real dv_edge_inv = 1._fp / m_dv_edge(iedge);

    OMEGA_SIMD_PRAGMA
    for (Int kvec = 0; kvec < vector_length; ++kvec) {
      const Int k = kstart + kvec;
      const Real del2u =
          ((div_cell(icell1, k) - div_cell(icell0, k)) * dc_edge_inv -
           (rvort_vertex(ivertex1, k) - rvort_vertex(ivertex0, k)) *
               dv_edge_inv);

      vn_tend_edge(iedge, k) += m_edge_mask(iedge, k) * m_visc_del2 *
                                m_mesh_scaling_del2(iedge) * del2u;
    }
  }
  
  KOKKOS_FUNCTION void operator()(Vec &vn_tend_edge, Int iedge,
                                  Int kchunk, const RealConst2d &div_cell,
                                  const RealConst2d &rvort_vertex) const {
    const Int kstart = kchunk * vector_length;

    const Int icell0 = m_cells_on_edge(iedge, 0);
    const Int icell1 = m_cells_on_edge(iedge, 1);

    const Int ivertex0 = m_vertices_on_edge(iedge, 0);
    const Int ivertex1 = m_vertices_on_edge(iedge, 1);

    const Real dc_edge_inv = 1._fp / m_dc_edge(iedge);
    const Real dv_edge_inv = 1._fp / m_dv_edge(iedge);

    OMEGA_SIMD_PRAGMA
    for (Int kvec = 0; kvec < vector_length; ++kvec) {
      const Int k = kstart + kvec;
      const Real del2u =
          ((div_cell(icell1, k) - div_cell(icell0, k)) * dc_edge_inv -
           (rvort_vertex(ivertex1, k) - rvort_vertex(ivertex0, k)) *
               dv_edge_inv);

      vn_tend_edge[kvec] += m_edge_mask(iedge, k) * m_visc_del2 *
                                m_mesh_scaling_del2(iedge) * del2u;
    }
  }
#endif

  VelocityDiffusionOnEdge(const MPASMesh *mesh, Real visc_del2)
      : m_cells_on_edge(mesh->m_cells_on_edge),
        m_vertices_on_edge(mesh->m_vertices_on_edge),
        m_dc_edge(mesh->m_dc_edge), m_dv_edge(mesh->m_dv_edge),
        m_mesh_scaling_del2(mesh->m_mesh_scaling_del2),
        m_edge_mask(mesh->m_edge_mask), m_visc_del2(visc_del2) {}
};

} // namespace omega
