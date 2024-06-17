#pragma once

#include <common.hpp>
#include <mesh/mpas_mesh.hpp>
#include <model/shallow_water_auxstate.hpp>

namespace omega {

struct VelocityHyperDiffusionOnEdge {
  bool m_enabled = false;

  Real2d m_vel_del2_edge;
  Real2d m_vel_del2_rvort_vertex;
  Real2d m_vel_del2_div_cell;

  Int2d m_cells_on_edge;
  Int2d m_vertices_on_edge;
  Int2d m_edges_on_vertex;
  Int1d m_nedges_on_cell;
  Int2d m_edges_on_cell;
  Real1d m_dc_edge;
  Real1d m_dv_edge;
  Real1d m_area_cell;
  Real1d m_area_triangle;
  Real1d m_mesh_scaling_del4;
  Real2d m_edge_mask;
  Real2d m_edge_sign_on_vertex;
  Real2d m_edge_sign_on_cell;

  Real m_visc_del4;

  void enable(ShallowWaterAuxiliaryState &aux_state) { m_enabled = true; }

#ifdef OMEGA_KOKKOS_SIMD
  KOKKOS_FUNCTION void compute_vel_del2(Int iedge, Int kchunk,
                                        const RealConst2d &div_cell,
                                        const RealConst2d &rvort_vertex) const {
    const Int kstart = kchunk * vector_length;
    const Int icell0 = m_cells_on_edge(iedge, 0);
    const Int icell1 = m_cells_on_edge(iedge, 1);

    const Int ivertex0 = m_vertices_on_edge(iedge, 0);
    const Int ivertex1 = m_vertices_on_edge(iedge, 1);

    const Real dc_edge_inv = 1._fp / m_dc_edge(iedge);
    const Real dv_edge_inv =
        1._fp / std::max(m_dv_edge(iedge), 0.25_fp * m_dc_edge(iedge)); // huh
                                                                        //
    Vec div_icell0;
    div_icell0.copy_from(&div_cell(icell0, kstart), VecTag());
    Vec div_icell1;
    div_icell1.copy_from(&div_cell(icell1, kstart), VecTag());
    
    Vec rvort_ivertex0;
    rvort_ivertex0.copy_from(&rvort_vertex(ivertex0, kstart), VecTag());
    Vec rvort_ivertex1;
    rvort_ivertex1.copy_from(&rvort_vertex(ivertex1, kstart), VecTag());

    const Vec del2u =
        ((div_icell1 - div_icell0) * dc_edge_inv -
         (rvort_ivertex1 - rvort_ivertex0) * dv_edge_inv);

    del2u.copy_to(&m_vel_del2_edge(iedge, kstart), VecTag());
  }

  KOKKOS_FUNCTION void compute_vel_del2_rvort(Int ivertex, Int kchunk) const {
    const Int kstart = kchunk * vector_length;
    const Real inv_area_triangle = 1._fp / m_area_triangle(ivertex);

    Vec rvort = 0;
    for (Int j = 0; j < 3; ++j) {
      const Int jedge = m_edges_on_vertex(ivertex, j);
      Vec vel_del2_jedge;
      vel_del2_jedge.copy_from(&m_vel_del2_edge(jedge, kstart), VecTag());
      rvort += m_dc_edge(jedge) * inv_area_triangle * m_edge_sign_on_vertex(ivertex, j) *
               m_vel_del2_edge(jedge, kstart);
    }

    rvort.copy_to(&m_vel_del2_rvort_vertex(ivertex, kstart), VecTag());
  }

  KOKKOS_FUNCTION void compute_vel_del2_div(Int icell, Int kchunk) const {
    const Int kstart = kchunk * vector_length;
    const Real inv_area_cell = 1._fp / m_area_cell(icell);

    Vec accum = 0;
    for (Int j = 0; j < m_nedges_on_cell(icell); ++j) {
      const Int jedge = m_edges_on_cell(icell, j);
      Vec vel_del2_jedge;
      vel_del2_jedge.copy_from(&m_vel_del2_edge(jedge, kstart), VecTag());

      accum += m_dv_edge(jedge) * inv_area_cell * m_edge_sign_on_cell(icell, j) *
                 vel_del2_jedge;
    }
   
    accum.copy_to(&m_vel_del2_div_cell(icell, kstart), VecTag());
  }

  KOKKOS_FUNCTION void operator()(const Real2d &vn_tend_edge, Int iedge,
                                  Int kchunk) const {
    const Int kstart = kchunk * vector_length;

    const Int icell0 = m_cells_on_edge(iedge, 0);
    const Int icell1 = m_cells_on_edge(iedge, 1);

    const Int ivertex0 = m_vertices_on_edge(iedge, 0);
    const Int ivertex1 = m_vertices_on_edge(iedge, 1);

    const Real dc_edge_inv = 1._fp / m_dc_edge(iedge);
    const Real dv_edge_inv = 1._fp / m_dv_edge(iedge);

    Vec vel_del2_div_icell0;
    vel_del2_div_icell0.copy_from(&m_vel_del2_div_cell(icell0, kstart), VecTag());
    Vec vel_del2_div_icell1;
    vel_del2_div_icell1.copy_from(&m_vel_del2_div_cell(icell1, kstart), VecTag());

    Vec vel_del2_rvort_ivertex0;
    vel_del2_rvort_ivertex0.copy_from(&m_vel_del2_rvort_vertex(ivertex0, kstart), VecTag());
    Vec vel_del2_rvort_ivertex1;
    vel_del2_rvort_ivertex1.copy_from(&m_vel_del2_rvort_vertex(ivertex1, kstart), VecTag());

    const Vec del2u = (vel_del2_div_icell1 - vel_del2_div_icell0) * dc_edge_inv -
         (vel_del2_rvort_ivertex1 - vel_del2_rvort_ivertex0) * dv_edge_inv;

    Vec edge_mask_iedge;
    edge_mask_iedge.copy_from(&m_edge_mask(iedge, kstart), VecTag());
    
    Vec vn_tend_iedge;
    vn_tend_iedge.copy_from(&vn_tend_edge(iedge, kstart), VecTag());
    vn_tend_iedge -= edge_mask_iedge * m_visc_del4 * m_mesh_scaling_del4(iedge) * del2u;
    vn_tend_iedge.copy_to(&vn_tend_edge(iedge, kstart), VecTag());
  }
#else
  KOKKOS_FUNCTION void compute_vel_del2(Int iedge, Int kchunk,
                                        const RealConst2d &div_cell,
                                        const RealConst2d &rvort_vertex) const {
    const Int kstart = kchunk * vector_length;
    const Int icell0 = m_cells_on_edge(iedge, 0);
    const Int icell1 = m_cells_on_edge(iedge, 1);

    const Int ivertex0 = m_vertices_on_edge(iedge, 0);
    const Int ivertex1 = m_vertices_on_edge(iedge, 1);

    const Real dc_edge_inv = 1._fp / m_dc_edge(iedge);
    const Real dv_edge_inv =
        1._fp / std::max(m_dv_edge(iedge), 0.25_fp * m_dc_edge(iedge)); // huh

    for (Int kvec = 0; kvec < vector_length; ++kvec) {
      const Int k = kstart + kvec;
      const Real del2u =
          ((div_cell(icell1, k) - div_cell(icell0, k)) * dc_edge_inv -
           (rvort_vertex(ivertex1, k) - rvort_vertex(ivertex0, k)) * dv_edge_inv);

      m_vel_del2_edge(iedge, k) = del2u;
    }
  }

  KOKKOS_FUNCTION void compute_vel_del2_rvort(Int ivertex, Int kchunk) const {
    const Int kstart = kchunk * vector_length;
    const Real inv_area_triangle = 1._fp / m_area_triangle(ivertex);

    Real rvort[vector_length] = {0};
    for (Int j = 0; j < 3; ++j) {
      const Int jedge = m_edges_on_vertex(ivertex, j);
      for (Int kvec = 0; kvec < vector_length; ++kvec) {
        const Int k = kstart + kvec;
        rvort[kvec] += m_dc_edge(jedge) * inv_area_triangle * m_edge_sign_on_vertex(ivertex, j) *
                 m_vel_del2_edge(jedge, k);
      }
    }

    for (Int kvec = 0; kvec < vector_length; ++kvec) {
      const Int k = kstart + kvec;
      m_vel_del2_rvort_vertex(ivertex, k) = rvort[kvec];
    }
  }

  KOKKOS_FUNCTION void compute_vel_del2_div(Int icell, Int kchunk) const {
    const Int kstart = kchunk * vector_length;
    const Real inv_area_cell = 1._fp / m_area_cell(icell);

    Real accum[vector_length] = {0};
    for (Int j = 0; j < m_nedges_on_cell(icell); ++j) {
      const Int jedge = m_edges_on_cell(icell, j);
      for (Int kvec = 0; kvec < vector_length; ++kvec) {
        const Int k = kstart + kvec;
        accum[kvec] += m_dv_edge(jedge) * inv_area_cell * m_edge_sign_on_cell(icell, j) *
                 m_vel_del2_edge(jedge, k);
      }
    }
    
    for (Int kvec = 0; kvec < vector_length; ++kvec) {
      const Int k = kstart + kvec;
      m_vel_del2_div_cell(icell, k) = accum[kvec];
    }
  }

  KOKKOS_FUNCTION void operator()(const Real2d &vn_tend_edge, Int iedge,
                                  Int kchunk) const {
    const Int kstart = kchunk * vector_length;
    const Int icell0 = m_cells_on_edge(iedge, 0);
    const Int icell1 = m_cells_on_edge(iedge, 1);

    const Int ivertex0 = m_vertices_on_edge(iedge, 0);
    const Int ivertex1 = m_vertices_on_edge(iedge, 1);

    const Real dc_edge_inv = 1._fp / m_dc_edge(iedge);
    const Real dv_edge_inv = 1._fp / m_dv_edge(iedge);

    for (Int kvec = 0; kvec < vector_length; ++kvec) {
      const Int k = kstart + kvec;
      const Real del2u =
          ((m_vel_del2_div_cell(icell1, k) - m_vel_del2_div_cell(icell0, k)) *
               dc_edge_inv -
           (m_vel_del2_rvort_vertex(ivertex1, k) -
            m_vel_del2_rvort_vertex(ivertex0, k)) *
               dv_edge_inv);

      vn_tend_edge(iedge, k) -= m_edge_mask(iedge, k) * m_visc_del4 *
                                m_mesh_scaling_del4(iedge) * del2u;
    }
  }
#endif

  VelocityHyperDiffusionOnEdge(const MPASMesh *mesh, Real visc_del4)
      : m_vel_del2_edge("vel_del2_edge", mesh->m_nedges, mesh->m_nlayers),
        m_vel_del2_rvort_vertex("vel_del2_rvort_vertex", mesh->m_nvertices,
                                mesh->m_nlayers),
        m_vel_del2_div_cell("vel_del2_div_cell", mesh->m_ncells,
                            mesh->m_nlayers),
        m_cells_on_edge(mesh->m_cells_on_edge),
        m_vertices_on_edge(mesh->m_vertices_on_edge),
        m_edges_on_vertex(mesh->m_edges_on_vertex),
        m_nedges_on_cell(mesh->m_nedges_on_cell),
        m_edges_on_cell(mesh->m_edges_on_cell), m_dc_edge(mesh->m_dc_edge),
        m_dv_edge(mesh->m_dv_edge), m_area_cell(mesh->m_area_cell),
        m_area_triangle(mesh->m_area_triangle),
        m_mesh_scaling_del4(mesh->m_mesh_scaling_del4),
        m_edge_mask(mesh->m_edge_mask),
        m_edge_sign_on_vertex(mesh->m_edge_sign_on_vertex),
        m_edge_sign_on_cell(mesh->m_edge_sign_on_cell), m_visc_del4(visc_del4) {
  }
};

} // namespace omega
