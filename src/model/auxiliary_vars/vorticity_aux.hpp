#pragma once

#include <common.hpp>
#include <mesh/mpas_mesh.hpp>

namespace omega {

struct VorticityAuxVars {
  Real2d m_rvort_vertex;
  Real2d m_norm_rvort_vertex;
  Real2d m_norm_pvort_vertex;

  Real2d m_norm_rvort_edge;
  Real2d m_norm_pvort_edge;

  Int2d m_edges_on_vertex;
  Int2d m_cells_on_vertex;
  Int2d m_vertices_on_edge;

  Real2d m_edge_sign_on_vertex;
  Real1d m_dc_edge;
  Real1d m_area_triangle;
  Real2d m_kiteareas_on_vertex;

#ifdef OMEGA_KOKKOS_SIMD
  KOKKOS_FUNCTION void compute_vort_vertex(Int ivertex, Int kchunk,
                                           const RealConst2d &h_cell,
                                           const RealConst2d &vn_edge,
                                           const RealConst1d &f_vertex) const {
    const Int kstart = kchunk * vector_length;
    const Real inv_area_triangle = 1._fp / m_area_triangle(ivertex);

    Vec rvort_vertex = 0;
    Vec thick_vertex = 0;
    for (Int j = 0; j < 3; ++j) {
      const Int jedge = m_edges_on_vertex(ivertex, j);
      const Int jcell = m_cells_on_vertex(ivertex, j);

      Vec vn_edge_tmp;
      vn_edge_tmp.copy_from(&vn_edge(jedge, kstart), VecTag());

      Vec h_cell_tmp;
      h_cell_tmp.copy_from(&h_cell(jcell, kstart), VecTag());

      rvort_vertex += m_dc_edge(jedge) * inv_area_triangle *
                      m_edge_sign_on_vertex(ivertex, j) * vn_edge_tmp;
      thick_vertex +=
          m_kiteareas_on_vertex(ivertex, j) * inv_area_triangle * h_cell_tmp;
    }
    Vec inv_thick_vertex = 1._fp / thick_vertex;

    rvort_vertex.copy_to(&m_rvort_vertex(ivertex, kstart), VecTag());
    rvort_vertex *= inv_thick_vertex;
    rvort_vertex.copy_to(&m_norm_rvort_vertex(ivertex, kstart), VecTag());

    inv_thick_vertex *= f_vertex(ivertex);
    inv_thick_vertex.copy_to(&m_norm_pvort_vertex(ivertex, kstart), VecTag());
  }

  KOKKOS_FUNCTION void compute_vort_edge(Int iedge, Int kchunk) const {
    const Int kstart = kchunk * vector_length;
    const Int jvertex0 = m_vertices_on_edge(iedge, 0);
    const Int jvertex1 = m_vertices_on_edge(iedge, 1);

    Vec norm_rvort0;
    Vec norm_rvort1;

    norm_rvort0.copy_from(&m_norm_rvort_vertex(jvertex0, kstart), VecTag());
    norm_rvort1.copy_from(&m_norm_rvort_vertex(jvertex1, kstart), VecTag());

    norm_rvort0 = 0.5_fp * (norm_rvort0 + norm_rvort1);

    Vec norm_pvort0;
    Vec norm_pvort1;

    norm_pvort0.copy_from(&m_norm_pvort_vertex(jvertex0, kstart), VecTag());
    norm_pvort1.copy_from(&m_norm_pvort_vertex(jvertex1, kstart), VecTag());

    norm_pvort0 = 0.5_fp * (norm_pvort0 + norm_pvort1);

    norm_rvort0.copy_to(&m_norm_rvort_edge(iedge, kstart), VecTag());
    norm_pvort0.copy_to(&m_norm_pvort_edge(iedge, kstart), VecTag());
  }
#else
  KOKKOS_FUNCTION void compute_vort_vertex(Int ivertex, Int kchunk,
                                           const RealConst2d &h_cell,
                                           const RealConst2d &vn_edge,
                                           const RealConst1d &f_vertex) const {
    const Int kstart = kchunk * vector_length;
    const Real inv_area_triangle = 1._fp / m_area_triangle(ivertex);

    Real rvort_vertex[vector_length] = {0};
    Real thick_vertex[vector_length] = {0};
    for (Int j = 0; j < 3; ++j) {
      const Int jedge = m_edges_on_vertex(ivertex, j);
      const Int jcell = m_cells_on_vertex(ivertex, j);
      OMEGA_SIMD_PRAGMA
      for (Int kvec = 0; kvec < vector_length; ++kvec) {
        const Int k = kstart + kvec;
        rvort_vertex[kvec] += m_dc_edge(jedge) * inv_area_triangle *
                              m_edge_sign_on_vertex(ivertex, j) *
                              vn_edge(jedge, k);
        thick_vertex[kvec] += m_kiteareas_on_vertex(ivertex, j) *
                              inv_area_triangle * h_cell(jcell, k);
      }
    }

    OMEGA_SIMD_PRAGMA
    for (Int kvec = 0; kvec < vector_length; ++kvec) {
      const Real inv_thick_vertex = 1._fp / thick_vertex[kvec];
      const Int k = kstart + kvec;
      m_rvort_vertex(ivertex, k) = rvort_vertex[kvec];
      m_norm_rvort_vertex(ivertex, k) = inv_thick_vertex * rvort_vertex[kvec];
      m_norm_pvort_vertex(ivertex, k) = inv_thick_vertex * f_vertex(ivertex);
    }
  }

  KOKKOS_FUNCTION void compute_vort_edge(Int iedge, Int kchunk) const {
    const Int kstart = kchunk * vector_length;
    const Int jvertex0 = m_vertices_on_edge(iedge, 0);
    const Int jvertex1 = m_vertices_on_edge(iedge, 1);

    OMEGA_SIMD_PRAGMA
    for (Int kvec = 0; kvec < vector_length; ++kvec) {
      const Int k = kstart + kvec;
      m_norm_rvort_edge(iedge, k) = 0.5_fp * (m_norm_rvort_vertex(jvertex0, k) +
                                              m_norm_rvort_vertex(jvertex1, k));

      m_norm_pvort_edge(iedge, k) = 0.5_fp * (m_norm_pvort_vertex(jvertex0, k) +
                                              m_norm_pvort_vertex(jvertex1, k));
    }
  }
#endif

  VorticityAuxVars(const MPASMesh *mesh)
      : m_rvort_vertex("rvort_vertex", mesh->m_nvertices, mesh->m_nlayers),
        m_norm_rvort_vertex("norm_rvort_vertex", mesh->m_nvertices,
                            mesh->m_nlayers),
        m_norm_pvort_vertex("norm_pvort_vertex", mesh->m_nvertices,
                            mesh->m_nlayers),
        m_norm_rvort_edge("norm_rvort_edge", mesh->m_nedges, mesh->m_nlayers),
        m_norm_pvort_edge("norm_pvort_edge", mesh->m_nedges, mesh->m_nlayers),
        m_edges_on_vertex(mesh->m_edges_on_vertex),
        m_cells_on_vertex(mesh->m_cells_on_vertex),
        m_vertices_on_edge(mesh->m_vertices_on_edge),
        m_edge_sign_on_vertex(mesh->m_edge_sign_on_vertex),
        m_dc_edge(mesh->m_dc_edge), m_area_triangle(mesh->m_area_triangle),
        m_kiteareas_on_vertex(mesh->m_kiteareas_on_vertex) {}
};
} // namespace omega
