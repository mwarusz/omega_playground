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

  KOKKOS_FUNCTION void compute_vort_vertex(Int ivertex, Int k,
                                           const RealConst2d &h_cell,
                                           const RealConst2d &vn_edge,
                                           const RealConst1d &f_vertex) const {
    const Real inv_area_triangle = 1._fp / m_area_triangle(ivertex);

    Real rvort_vertex = 0;
    Real thick_vertex = 0;
    for (Int j = 0; j < 3; ++j) {
      const Int jedge = m_edges_on_vertex(ivertex, j);
      const Int jcell = m_cells_on_vertex(ivertex, j);
      rvort_vertex += m_dc_edge(jedge) * m_edge_sign_on_vertex(ivertex, j) *
                      vn_edge(jedge, k);
      thick_vertex += m_kiteareas_on_vertex(ivertex, j) * h_cell(jcell, k);
    }
    thick_vertex *= inv_area_triangle;
    rvort_vertex *= inv_area_triangle;

    const Real inv_thick_vertex = 1._fp / thick_vertex;

    m_rvort_vertex(ivertex, k) = rvort_vertex;
    m_norm_rvort_vertex(ivertex, k) = inv_thick_vertex * rvort_vertex;
    m_norm_pvort_vertex(ivertex, k) = inv_thick_vertex * f_vertex(ivertex);
  }

  KOKKOS_FUNCTION void compute_vort_edge(Int iedge, Int k) const {
    const Int jvertex0 = m_vertices_on_edge(iedge, 0);
    const Int jvertex1 = m_vertices_on_edge(iedge, 1);

    m_norm_rvort_edge(iedge, k) = 0.5_fp * (m_norm_rvort_vertex(jvertex0, k) +
                                            m_norm_rvort_vertex(jvertex1, k));

    m_norm_pvort_edge(iedge, k) = 0.5_fp * (m_norm_pvort_vertex(jvertex0, k) +
                                            m_norm_pvort_vertex(jvertex1, k));
  }

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
