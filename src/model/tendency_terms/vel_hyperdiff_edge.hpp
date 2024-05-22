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

  KOKKOS_FUNCTION void compute_vel_del2(Int iedge, Int k,
                                        const RealConst2d &div_cell,
                                        const RealConst2d &rvort_vertex) const {
    const Int icell0 = m_cells_on_edge(iedge, 0);
    const Int icell1 = m_cells_on_edge(iedge, 1);

    const Int ivertex0 = m_vertices_on_edge(iedge, 0);
    const Int ivertex1 = m_vertices_on_edge(iedge, 1);

    const Real dc_edge_inv = 1._fp / m_dc_edge(iedge);
    const Real dv_edge_inv =
        1._fp / std::max(m_dv_edge(iedge), 0.25_fp * m_dc_edge(iedge)); // huh

    const Real del2u =
        ((div_cell(icell1, k) - div_cell(icell0, k)) * dc_edge_inv -
         (rvort_vertex(ivertex1, k) - rvort_vertex(ivertex0, k)) * dv_edge_inv);

    m_vel_del2_edge(iedge, k) = del2u;
  }

  KOKKOS_FUNCTION void compute_vel_del2_rvort(Int ivertex, Int k) const {
    Real rvort = -0;
    for (Int j = 0; j < 3; ++j) {
      Int jedge = m_edges_on_vertex(ivertex, j);
      rvort += m_dc_edge(jedge) * m_edge_sign_on_vertex(ivertex, j) *
               m_vel_del2_edge(jedge, k);
    }
    const Real inv_area_triangle = 1._fp / m_area_triangle(ivertex);
    rvort *= inv_area_triangle;

    m_vel_del2_rvort_vertex(ivertex, k) = rvort;
  }

  KOKKOS_FUNCTION void compute_vel_del2_div(Int icell, Int k) const {
    Real accum = 0;
    for (Int j = 0; j < m_nedges_on_cell(icell); ++j) {
      Int jedge = m_edges_on_cell(icell, j);
      accum += m_dv_edge(jedge) * m_edge_sign_on_cell(icell, j) *
               m_vel_del2_edge(jedge, k);
    }
    Real inv_area_cell = 1._fp / m_area_cell(icell);
    m_vel_del2_div_cell(icell, k) = accum * inv_area_cell;
  }

  KOKKOS_FUNCTION Real operator()(Int iedge, Int k) const {
    const Int icell0 = m_cells_on_edge(iedge, 0);
    const Int icell1 = m_cells_on_edge(iedge, 1);

    const Int ivertex0 = m_vertices_on_edge(iedge, 0);
    const Int ivertex1 = m_vertices_on_edge(iedge, 1);

    const Real dc_edge_inv = 1._fp / m_dc_edge(iedge);
    const Real dv_edge_inv = 1._fp / m_dv_edge(iedge);

    const Real del2u =
        ((m_vel_del2_div_cell(icell1, k) - m_vel_del2_div_cell(icell0, k)) *
             dc_edge_inv -
         (m_vel_del2_rvort_vertex(ivertex1, k) -
          m_vel_del2_rvort_vertex(ivertex0, k)) *
             dv_edge_inv);

    return -m_edge_mask(iedge, k) * m_visc_del4 * m_mesh_scaling_del4(iedge) *
           del2u;
  }

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
