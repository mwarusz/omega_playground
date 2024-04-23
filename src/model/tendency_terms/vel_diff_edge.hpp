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

  void enable(ShallowWaterAuxiliaryState &aux_state) {
    m_enabled = true;
    aux_state.m_vel_div_cell.enable(aux_state);
    aux_state.m_rvort_vertex.enable(aux_state);
  }

  KOKKOS_FUNCTION Real operator()(Int iedge, Int k, const RealConst2d &div_cell,
                                  const RealConst2d &rvort_vertex) const {
    const Int icell0 = m_cells_on_edge(iedge, 0);
    const Int icell1 = m_cells_on_edge(iedge, 1);

    const Int ivertex0 = m_vertices_on_edge(iedge, 0);
    const Int ivertex1 = m_vertices_on_edge(iedge, 1);

    const Real dc_edge_inv = 1._fp / m_dc_edge(iedge);
    const Real dv_edge_inv = 1._fp / m_dv_edge(iedge);

    const Real del2u =
        ((div_cell(icell1, k) - div_cell(icell0, k)) * dc_edge_inv -
         (rvort_vertex(ivertex1, k) - rvort_vertex(ivertex0, k)) * dv_edge_inv);

    return m_edge_mask(iedge, k) * m_visc_del2 * m_mesh_scaling_del2(iedge) *
           del2u;
  }

  VelocityDiffusionOnEdge(const MPASMesh *mesh, Real visc_del2)
      : m_cells_on_edge(mesh->m_cells_on_edge),
        m_vertices_on_edge(mesh->m_vertices_on_edge),
        m_dc_edge(mesh->m_dc_edge), m_dv_edge(mesh->m_dv_edge),
        m_mesh_scaling_del2(mesh->m_mesh_scaling_del2),
        m_edge_mask(mesh->m_edge_mask), m_visc_del2(visc_del2) {}
};

} // namespace omega
