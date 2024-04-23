#pragma once

#include <common.hpp>
#include <mesh/mpas_mesh.hpp>

namespace omega {

struct ShallowWaterAuxiliaryState;

struct VelocityDel2OnEdge {
  bool m_enabled = false;
  Real2d m_array;

  Int2d m_cells_on_edge;
  Int2d m_vertices_on_edge;
  Real1d m_dc_edge;
  Real1d m_dv_edge;

  void enable(ShallowWaterAuxiliaryState &aux_state);
  void allocate(const MPASMesh *mesh);
  RealConst2d const_array() const { return m_array; }

  KOKKOS_FUNCTION Real operator()(Int iedge, Int k, const RealConst2d &div_cell,
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

    return del2u;
  }

  VelocityDel2OnEdge(const MPASMesh *mesh)
      : m_cells_on_edge(mesh->m_cells_on_edge),
        m_vertices_on_edge(mesh->m_vertices_on_edge),
        m_dc_edge(mesh->m_dc_edge), m_dv_edge(mesh->m_dv_edge) {}
};
} // namespace omega
