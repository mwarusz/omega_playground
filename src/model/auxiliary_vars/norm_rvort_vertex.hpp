#pragma once

#include <common.hpp>
#include <mesh/mpas_mesh.hpp>

namespace omega {

struct ShallowWaterAuxiliaryState;

struct NormRelVortOnVertex {
  bool m_enabled = false;
  Real2d m_array;

  Int2d m_cells_on_vertex;
  Real2d m_kiteareas_on_vertex;
  Real1d m_area_triangle;

  void enable(ShallowWaterAuxiliaryState &aux_state);
  void allocate(const MPASMesh *mesh);
  RealConst2d const_array() const { return m_array; }

  KOKKOS_FUNCTION Real thickness_vertex(Int ivertex, Int k,
                                        const RealConst2d &h_cell) const {
    Real h = -0;
    Real inv_area_triangle = 1._fp / m_area_triangle(ivertex);
    for (Int j = 0; j < 3; ++j) {
      Int jcell = m_cells_on_vertex(ivertex, j);
      h += m_kiteareas_on_vertex(ivertex, j) * h_cell(jcell, k);
    }
    h *= inv_area_triangle;
    return h;
  }

  KOKKOS_FUNCTION Real operator()(Int ivertex, Int k,
                                  const RealConst2d &rvort_vertex,
                                  const RealConst2d &h_cell) const {
    const Real inv_h_vertex = 1. / thickness_vertex(ivertex, k, h_cell);
    return rvort_vertex(ivertex, k) * inv_h_vertex;
  }

  NormRelVortOnVertex(const MPASMesh *mesh)
      : m_cells_on_vertex(mesh->m_cells_on_vertex),
        m_kiteareas_on_vertex(mesh->m_kiteareas_on_vertex),
        m_area_triangle(mesh->m_area_triangle) {}
};
} // namespace omega
