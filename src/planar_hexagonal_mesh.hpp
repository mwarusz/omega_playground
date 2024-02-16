#pragma once

#include <common.hpp>
#include <mpas_mesh.hpp>

namespace omega {

struct PlanarHexagonalMesh : MPASMesh {
  static constexpr Int maxedges = 6;

  Int m_nx;
  Int m_ny;
  Real m_dc;
  Real m_period_x;
  Real m_period_y;

  PlanarHexagonalMesh(Int nx, Int ny, Int nlayers = 1);
  PlanarHexagonalMesh(Int nx, Int ny, Real dc, Int nlayers = 1);

  KOKKOS_INLINE_FUNCTION Int cellidx(Int icol, Int irow) const;
  KOKKOS_INLINE_FUNCTION Int cell_on_cell(Int icol, Int irow, Int nb) const;
  KOKKOS_INLINE_FUNCTION Int edge_on_cell(Int icell, Int icol, Int irow,
                                          Int nb) const;
  KOKKOS_INLINE_FUNCTION Int vertex_on_cell(Int icell, Int icol, Int irow,
                                            Int nb) const;

  void compute_mesh_arrays();
};
} // namespace omega
