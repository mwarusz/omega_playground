#include "shallow_water_auxstate.hpp"

namespace omega {

ShallowWaterAuxiliaryState::ShallowWaterAuxiliaryState(const MPASMesh *mesh,
                                                       Int ntracers)
    : m_thickness_aux(mesh), m_vorticity_aux(mesh), m_kinetic_aux(mesh),
      m_tracer_aux(mesh, ntracers) {}

void ShallowWaterAuxiliaryState::allocate(const MPASMesh *mesh, Int ntracers) {}

void ShallowWaterAuxiliaryState::compute(RealConst2d h_cell,
                                         RealConst2d vn_edge,
                                         RealConst3d tr_cell,
                                         RealConst1d f_vertex,
                                         const MPASMesh *mesh) const {

  const auto &thickness_aux = m_thickness_aux;
  const auto &vorticity_aux = m_vorticity_aux;
  const auto &kinetic_aux = m_kinetic_aux;
  const auto &tracer_aux = m_tracer_aux;

  omega_parallel_for(
      "compute_vertex_auxiliarys", {mesh->m_nvertices, mesh->m_nlayers_vec},
      KOKKOS_LAMBDA(Int ivertex, Int k) {
        vorticity_aux.compute_vort_vertex(ivertex, k, h_cell, vn_edge,
                                          f_vertex);
      });

  const Int ntracers = tr_cell.extent(0);

  omega_parallel_for(
      "compute_cell_auxiliarys", {mesh->m_ncells, mesh->m_nlayers_vec},
      KOKKOS_LAMBDA(Int icell, Int k) {
        kinetic_aux.compute_kinetic_cell(icell, k, vn_edge);
        for (Int l = 0; l < ntracers; ++l) {
          tracer_aux.compute_norm_tracer_cell(l, icell, k, tr_cell, h_cell);
        }
      });

  const auto rvort_vertex = const_view(vorticity_aux.m_rvort_vertex);
  const auto vel_div_cell = const_view(kinetic_aux.m_vel_div_cell);

  omega_parallel_for(
      "compute_edge_auxiliarys", {mesh->m_nedges, mesh->m_nlayers_vec},
      KOKKOS_LAMBDA(Int iedge, Int k) {
        thickness_aux.compute_thickness_edge(iedge, k, h_cell);
        vorticity_aux.compute_vort_edge(iedge, k);
      });
}
} // namespace omega
