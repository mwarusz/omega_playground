#pragma once

#include "auxiliary_vars/auxiliary_vars.hpp"
#include <common.hpp>
#include <mesh/mpas_mesh.hpp>

namespace omega {

struct ShallowWaterAuxiliaryState {
  KineticEnergyOnCell m_ke_cell;
  VelDivOnCell m_vel_div_cell;
  VelDel2DivOnCell m_vel_del2_div_cell;
  NormTracerOnCell m_norm_tr_cell;
  TracerDel2OnCell m_tr_del2_cell;

  FluxThicknessOnEdge m_h_flux_edge;
  MeanThicknessOnEdge m_h_mean_edge;
  DragThicknessOnEdge m_h_drag_edge;
  NormRelVortOnEdge m_norm_rvort_edge;
  NormCoriolisOnEdge m_norm_f_edge;
  VelocityDel2OnEdge m_vel_del2_edge;

  RelVortOnVertex m_rvort_vertex;
  VelDel2RelVortOnVertex m_vel_del2_rvort_vertex;
  NormRelVortOnVertex m_norm_rvort_vertex;
  NormCoriolisOnVertex m_norm_f_vertex;

  ShallowWaterAuxiliaryState(const MPASMesh *mesh);

  void allocate(const MPASMesh *mesh, Int ntracers);

  void compute(RealConst2d h_cell, RealConst2d vn_edge, RealConst3d tr_cell,
               RealConst1d f_vertex, const MPASMesh *mesh) const;
};
} // namespace omega
