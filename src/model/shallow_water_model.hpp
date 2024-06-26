#pragma once

#include "shallow_water_auxstate.hpp"
#include "shallow_water_params.hpp"
#include "shallow_water_state.hpp"
#include "tendency_terms/tendency_terms.hpp"
#include <common.hpp>
#include <mesh/planar_hexagonal_mesh.hpp>

namespace omega {

enum class AddMode { replace, increment };

struct ShallowWaterModel {
  ShallowWaterParams m_params;

  MPASMesh *m_mesh;
  ShallowWaterAuxiliaryState m_aux_state;

  ThicknessHorzAdvOnCell m_thickness_hadv_cell;

  PotentialVortFluxOnEdge m_pv_flux_edge;
  KineticEnergyGradOnEdge m_ke_grad_edge;
  SSHGradOnEdge m_ssh_grad_edge;
  VelocityDiffusionOnEdge m_vel_diff_edge;
  VelocityHyperDiffusionOnEdge m_vel_hyperdiff_edge;

  TracerHorzAdvOnCell m_tracer_hadv_cell;
  TracerDiffusionOnCell m_tracer_diff_cell;
  TracerHyperDiffusionOnCell m_tracer_hyperdiff_cell;

  Real1d m_f_vertex;
  Real1d m_f_edge;

  void compute_h_tendency(Real2d h_tend_cell, RealConst2d h_cell,
                          RealConst2d vn_edge, AddMode add_mode) const;
  void compute_vn_tendency(Real2d vn_tend_edge, RealConst2d h_cell,
                           RealConst2d vn_edge, AddMode add_mode) const;
  void compute_tr_tendency(Real3d tr_tend_cell, RealConst3d tr_cell,
                           RealConst2d vn_edge, AddMode add_mode) const;

  virtual void additional_tendency(Real2d h_tend_cell, Real2d vn_tend_edge,
                                   RealConst2d h_cell, RealConst2d vn_edge,
                                   Real t) const {}

  void compute_tendency(const ShallowWaterState &tend,
                        const ShallowWaterState &state, Real t,
                        AddMode add_mode = AddMode::replace) const;

  ShallowWaterModel(MPASMesh *mesh, const ShallowWaterParams &params);
};

} // namespace omega
