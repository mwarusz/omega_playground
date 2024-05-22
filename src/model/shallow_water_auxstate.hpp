#pragma once

#include "auxiliary_vars/auxiliary_vars.hpp"
#include <common.hpp>
#include <mesh/mpas_mesh.hpp>

namespace omega {

struct ShallowWaterAuxiliaryState {
  ThicknessAuxVars m_thickness_aux;
  VorticityAuxVars m_vorticity_aux;
  KineticAuxVars m_kinetic_aux;
  TracerAuxVars m_tracer_aux;

  ShallowWaterAuxiliaryState(const MPASMesh *mesh, Int ntracers);

  void allocate(const MPASMesh *mesh, Int ntracers);

  void compute(RealConst2d h_cell, RealConst2d vn_edge, RealConst3d tr_cell,
               RealConst1d f_vertex, const MPASMesh *mesh) const;
};
} // namespace omega
