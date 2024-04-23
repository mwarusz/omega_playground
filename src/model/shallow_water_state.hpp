#pragma once

#include "shallow_water_params.hpp"
#include <common.hpp>
#include <mesh/mpas_mesh.hpp>

namespace omega {

struct ShallowWaterState {
  Real2d m_h_cell;
  Real2d m_vn_edge;
  Real3d m_tr_cell;

  ShallowWaterState(const MPASMesh *mesh, Int ntracers);
  ShallowWaterState(const MPASMesh *mesh, const ShallowWaterParams &params);
};
} // namespace omega
