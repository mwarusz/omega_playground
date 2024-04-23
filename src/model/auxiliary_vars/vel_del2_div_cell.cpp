#include "vel_del2_div_cell.hpp"
#include <model/shallow_water_auxstate.hpp>

namespace omega {

void VelDel2DivOnCell::enable(ShallowWaterAuxiliaryState &aux_state) {
  m_enabled = true;
  aux_state.m_vel_del2_edge.enable(aux_state);
}

void VelDel2DivOnCell::allocate(const MPASMesh *mesh) {
  if (m_enabled) {
    m_array = Real2d("vel_del2_div_cell", mesh->m_ncells, mesh->m_nlayers);
  }
}
} // namespace omega
