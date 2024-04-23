#include "vel_div_cell.hpp"
#include <model/shallow_water_auxstate.hpp>

namespace omega {

void VelDivOnCell::enable(ShallowWaterAuxiliaryState &aux_state) {
  m_enabled = true;
}

void VelDivOnCell::allocate(const MPASMesh *mesh) {
  if (m_enabled) {
    m_array = Real2d("vel_div_cell", mesh->m_ncells, mesh->m_nlayers);
  }
}
} // namespace omega
