#include "kinetic_energy_cell.hpp"
#include <model/shallow_water_auxstate.hpp>

namespace omega {

void KineticEnergyOnCell::enable(ShallowWaterAuxiliaryState &aux_state) {
  m_enabled = true;
}

void KineticEnergyOnCell::allocate(const MPASMesh *mesh) {
  if (m_enabled) {
    m_array = Real2d("ke_cell", mesh->m_ncells, mesh->m_nlayers);
  }
}
} // namespace omega
