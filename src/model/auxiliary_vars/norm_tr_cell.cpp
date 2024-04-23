#include "norm_tr_cell.hpp"
#include <model/shallow_water_auxstate.hpp>

namespace omega {

void NormTracerOnCell::enable(ShallowWaterAuxiliaryState &aux_state) {
  m_enabled = true;
}

void NormTracerOnCell::allocate(const MPASMesh *mesh, Int ntracers) {
  if (m_enabled) {
    m_array = Real3d("norm_tr_cell", ntracers, mesh->m_ncells, mesh->m_nlayers);
  }
}
} // namespace omega
