#include "tracer_del2_cell.hpp"
#include <model/shallow_water_auxstate.hpp>

namespace omega {

void TracerDel2OnCell::enable(ShallowWaterAuxiliaryState &aux_state) {
  m_enabled = true;
  aux_state.m_norm_tr_cell.enable(aux_state);
  aux_state.m_h_mean_edge.enable(aux_state);
}

void TracerDel2OnCell::allocate(const MPASMesh *mesh, Int ntracers) {
  if (m_enabled) {
    m_array = Real3d("tr_del2_cell", ntracers, mesh->m_ncells, mesh->m_nlayers);
  }
}
} // namespace omega
