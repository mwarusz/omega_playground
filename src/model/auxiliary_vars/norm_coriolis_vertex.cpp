#include "norm_coriolis_vertex.hpp"
#include <model/shallow_water_auxstate.hpp>

namespace omega {

void NormCoriolisOnVertex::enable(ShallowWaterAuxiliaryState &aux_state) {
  m_enabled = true;
}

void NormCoriolisOnVertex::allocate(const MPASMesh *mesh) {
  if (m_enabled) {
    m_array = Real2d("norm_f_vertex", mesh->m_nvertices, mesh->m_nlayers);
  }
}
} // namespace omega
