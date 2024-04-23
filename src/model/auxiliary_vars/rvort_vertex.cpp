#include "rvort_vertex.hpp"
#include <model/shallow_water_auxstate.hpp>

namespace omega {

void RelVortOnVertex::enable(ShallowWaterAuxiliaryState &aux_state) {
  m_enabled = true;
}

void RelVortOnVertex::allocate(const MPASMesh *mesh) {
  if (m_enabled) {
    m_array = Real2d("rvort_vertex", mesh->m_nvertices, mesh->m_nlayers);
  }
}
} // namespace omega
