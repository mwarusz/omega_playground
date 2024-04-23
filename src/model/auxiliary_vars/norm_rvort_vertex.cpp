#include "norm_rvort_vertex.hpp"
#include <model/shallow_water_auxstate.hpp>

namespace omega {

void NormRelVortOnVertex::enable(ShallowWaterAuxiliaryState &aux_state) {
  m_enabled = true;
  aux_state.m_rvort_vertex.enable(aux_state);
}

void NormRelVortOnVertex::allocate(const MPASMesh *mesh) {
  if (m_enabled) {
    m_array = Real2d("norm_rvort_vertex", mesh->m_nvertices, mesh->m_nlayers);
  }
}
} // namespace omega
