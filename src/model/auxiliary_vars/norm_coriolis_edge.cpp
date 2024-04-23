#include "norm_coriolis_edge.hpp"
#include <model/shallow_water_auxstate.hpp>

namespace omega {

void NormCoriolisOnEdge::enable(ShallowWaterAuxiliaryState &aux_state) {
  m_enabled = true;
  aux_state.m_norm_f_vertex.enable(aux_state);
}

void NormCoriolisOnEdge::allocate(const MPASMesh *mesh) {
  if (m_enabled) {
    m_array = Real2d("norm_f_edge", mesh->m_nedges, mesh->m_nlayers);
  }
}
} // namespace omega
