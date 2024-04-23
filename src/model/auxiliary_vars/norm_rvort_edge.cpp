#include "norm_rvort_edge.hpp"
#include <model/shallow_water_auxstate.hpp>

namespace omega {

void NormRelVortOnEdge::enable(ShallowWaterAuxiliaryState &aux_state) {
  m_enabled = true;
  aux_state.m_norm_rvort_vertex.enable(aux_state);
}

void NormRelVortOnEdge::allocate(const MPASMesh *mesh) {
  if (m_enabled) {
    m_array = Real2d("norm_rvort_edge", mesh->m_nedges, mesh->m_nlayers);
  }
}
} // namespace omega
