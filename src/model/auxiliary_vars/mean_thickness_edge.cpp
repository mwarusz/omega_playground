#include "mean_thickness_edge.hpp"
#include <model/shallow_water_auxstate.hpp>

namespace omega {

void MeanThicknessOnEdge::enable(ShallowWaterAuxiliaryState &aux_state) {
  m_enabled = true;
}

void MeanThicknessOnEdge::allocate(const MPASMesh *mesh) {
  if (m_enabled) {
    m_array = Real2d("h_mean_edge", mesh->m_nedges, mesh->m_nlayers);
  }
}
} // namespace omega
