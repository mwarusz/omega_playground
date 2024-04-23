#include "flux_thickness_edge.hpp"
#include <model/shallow_water_auxstate.hpp>

namespace omega {

void FluxThicknessOnEdge::enable(ShallowWaterAuxiliaryState &aux_state) {
  m_enabled = true;
}

void FluxThicknessOnEdge::allocate(const MPASMesh *mesh) {
  if (m_enabled) {
    m_array = Real2d("h_flux_edge", mesh->m_nedges, mesh->m_nlayers);
  }
}
} // namespace omega
