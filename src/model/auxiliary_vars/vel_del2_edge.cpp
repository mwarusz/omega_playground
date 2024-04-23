#include "norm_rvort_edge.hpp"
#include <model/shallow_water_auxstate.hpp>

namespace omega {

void VelocityDel2OnEdge::enable(ShallowWaterAuxiliaryState &aux_state) {
  m_enabled = true;
  aux_state.m_vel_div_cell.enable(aux_state);
  aux_state.m_rvort_vertex.enable(aux_state);
}

void VelocityDel2OnEdge::allocate(const MPASMesh *mesh) {
  if (m_enabled) {
    m_array = Real2d("vel_del2_edge", mesh->m_nedges, mesh->m_nlayers);
  }
}
} // namespace omega
