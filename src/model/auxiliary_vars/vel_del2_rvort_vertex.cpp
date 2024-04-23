#include "vel_del2_rvort_vertex.hpp"
#include <model/shallow_water_auxstate.hpp>

namespace omega {

void VelDel2RelVortOnVertex::enable(ShallowWaterAuxiliaryState &aux_state) {
  m_enabled = true;
  aux_state.m_vel_del2_edge.enable(aux_state);
}

void VelDel2RelVortOnVertex::allocate(const MPASMesh *mesh) {
  if (m_enabled) {
    m_array =
        Real2d("vel_del2_rvort_vertex", mesh->m_nvertices, mesh->m_nlayers);
  }
}
} // namespace omega
