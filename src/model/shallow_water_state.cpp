#include "shallow_water_state.hpp"
#include "shallow_water_model.hpp"

namespace omega {

// State
ShallowWaterState::ShallowWaterState(const MPASMesh *mesh, Int ntracers)
    : m_h_cell("h_cell", mesh->m_ncells, mesh->m_nlayers),
      m_vn_edge("vn_edge", mesh->m_nedges, mesh->m_nlayers),
      m_tr_cell("tr_cell", ntracers, mesh->m_ncells, mesh->m_nlayers) {}

ShallowWaterState::ShallowWaterState(const MPASMesh *mesh,
                                     const ShallowWaterParams &params)
    : ShallowWaterState(mesh, params.m_ntracers) {}

} // namespace omega
