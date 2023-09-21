#include <time_stepper.hpp>

#include <string>
using namespace std::string_literals;

namespace omega {

RK4Stepper::RK4Stepper(ShallowWaterModelBase &shallow_water)
    : TimeStepper(shallow_water), m_rka(nstages - 1), m_rkb(nstages),
      m_rkc(nstages - 1), m_tend(shallow_water), m_provis_state(shallow_water),
      m_old_state(shallow_water) {

  m_rka[0] = 1. / 2;
  m_rka[1] = 1. / 2;
  m_rka[2] = 1;

  m_rkb[0] = 1. / 6;
  m_rkb[1] = 1. / 3;
  m_rkb[2] = 1. / 3;
  m_rkb[3] = 1. / 6;

  m_rkc[0] = 1. / 2;
  m_rkc[1] = 1. / 2;
  m_rkc[2] = 1;
}

void RK4Stepper::do_step(Real t, Real dt,
                         const ShallowWaterState &state) const {
  const auto &mesh = m_shallow_water->m_mesh;

  YAKL_SCOPE(h_old_cell, m_old_state.m_h_cell);
  YAKL_SCOPE(vn_old_edge, m_old_state.m_vn_edge);
  YAKL_SCOPE(tr_old_cell, m_old_state.m_tr_cell);

  YAKL_SCOPE(h_tend_cell, m_tend.m_h_cell);
  YAKL_SCOPE(vn_tend_edge, m_tend.m_vn_edge);
  YAKL_SCOPE(tr_tend_cell, m_tend.m_tr_cell);

  YAKL_SCOPE(h_provis_cell, m_provis_state.m_h_cell);
  YAKL_SCOPE(vn_provis_edge, m_provis_state.m_vn_edge);
  YAKL_SCOPE(tr_provis_cell, m_provis_state.m_tr_cell);
  Int ntracers = m_shallow_water->m_ntracers;

  state.m_h_cell.deep_copy_to(h_old_cell);
  state.m_vn_edge.deep_copy_to(vn_old_edge);
  state.m_tr_cell.deep_copy_to(tr_old_cell);

  // k1
  m_shallow_water->compute_tendency(m_tend, state, t);

  for (Int stage = 0; stage < nstages; ++stage) {

    const Real rkb_stage = m_rkb[stage];
    parallel_for(
        "rk4_accumulate_h", SimpleBounds<2>(mesh->m_ncells, mesh->m_nlayers),
        YAKL_LAMBDA(Int icell, Int k) {
          state.m_h_cell(icell, k) += dt * rkb_stage * h_tend_cell(icell, k);
        });
    parallel_for(
        "rk4_accumulate_v", SimpleBounds<2>(mesh->m_nedges, mesh->m_nlayers),
        YAKL_LAMBDA(Int iedge, Int k) {
          state.m_vn_edge(iedge, k) += dt * rkb_stage * vn_tend_edge(iedge, k);
        });
    parallel_for(
        "rk4_accumulate_tr",
        SimpleBounds<3>(ntracers, mesh->m_ncells, mesh->m_nlayers),
        YAKL_LAMBDA(Int l, Int icell, Int k) {
          state.m_tr_cell(l, icell, k) +=
              dt * rkb_stage * tr_tend_cell(l, icell, k);
        });

    if (stage < nstages - 1) {
      Real stagetime = t + m_rkc[stage] * dt;
      const Real rka_stage = m_rka[stage];

      parallel_for(
          "rk4_compute_h_provis",
          SimpleBounds<2>(mesh->m_ncells, mesh->m_nlayers),
          YAKL_LAMBDA(Int icell, Int k) {
            h_provis_cell(icell, k) =
                h_old_cell(icell, k) + dt * rka_stage * h_tend_cell(icell, k);
          });
      parallel_for(
          "rk4_compute_vn_provis",
          SimpleBounds<2>(mesh->m_nedges, mesh->m_nlayers),
          YAKL_LAMBDA(Int iedge, Int k) {
            vn_provis_edge(iedge, k) =
                vn_old_edge(iedge, k) + dt * rka_stage * vn_tend_edge(iedge, k);
          });
      parallel_for(
          "rk4_compute_tr_provis",
          SimpleBounds<3>(ntracers, mesh->m_ncells, mesh->m_nlayers),
          YAKL_LAMBDA(Int l, Int icell, Int k) {
            tr_provis_cell(l, icell, k) =
                tr_old_cell(l, icell, k) +
                dt * rka_stage * tr_tend_cell(l, icell, k);
          });

      m_shallow_water->compute_tendency(m_tend, m_provis_state, stagetime);
    }
  }
}
} // namespace omega
