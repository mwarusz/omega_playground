#include "time_stepper.hpp"

#include <string>
using namespace std::string_literals;

namespace omega {

RK4Stepper::RK4Stepper(ShallowWaterModel &shallow_water)
    : TimeStepper(shallow_water), m_rka(nstages - 1), m_rkb(nstages),
      m_rkc(nstages - 1), m_tend(shallow_water.m_mesh, shallow_water.m_params),
      m_provis_state(shallow_water.m_mesh, shallow_water.m_params),
      m_old_state(shallow_water.m_mesh, shallow_water.m_params) {

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

  OMEGA_SCOPE(h_old_cell, m_old_state.m_h_cell);
  OMEGA_SCOPE(vn_old_edge, m_old_state.m_vn_edge);
  OMEGA_SCOPE(tr_old_cell, m_old_state.m_tr_cell);

  OMEGA_SCOPE(h_tend_cell, m_tend.m_h_cell);
  OMEGA_SCOPE(vn_tend_edge, m_tend.m_vn_edge);
  OMEGA_SCOPE(tr_tend_cell, m_tend.m_tr_cell);

  OMEGA_SCOPE(h_provis_cell, m_provis_state.m_h_cell);
  OMEGA_SCOPE(vn_provis_edge, m_provis_state.m_vn_edge);
  OMEGA_SCOPE(tr_provis_cell, m_provis_state.m_tr_cell);
  Int ntracers = m_shallow_water->m_params.m_ntracers;

  deep_copy(h_old_cell, state.m_h_cell);
  deep_copy(vn_old_edge, state.m_vn_edge);
  deep_copy(tr_old_cell, state.m_tr_cell);

  // k1
  m_shallow_water->compute_tendency(m_tend, state, t);

  for (Int stage = 0; stage < nstages; ++stage) {
 
    timer_start("rk1");
    const Real rkb_stage = m_rkb[stage];
    omega_parallel_for(
        "rk4_accumulate_h", {mesh->m_ncells, mesh->m_nlayers},
        KOKKOS_LAMBDA(Int icell, Int k) {
          state.m_h_cell(icell, k) += dt * rkb_stage * h_tend_cell(icell, k);
        });
    omega_parallel_for(
        "rk4_accumulate_v", {mesh->m_nedges, mesh->m_nlayers},
        KOKKOS_LAMBDA(Int iedge, Int k) {
          state.m_vn_edge(iedge, k) += dt * rkb_stage * vn_tend_edge(iedge, k);
        });
    if (ntracers > 0) {
      omega_parallel_for(
          "rk4_accumulate_tr", {ntracers, mesh->m_ncells, mesh->m_nlayers},
          KOKKOS_LAMBDA(Int l, Int icell, Int k) {
            state.m_tr_cell(l, icell, k) +=
                dt * rkb_stage * tr_tend_cell(l, icell, k);
          });
    }
    timer_stop("rk1");

    if (stage < nstages - 1) {
      Real stagetime = t + m_rkc[stage] * dt;
      const Real rka_stage = m_rka[stage];

      timer_start("rk2");
      omega_parallel_for(
          "rk4_compute_h_provis", {mesh->m_ncells, mesh->m_nlayers},
          KOKKOS_LAMBDA(Int icell, Int k) {
            h_provis_cell(icell, k) =
                h_old_cell(icell, k) + dt * rka_stage * h_tend_cell(icell, k);
          });
      omega_parallel_for(
          "rk4_compute_vn_provis", {mesh->m_nedges, mesh->m_nlayers},
          KOKKOS_LAMBDA(Int iedge, Int k) {
            vn_provis_edge(iedge, k) =
                vn_old_edge(iedge, k) + dt * rka_stage * vn_tend_edge(iedge, k);
          });
      if (ntracers > 0) {
        omega_parallel_for(
            "rk4_compute_tr_provis",
            {ntracers, mesh->m_ncells, mesh->m_nlayers},
            KOKKOS_LAMBDA(Int l, Int icell, Int k) {
              tr_provis_cell(l, icell, k) =
                  tr_old_cell(l, icell, k) +
                  dt * rka_stage * tr_tend_cell(l, icell, k);
            });
      }
      timer_stop("rk2");

      m_shallow_water->compute_tendency(m_tend, m_provis_state, stagetime);
    }
  }
}
} // namespace omega
