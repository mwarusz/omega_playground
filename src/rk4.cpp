#include <time_stepper.hpp>

#include <string>
using namespace std::string_literals;

namespace omega {

RK4Stepper::RK4Stepper(ShallowWaterBase &shallow_water)
    : TimeStepper(shallow_water), rka(nstages - 1), rkb(nstages),
      rkc(nstages - 1), tend(*shallow_water.mesh),
      provis_state(*shallow_water.mesh), old_state(*shallow_water.mesh) {

  rka[0] = 1. / 2;
  rka[1] = 1. / 2;
  rka[2] = 1;

  rkb[0] = 1. / 6;
  rkb[1] = 1. / 3;
  rkb[2] = 1. / 3;
  rkb[3] = 1. / 6;

  rkc[0] = 1. / 2;
  rkc[1] = 1. / 2;
  rkc[2] = 1;
}

void RK4Stepper::do_step(Real t, Real dt,
                         const ShallowWaterState &state) const {
  auto mesh = shallow_water->mesh;

  YAKL_SCOPE(h_old_cell, this->old_state.h_cell);
  YAKL_SCOPE(vn_old_edge, this->old_state.vn_edge);

  YAKL_SCOPE(h_tend_cell, this->tend.h_cell);
  YAKL_SCOPE(vn_tend_edge, this->tend.vn_edge);

  YAKL_SCOPE(h_provis_cell, this->provis_state.h_cell);
  YAKL_SCOPE(vn_provis_edge, this->provis_state.vn_edge);

  state.h_cell.deep_copy_to(h_old_cell);
  state.vn_edge.deep_copy_to(vn_old_edge);

  // k1
  shallow_water->compute_tendency(tend, state, t);

  for (Int stage = 0; stage < nstages; ++stage) {

    const Real rkb_stage = rkb[stage];
    parallel_for(
        "rk4_accumulate_h", SimpleBounds<2>(mesh->ncells, mesh->nlayers),
        YAKL_LAMBDA(Int icell, Int k) {
          state.h_cell(icell, k) += dt * rkb_stage * h_tend_cell(icell, k);
        });
    parallel_for(
        "rk4_accumulate_v", SimpleBounds<2>(mesh->nedges, mesh->nlayers),
        YAKL_LAMBDA(Int iedge, Int k) {
          state.vn_edge(iedge, k) += dt * rkb_stage * vn_tend_edge(iedge, k);
        });

    if (stage < nstages - 1) {
      Real stagetime = t + rkc[stage] * dt;
      const Real rka_stage = rka[stage];

      parallel_for(
          "rk4_compute_hprovis", SimpleBounds<2>(mesh->ncells, mesh->nlayers),
          YAKL_LAMBDA(Int icell, Int k) {
            h_provis_cell(icell, k) =
                h_old_cell(icell, k) + dt * rka_stage * h_tend_cell(icell, k);
          });
      parallel_for(
          "rk4_compute_vprovis", SimpleBounds<2>(mesh->nedges, mesh->nlayers),
          YAKL_LAMBDA(Int iedge, Int k) {
            vn_provis_edge(iedge, k) =
                vn_old_edge(iedge, k) + dt * rka_stage * vn_tend_edge(iedge, k);
          });

      shallow_water->compute_tendency(tend, provis_state, stagetime);
    }
  }
}
} // namespace omega
