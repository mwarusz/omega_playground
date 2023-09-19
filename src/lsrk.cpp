#include <time_stepper.hpp>

namespace omega {

LSRKStepper::LSRKStepper(ShallowWaterBase &shallow_water)
    : TimeStepper(shallow_water), rka(nstages), rkb(nstages), rkc(nstages),
      tend(*shallow_water.mesh) {

  yakl::memset(tend.h_cell, 0);
  yakl::memset(tend.vn_edge, 0);

  rka = {0., -567301805773. / 1357537059087., -2404267990393. / 2016746695238.,
         -3550918686646. / 2091501179385., -1275806237668. / 842570457699.};

  rkb = {1432997174477. / 9575080441755., 5161836677717. / 13612068292357.,
         1720146321549. / 2090206949498., 3134564353537. / 4481467310338.,
         2277821191437. / 14882151754819.};

  rkc = {0., 1432997174477. / 9575080441755., 2526269341429. / 6820363962896.,
         2006345519317. / 3224310063776., 2802321613138. / 2924317926251.};
}

void LSRKStepper::do_step(Real t, Real dt,
                          const ShallowWaterState &state) const {
  auto mesh = shallow_water->mesh;

  YAKL_SCOPE(h_tend_cell, this->tend.h_cell);
  YAKL_SCOPE(vn_tend_edge, this->tend.vn_edge);

  for (Int stage = 0; stage < nstages; ++stage) {
    Real rka_stage = rka[stage];
    parallel_for(
        "lsrk1_h", SimpleBounds<2>(mesh->ncells, mesh->nlayers),
        YAKL_LAMBDA(Int icell, Int k) { h_tend_cell(icell, k) *= rka_stage; });
    parallel_for(
        "lsrk1_v", SimpleBounds<2>(mesh->nedges, mesh->nlayers),
        YAKL_LAMBDA(Int iedge, Int k) { vn_tend_edge(iedge, k) *= rka_stage; });

    Real stagetime = t + rkc[stage] * dt;
    shallow_water->compute_tendency(tend, state, stagetime, AddMode::increment);

    Real rkb_stage = rkb[stage];
    parallel_for(
        "lsrk2_h", SimpleBounds<2>(mesh->ncells, mesh->nlayers),
        YAKL_LAMBDA(Int icell, Int k) {
          state.h_cell(icell, k) += dt * rkb_stage * h_tend_cell(icell, k);
        });
    parallel_for(
        "lsrk2_v", SimpleBounds<2>(mesh->nedges, mesh->nlayers),
        YAKL_LAMBDA(Int iedge, Int k) {
          state.vn_edge(iedge, k) += dt * rkb_stage * vn_tend_edge(iedge, k);
        });
  }
}
} // namespace omega
