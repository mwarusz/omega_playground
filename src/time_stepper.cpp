#include <time_stepper.hpp>

namespace omega {

LSRKStepper::LSRKStepper(ShallowWater &shallow_water) :
  TimeStepper(shallow_water),
  rka(nstages),
  rkb(nstages),
  htend("htend", shallow_water.mesh->ncells),
  vtend("vtend", shallow_water.mesh->nedges) {

  yakl::memset(htend, 0);
  yakl::memset(vtend, 0);

  rka = {
      0., -567301805773. / 1357537059087., -2404267990393. / 2016746695238.,
      -3550918686646. / 2091501179385., -1275806237668. / 842570457699.
  };

  rkb = {
      1432997174477. / 9575080441755., 5161836677717. / 13612068292357.,
      1720146321549. / 2090206949498., 3134564353537. / 4481467310338.,
      2277821191437. / 14882151754819.
  };
  
  rkc = {
        0., 1432997174477. / 9575080441755., 2526269341429. / 6820363962896.,
        2006345519317. / 3224310063776., 2802321613138. / 2924317926251.
  };
}

void LSRKStepper::do_step(Real t, Real dt, Real1d h, Real1d v) const {
  auto mesh = shallow_water->mesh;

  YAKL_SCOPE(htend, this->htend);
  YAKL_SCOPE(vtend, this->vtend);

  for (Int stage = 0; stage < nstages; ++stage) {
    Real rka_stage = rka[stage];
    parallel_for("lsrk1_h", mesh->ncells, YAKL_LAMBDA (Int icell) {
        htend(icell) *= rka_stage;
    });
    parallel_for("lsrk1_v", mesh->nedges, YAKL_LAMBDA (Int iedge) {
        vtend(iedge) *= rka_stage;
    });

    Real stagetime = t + rkc[stage] * dt;
    shallow_water->compute_tendency(htend, vtend, h, v, stagetime);
    
    Real rkb_stage = rkb[stage];
    parallel_for("lsrk2_h", mesh->ncells, YAKL_LAMBDA (Int icell) {
        h(icell) += dt * rkb_stage * htend(icell);
    });
    parallel_for("lsrk2_v", mesh->nedges, YAKL_LAMBDA (Int iedge) {
        v(iedge) += dt * rkb_stage * vtend(iedge);
    });
  }
}
}
