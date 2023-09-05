#include <time_stepper.hpp>

#include <string>
using namespace std::string_literals;

namespace omega {

RK4Stepper::RK4Stepper(ShallowWaterBase &shallow_water)
    : TimeStepper(shallow_water), rka(nstages - 1), rkb(nstages),
      rkc(nstages - 1),
      htend("htend", shallow_water.mesh->ncells, shallow_water.mesh->nlayers),
      vtend("vtend", shallow_water.mesh->nedges, shallow_water.mesh->nlayers),
      hprovis("hprovis", shallow_water.mesh->ncells,
              shallow_water.mesh->nlayers),
      vprovis("vprovis", shallow_water.mesh->nedges,
              shallow_water.mesh->nlayers),
      hold("hold", shallow_water.mesh->ncells, shallow_water.mesh->nlayers),
      vold("vold", shallow_water.mesh->nedges, shallow_water.mesh->nlayers) {

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

void RK4Stepper::do_step(Real t, Real dt, Real2d h, Real2d v) const {
  auto mesh = shallow_water->mesh;

  YAKL_SCOPE(hold, this->hold);
  YAKL_SCOPE(vold, this->vold);

  YAKL_SCOPE(htend, this->htend);
  YAKL_SCOPE(vtend, this->vtend);

  YAKL_SCOPE(hprovis, this->hprovis);
  YAKL_SCOPE(vprovis, this->vprovis);

  h.deep_copy_to(hold);
  v.deep_copy_to(vold);

  // k1
  shallow_water->compute_tendency(htend, vtend, h, v, t);

  for (Int stage = 0; stage < nstages; ++stage) {

    const Real rkb_stage = rkb[stage];
    parallel_for(
        "rk4_accumulate_h", SimpleBounds<2>(mesh->ncells, mesh->nlayers),
        YAKL_LAMBDA(Int icell, Int k) {
          h(icell, k) += dt * rkb_stage * htend(icell, k);
        });
    parallel_for(
        "rk4_accumulate_v", SimpleBounds<2>(mesh->nedges, mesh->nlayers),
        YAKL_LAMBDA(Int iedge, Int k) {
          v(iedge, k) += dt * rkb_stage * vtend(iedge, k);
        });

    if (stage < nstages - 1) {
      Real stagetime = t + rkc[stage] * dt;
      const Real rka_stage = rka[stage];

      parallel_for(
          "rk4_compute_hprovis", SimpleBounds<2>(mesh->ncells, mesh->nlayers),
          YAKL_LAMBDA(Int icell, Int k) {
            hprovis(icell, k) =
                hold(icell, k) + dt * rka_stage * htend(icell, k);
          });
      parallel_for(
          "rk4_compute_vprovis", SimpleBounds<2>(mesh->nedges, mesh->nlayers),
          YAKL_LAMBDA(Int iedge, Int k) {
            vprovis(iedge, k) =
                vold(iedge, k) + dt * rka_stage * vtend(iedge, k);
          });

      shallow_water->compute_tendency(htend, vtend, hprovis, vprovis,
                                      stagetime);
    }
  }
}
} // namespace omega
