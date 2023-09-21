#include <time_stepper.hpp>

namespace omega {

LSRKStepper::LSRKStepper(ShallowWaterModelBase &shallow_water)
    : TimeStepper(shallow_water), m_rka(nstages), m_rkb(nstages),
      m_rkc(nstages), m_tend(shallow_water) {

  yakl::memset(m_tend.m_h_cell, 0);
  yakl::memset(m_tend.m_vn_edge, 0);

  m_rka = {0., -567301805773. / 1357537059087.,
           -2404267990393. / 2016746695238., -3550918686646. / 2091501179385.,
           -1275806237668. / 842570457699.};

  m_rkb = {1432997174477. / 9575080441755., 5161836677717. / 13612068292357.,
           1720146321549. / 2090206949498., 3134564353537. / 4481467310338.,
           2277821191437. / 14882151754819.};

  m_rkc = {0., 1432997174477. / 9575080441755., 2526269341429. / 6820363962896.,
           2006345519317. / 3224310063776., 2802321613138. / 2924317926251.};
}

void LSRKStepper::do_step(Real t, Real dt,
                          const ShallowWaterState &state) const {
  const auto &mesh = m_shallow_water->m_mesh;

  YAKL_SCOPE(h_tend_cell, m_tend.m_h_cell);
  YAKL_SCOPE(vn_tend_edge, m_tend.m_vn_edge);
  YAKL_SCOPE(tr_tend_cell, m_tend.m_tr_cell);
  Int ntracers = m_shallow_water->m_ntracers;

  for (Int stage = 0; stage < nstages; ++stage) {
    Real rka_stage = m_rka[stage];
    parallel_for(
        "lsrk1_h", SimpleBounds<2>(mesh->m_ncells, mesh->m_nlayers),
        YAKL_LAMBDA(Int icell, Int k) { h_tend_cell(icell, k) *= rka_stage; });
    parallel_for(
        "lsrk1_v", SimpleBounds<2>(mesh->m_nedges, mesh->m_nlayers),
        YAKL_LAMBDA(Int iedge, Int k) { vn_tend_edge(iedge, k) *= rka_stage; });
    parallel_for(
        "lsrk1_tr", SimpleBounds<3>(ntracers, mesh->m_ncells, mesh->m_nlayers),
        YAKL_LAMBDA(Int l, Int icell, Int k) {
          tr_tend_cell(l, icell, k) *= rka_stage;
        });

    Real stagetime = t + m_rkc[stage] * dt;
    m_shallow_water->compute_tendency(m_tend, state, stagetime,
                                      AddMode::increment);

    Real rkb_stage = m_rkb[stage];
    parallel_for(
        "lsrk2_h", SimpleBounds<2>(mesh->m_ncells, mesh->m_nlayers),
        YAKL_LAMBDA(Int icell, Int k) {
          state.m_h_cell(icell, k) += dt * rkb_stage * h_tend_cell(icell, k);
        });
    parallel_for(
        "lsrk2_v", SimpleBounds<2>(mesh->m_nedges, mesh->m_nlayers),
        YAKL_LAMBDA(Int iedge, Int k) {
          state.m_vn_edge(iedge, k) += dt * rkb_stage * vn_tend_edge(iedge, k);
        });
    parallel_for(
        "lsrk2_tr", SimpleBounds<3>(ntracers, mesh->m_nedges, mesh->m_nlayers),
        YAKL_LAMBDA(Int l, Int icell, Int k) {
          state.m_tr_cell(l, icell, k) +=
              dt * rkb_stage * tr_tend_cell(l, icell, k);
        });
  }
}
} // namespace omega
