#include <iostream>
#include <memory>
#include <omega.hpp>
#include <vector>

using namespace omega;

bool check_rate(Real rate, Real expected_rate, Real atol) {
  return std::abs(rate - expected_rate) < atol && !std::isnan(rate);
}

struct InertiaGravityWave {
  Real m_h0 = 1000;
  Real m_eta0 = 1;
  Real m_grav = 9.80616;
  Real m_f0 = 1e-4;
  Real m_lx = 10000 * 1e3;
  Real m_ly = std::sqrt(3) / 2 * m_lx;
  Int m_mx = 2;
  Int m_my = 2;
  Real m_kx = m_mx * (2 * pi / m_lx);
  Real m_ky = m_my * (2 * pi / m_ly);
  Real m_omega =
      std::sqrt(m_f0 * m_f0 + m_grav * m_h0 * (m_kx * m_kx + m_ky * m_ky));

  KOKKOS_INLINE_FUNCTION Real h(Real x, Real y, Real t) const {
    return m_h0 + m_eta0 * std::cos(m_kx * x + m_ky * y - m_omega * t);
  }

  KOKKOS_INLINE_FUNCTION Real vx(Real x, Real y, Real t) const {
    Real a = m_kx * x + m_ky * y - m_omega * t;
    return m_eta0 * m_grav / (m_omega * m_omega - m_f0 * m_f0) *
           (m_omega * m_kx * std::cos(a) - m_f0 * m_ky * std::sin(a));
  }

  KOKKOS_INLINE_FUNCTION Real vy(Real x, Real y, Real t) const {
    Real a = m_kx * x + m_ky * y - m_omega * t;
    return m_eta0 * m_grav / (m_omega * m_omega - m_f0 * m_f0) *
           (m_omega * m_ky * std::cos(a) + m_f0 * m_kx * std::sin(a));
  }
};

Real run(Int nx) {
  InertiaGravityWave inertia_gravity_wave;

  auto mesh = std::make_unique<PlanarHexagonalMesh>(
      nx, nx, inertia_gravity_wave.m_lx / nx, 1);

  LinearShallowWaterParams params;
  params.m_h0 = inertia_gravity_wave.m_h0;
  params.m_f0 = inertia_gravity_wave.m_f0;
  params.m_grav = inertia_gravity_wave.m_grav;

  LinearShallowWaterModel shallow_water(mesh.get(), params);

  ShallowWaterState state(shallow_water);

  RK4Stepper stepper(shallow_water);

  Real timeend = 10 * 60 * 60;
  Real dt_per_km = 3;
  Real dt = dt_per_km * mesh->m_dc / 1e3;
  Int numberofsteps = std::ceil(timeend / dt);
  dt = timeend / numberofsteps;

  auto &h_cell = state.m_h_cell;
  Real2d hexact_cell("hexact_cell", mesh->m_ncells, mesh->m_nlayers);
  OMEGA_SCOPE(x_cell, mesh->m_x_cell);
  OMEGA_SCOPE(y_cell, mesh->m_y_cell);
  parallel_for(
      "init_h", MDRangePolicy<2>({0, 0}, {mesh->m_ncells, mesh->m_nlayers}),
      KOKKOS_LAMBDA(Int icell, Int k) {
        Real x = x_cell(icell);
        Real y = y_cell(icell);
        h_cell(icell, k) = inertia_gravity_wave.h(x, y, 0);
        hexact_cell(icell, k) = inertia_gravity_wave.h(x, y, timeend);
      });

  auto &vn_edge = state.m_vn_edge;
  OMEGA_SCOPE(x_edge, mesh->m_x_edge);
  OMEGA_SCOPE(y_edge, mesh->m_y_edge);
  OMEGA_SCOPE(angle_edge, mesh->m_angle_edge);
  parallel_for(
      "init_vn", MDRangePolicy<2>({0, 0}, {mesh->m_nedges, mesh->m_nlayers}),
      KOKKOS_LAMBDA(Int iedge, Int k) {
        Real x = x_edge(iedge);
        Real y = y_edge(iedge);
        Real nx = std::cos(angle_edge(iedge));
        Real ny = std::sin(angle_edge(iedge));
        Real vx = inertia_gravity_wave.vx(x, y, 0);
        Real vy = inertia_gravity_wave.vy(x, y, 0);
        vn_edge(iedge, k) = nx * vx + ny * vy;
      });

  for (Int step = 0; step < numberofsteps; ++step) {
    Real t = step * dt;
    stepper.do_step(t, dt, state);
  }

  Real errf;
  parallel_reduce(
      "compute_error",
      MDRangePolicy<2>({0, 0}, {mesh->m_ncells, mesh->m_nlayers}),
      KOKKOS_LAMBDA(Int icell, Int k, Real & accum) {
        Real err = hexact_cell(icell, k) - h_cell(icell, k);
        accum += err * err;
      },
      errf);

  return std::sqrt(errf / (mesh->m_nx * mesh->m_ny));
}

int main() {
  Kokkos::initialize();

  Int nlevels = 2;
  Int nx = 50;

  std::vector<Real> err(nlevels);
  for (Int l = 0; l < nlevels; ++l) {
    err[l] = run(nx);
    nx *= 2;
  }

  if (nlevels > 1) {
    std::vector<Real> rate(nlevels - 1);
    std::cout << "Inertia gravity wave convergence" << std::endl;
    for (Int l = 0; l < nlevels; ++l) {
      std::cout << l << " " << err[l];
      if (l > 0) {
        rate[l - 1] = std::log2(err[l - 1] / err[l]);
        std::cout << " " << rate[l - 1];
      }
      std::cout << std::endl;
    }

    if (!check_rate(rate.back(), 2, 0.05)) {
      throw std::runtime_error(
          "Inertia-gravity wave is not converging at the right rate");
    }
  }

  Kokkos::finalize();
}
