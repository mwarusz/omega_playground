#include <iostream>
#include <memory>
#include <omega.hpp>
#include <vector>

using namespace omega;

bool check_rate(Real rate, Real expected_rate, Real atol) {
  return std::abs(rate - expected_rate) < atol && !std::isnan(rate);
}

struct TracerHyperDiffusionTest {
  Real m_diff4 = 1;
  Real m_lx = 10;
  Real m_ly = std::sqrt(3) / 2 * m_lx;

  KOKKOS_INLINE_FUNCTION Real h(Real x, Real y, Real t) const { return 1; }

  KOKKOS_INLINE_FUNCTION Real tr(Real x, Real y, Real t) const {
    Real c = 4 * pi * pi * (1._fp / (m_lx * m_lx) + 1._fp / (m_ly * m_ly));
    return std::sin(2 * pi * x / m_lx) * std::sin(2 * pi * y / m_ly) *
           exp(-m_diff4 * c * c * t);
  }

  KOKKOS_INLINE_FUNCTION Real vx(Real x, Real y, Real t) const { return 0; }

  KOKKOS_INLINE_FUNCTION Real vy(Real x, Real y, Real t) const { return 0; }
};

Real run(Int nx) {
  TracerHyperDiffusionTest hyperdiffusion_test;

  auto mesh = std::make_unique<PlanarHexagonalMesh>(
      nx, nx, hyperdiffusion_test.m_lx / nx, 1);

  ShallowWaterParams params;
  params.m_disable_h_tendency = true;
  params.m_disable_vn_tendency = true;
  params.m_ntracers = 1;
  params.m_eddy_diff4 = hyperdiffusion_test.m_diff4;

  ShallowWaterModel shallow_water(mesh.get(), params);

  ShallowWaterState state(mesh.get(), params);

  LSRKStepper stepper(shallow_water);

  Real timeend = 2;
  Real dt = std::pow(mesh->m_dc, 4) / (8 * hyperdiffusion_test.m_diff4);
  Int numberofsteps = std::ceil(timeend / dt);
  dt = timeend / numberofsteps;

  auto &h_cell = state.m_h_cell;
  auto &tr_cell = state.m_tr_cell;
  Real3d tr_exact_cell("tr_exact_cell", params.m_ntracers, mesh->m_ncells,
                       mesh->m_nlayers);
  OMEGA_SCOPE(x_cell, mesh->m_x_cell);
  OMEGA_SCOPE(y_cell, mesh->m_y_cell);
  parallel_for(
      "init_h_and_tr",
      MDRangePolicy<2>({0, 0}, {mesh->m_ncells, mesh->m_nlayers}),
      KOKKOS_LAMBDA(Int icell, Int k) {
        Real x = x_cell(icell);
        Real y = y_cell(icell);
        h_cell(icell, k) = hyperdiffusion_test.h(x, y, 0);
        tr_cell(0, icell, k) = hyperdiffusion_test.tr(x, y, 0);
        tr_exact_cell(0, icell, k) = hyperdiffusion_test.tr(x, y, timeend);
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
        Real vx = hyperdiffusion_test.vx(x, y, 0);
        Real vy = hyperdiffusion_test.vy(x, y, 0);
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
        Real err = tr_exact_cell(0, icell, k) - tr_cell(0, icell, k);
        accum += err * err;
      },
      errf);

  return std::sqrt(errf / (mesh->m_nx * mesh->m_ny));
}

int main() {
  Kokkos::initialize();

  Int nlevels = 2;
  Int nx = 16;

  std::vector<Real> err(nlevels);
  for (Int l = 0; l < nlevels; ++l) {
    err[l] = run(nx);
    nx *= 2;
  }

  if (nlevels > 1) {
    std::vector<Real> rate(nlevels - 1);
    std::cout << "Tracer hyperdiffusion convergence" << std::endl;
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
          "Tracer hyperdiffusion is not converging at the right rate");
    }
  }

  Kokkos::finalize();
}
