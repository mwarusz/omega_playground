#include <iostream>
#include <omega.hpp>
#include <vector>

using namespace omega;

bool check_rate(Real rate, Real expected_rate, Real atol) {
  return std::abs(rate - expected_rate) < atol && !std::isnan(rate);
}

struct TracerAdvectionTest {
  Real m_lx = 1;
  Real m_ly = std::sqrt(3) / 2 * m_lx;

  YAKL_INLINE Real h(Real x, Real y, Real t) const {
    return std::exp(std::cos(2 * pi * x / m_lx));
  }

  YAKL_INLINE Real tr(Real x, Real y, Real t) const {
    return std::sin(2 * pi * (x - t) / m_lx) *
           std::sin(2 * pi * (y - t) / m_ly);
  }

  YAKL_INLINE Real vx(Real x, Real y, Real t) const { return 1; }

  YAKL_INLINE Real vy(Real x, Real y, Real t) const { return 1; }
};

Real run(Int nx) {
  TracerAdvectionTest advection_test;

  PlanarHexagonalMesh mesh(nx, nx, advection_test.m_lx / nx, 1);

  ShallowWaterParams params;
  params.m_disable_h_tendency = true;
  params.m_disable_vn_tendency = true;
  params.m_ntracers = 1;

  ShallowWaterModel shallow_water(mesh, params);

  ShallowWaterState state(shallow_water);

  LSRKStepper stepper(shallow_water);

  Real timeend = 1;
  Real cfl = 0.5;
  Real dt = cfl * mesh.m_dc;
  Int numberofsteps = std::ceil(timeend / dt);
  dt = timeend / numberofsteps;

  auto &h_cell = state.m_h_cell;
  auto &tr_cell = state.m_tr_cell;
  Real3d tr_exact_cell("tr_exact_cell", params.m_ntracers, mesh.m_ncells,
                       mesh.m_nlayers);
  parallel_for(
      "init_h_and_tr", SimpleBounds<2>(mesh.m_ncells, mesh.m_nlayers),
      YAKL_LAMBDA(Int icell, Int k) {
        Real x = mesh.m_x_cell(icell);
        Real y = mesh.m_y_cell(icell);
        h_cell(icell, k) = advection_test.h(x, y, 0);
        tr_cell(0, icell, k) = advection_test.tr(x, y, 0);
        tr_exact_cell(0, icell, k) = advection_test.tr(x, y, timeend);
      });

  auto &vn_edge = state.m_vn_edge;
  parallel_for(
      "init_vn", SimpleBounds<2>(mesh.m_nedges, mesh.m_nlayers),
      YAKL_LAMBDA(Int iedge, Int k) {
        Real x = mesh.m_x_edge(iedge);
        Real y = mesh.m_y_edge(iedge);
        Real nx = std::cos(mesh.m_angle_edge(iedge));
        Real ny = std::sin(mesh.m_angle_edge(iedge));
        Real vx = advection_test.vx(x, y, 0);
        Real vy = advection_test.vy(x, y, 0);
        vn_edge(iedge, k) = nx * vx + ny * vy;
      });

  for (Int step = 0; step < numberofsteps; ++step) {
    Real t = step * dt;
    stepper.do_step(t, dt, state);
  }

  parallel_for(
      "compute_error", SimpleBounds<2>(mesh.m_ncells, mesh.m_nlayers),
      YAKL_LAMBDA(Int icell, Int k) {
        tr_exact_cell(0, icell, k) -= tr_cell(0, icell, k);
        tr_exact_cell(0, icell, k) *= tr_exact_cell(0, icell, k);
      });

  return std::sqrt(yakl::intrinsics::sum(tr_exact_cell) /
                   (mesh.m_nx * mesh.m_ny));
}

int main() {
  yakl::init();

  Int nlevels = 2;
  Int nx = 50;

  std::vector<Real> err(nlevels);
  for (Int l = 0; l < nlevels; ++l) {
    err[l] = run(nx);
    nx *= 2;
  }

  if (nlevels > 1) {
    std::vector<Real> rate(nlevels - 1);
    std::cout << "Tracer advection convergence" << std::endl;
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
          "Tracer advection is not converging at the right rate");
    }
  }

  yakl::finalize();
}
