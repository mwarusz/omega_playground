#include <iostream>
#include <omega.hpp>
#include <vector>

using namespace omega;

bool check_rate(Real rate, Real expected_rate, Real atol) {
  return std::abs(rate - expected_rate) < atol && !std::isnan(rate);
}

struct TracerAdvection {
  Real lx = 1;
  Real ly = std::sqrt(3) / 2 * lx;

  YAKL_INLINE Real h(Real x, Real y, Real t) const {
    return std::exp(std::cos(2 * pi * x));
  }

  YAKL_INLINE Real tr(Real x, Real y, Real t) const {
    return std::sin(2 * pi * (x - t) / lx) * std::sin(2 * pi * (y - t) / ly);
  }

  YAKL_INLINE Real vx(Real x, Real y, Real t) const { return 1; }

  YAKL_INLINE Real vy(Real x, Real y, Real t) const { return 1; }
};

Real run(Int nx) {
  TracerAdvection advection;

  PlanarHexagonalMesh mesh(nx, nx, advection.lx / nx, 1);

  ShallowWaterParams params;
  params.disable_h_tendency = true;
  params.disable_vn_tendency = true;
  params.ntracers = 1;

  ShallowWaterModel shallow_water(mesh, params);

  ShallowWaterState state(shallow_water);

  LSRKStepper stepper(shallow_water);

  Real timeend = 1;
  Real cfl = 0.5;
  Real dt = cfl * mesh.dc;
  Int numberofsteps = std::ceil(timeend / dt);
  dt = timeend / numberofsteps;

  auto &h_cell = state.h_cell;
  auto &tr_cell = state.tr_cell;
  Real3d tr_exact_cell("tr_exact_cell", params.ntracers, mesh.ncells,
                       mesh.nlayers);
  parallel_for(
      "init_h_and_tr", SimpleBounds<2>(mesh.ncells, mesh.nlayers),
      YAKL_LAMBDA(Int icell, Int k) {
        Real x = mesh.x_cell(icell);
        Real y = mesh.y_cell(icell);
        h_cell(icell, k) = advection.h(x, y, 0);
        tr_cell(0, icell, k) = advection.tr(x, y, 0);
        tr_exact_cell(0, icell, k) = advection.tr(x, y, timeend);
      });

  auto &vn_edge = state.vn_edge;
  parallel_for(
      "init_vn", SimpleBounds<2>(mesh.nedges, mesh.nlayers),
      YAKL_LAMBDA(Int iedge, Int k) {
        Real x = mesh.x_edge(iedge);
        Real y = mesh.y_edge(iedge);
        Real nx = std::cos(mesh.angle_edge(iedge));
        Real ny = std::sin(mesh.angle_edge(iedge));
        Real vx = advection.vx(x, y, 0);
        Real vy = advection.vy(x, y, 0);
        vn_edge(iedge, k) = nx * vx + ny * vy;
      });

  for (Int step = 0; step < numberofsteps; ++step) {
    Real t = step * dt;
    stepper.do_step(t, dt, state);
  }

  parallel_for(
      "compute_error", SimpleBounds<2>(mesh.ncells, mesh.nlayers),
      YAKL_LAMBDA(Int icell, Int k) {
        tr_exact_cell(0, icell, k) -= tr_cell(0, icell, k);
        tr_exact_cell(0, icell, k) *= tr_exact_cell(0, icell, k);
      });

  return std::sqrt(yakl::intrinsics::sum(tr_exact_cell) / (mesh.nx * mesh.ny));
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
