#include <iostream>
#include <omega.hpp>
#include <vector>

using namespace omega;

bool check_rate(Real rate, Real expected_rate, Real atol) {
  return std::abs(rate - expected_rate) < atol && !std::isnan(rate);
}

struct ManufacturedSolution {
  Real grav = 9.80616;
  Real f0 = 1e-4;
  Real lx = 10000 * 1e3;
  Real ly = std::sqrt(3) / 2 * lx;
  Real eta0 = 1;
  Real h0 = 1000;
  Int mx = 2;
  Int my = 2;
  Real kx = mx * (2 * pi / lx);
  Real ky = my * (2 * pi / ly);
  Real omega = std::sqrt(grav * h0 * (kx * kx + ky * ky));

  YAKL_INLINE Real h(Real x, Real y, Real t) const {
    return h0 + eta0 * std::sin(kx * x + ky * y - omega * t);
  }

  YAKL_INLINE Real vx(Real x, Real y, Real t) const {
    return eta0 * std::cos(kx * x + ky * y - omega * t);
  }

  YAKL_INLINE Real vy(Real x, Real y, Real t) const {
    return eta0 * std::cos(kx * x + ky * y - omega * t);
  }

  YAKL_INLINE Real htend(Real x, Real y, Real t) const {
    using std::cos;
    using std::sin;

    Real phi = kx * x + ky * y - omega * t;
    return eta0 * (-h0 * (kx + ky) * sin(phi) - omega * cos(phi) +
                   eta0 * (kx + ky) * cos(2 * phi));
  }

  YAKL_INLINE Real vxtend(Real x, Real y, Real t) const {
    using std::cos;
    using std::sin;

    Real phi = kx * x + ky * y - omega * t;
    return eta0 * ((-f0 + grav * kx) * cos(phi) + omega * sin(phi) -
                   eta0 * (kx + ky) * sin(2 * phi) / 2);
  }

  YAKL_INLINE Real vytend(Real x, Real y, Real t) const {
    using std::cos;
    using std::sin;

    Real phi = kx * x + ky * y - omega * t;
    return eta0 * ((f0 + grav * ky) * cos(phi) + omega * sin(phi) -
                   eta0 * (kx + ky) * sin(2 * phi) / 2);
  }
};

struct ManufacturedShallowWater : ShallowWater {
  ManufacturedSolution manufactured_solution;

  ManufacturedShallowWater(PlanarHexagonalMesh &mesh,
                           const ManufacturedSolution &manufactured_solution)
      : ShallowWater(mesh, manufactured_solution.f0,
                     manufactured_solution.grav),
        manufactured_solution(manufactured_solution) {}

  void additional_tendency(Real2d htend, Real2d vtend, RealConst2d h,
                           RealConst2d v, Real t) const override {
    using std::cos;
    using std::sin;

    YAKL_SCOPE(manufactured_solution, this->manufactured_solution);

    YAKL_SCOPE(x_cell, mesh->x_cell);
    YAKL_SCOPE(y_cell, mesh->y_cell);
    YAKL_SCOPE(max_level_cell, mesh->max_level_cell);
    parallel_for(
        "manufactured_htend", mesh->ncells, YAKL_LAMBDA(Int icell) {
          for (Int k = 0; k < max_level_cell(icell); ++k) {
            Real x = x_cell(icell);
            Real y = y_cell(icell);
            htend(icell, k) += manufactured_solution.htend(x, y, t);
          }
        });

    YAKL_SCOPE(x_edge, mesh->x_edge);
    YAKL_SCOPE(y_edge, mesh->y_edge);
    YAKL_SCOPE(angle_edge, mesh->angle_edge);
    YAKL_SCOPE(max_level_edge_top, mesh->max_level_edge_top);
    parallel_for(
        "manufactured_vtend", mesh->nedges, YAKL_LAMBDA(Int iedge) {
          for (Int k = 0; k < max_level_edge_top(iedge); ++k) {
            Real x = x_edge(iedge);
            Real y = y_edge(iedge);

            Real nx = std::cos(angle_edge(iedge));
            Real ny = std::sin(angle_edge(iedge));

            Real vxtend = manufactured_solution.vxtend(x, y, t);
            Real vytend = manufactured_solution.vytend(x, y, t);

            vtend(iedge, k) += nx * vxtend + ny * vytend;
          }
        });
  }
};

Real run(Int n) {
  ManufacturedSolution manufactured_solution;
  PlanarHexagonalMesh mesh(n, n, manufactured_solution.lx / n);
  ManufacturedShallowWater shallow_water(mesh, manufactured_solution);
  RK4Stepper stepper(shallow_water);

  Real timeend = 10 * 60 * 60;
  Real dt_per_km = 3;
  Real dt = dt_per_km * mesh.dc / 1e3;
  Int numberofsteps = std::ceil(timeend / dt);
  dt = timeend / numberofsteps;

  Real2d h("h", mesh.ncells, mesh.nlayers);
  Real2d hexact("hexact", mesh.ncells, mesh.nlayers);
  Real2d v("v", mesh.nedges, mesh.nlayers);

  parallel_for(
      "init_h", SimpleBounds<2>(mesh.ncells, mesh.nlayers),
      YAKL_LAMBDA(Int icell, Int k) {
        Real x = mesh.x_cell(icell);
        Real y = mesh.y_cell(icell);
        h(icell, k) = manufactured_solution.h(x, y, 0);
        hexact(icell, k) = manufactured_solution.h(x, y, timeend);
      });

  parallel_for(
      "init_v", SimpleBounds<2>(mesh.nedges, mesh.nlayers),
      YAKL_LAMBDA(Int iedge, Int k) {
        Real x = mesh.x_edge(iedge);
        Real y = mesh.y_edge(iedge);
        Real nx = std::cos(mesh.angle_edge(iedge));
        Real ny = std::sin(mesh.angle_edge(iedge));
        Real vx = manufactured_solution.vx(x, y, 0);
        Real vy = manufactured_solution.vy(x, y, 0);
        v(iedge, k) = nx * vx + ny * vy;
      });

  for (Int step = 0; step < numberofsteps; ++step) {
    Real t = step * dt;
    stepper.do_step(t, dt, h, v);
  }

  parallel_for(
      "compute_error", SimpleBounds<2>(mesh.ncells, mesh.nlayers),
      YAKL_LAMBDA(Int icell, Int k) {
        hexact(icell, k) -= h(icell, k);
        hexact(icell, k) *= hexact(icell, k);
      });

  return std::sqrt(yakl::intrinsics::sum(hexact) / (mesh.nx * mesh.ny));
}

int main() {
  yakl::init();

  Int nlevels = 2;
  std::vector<Real> err(nlevels);
  Int nx = 50;

  for (Int l = 0; l < nlevels; ++l) {
    err[l] = run(nx);
    nx *= 2;
  }

  if (nlevels > 1) {
    std::vector<Real> rate(nlevels - 1);
    std::cout << "Manufactured solution convergence" << std::endl;
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
          "Manufactured solution is not converging at the right rate");
    }
  }

  yakl::finalize();
}
