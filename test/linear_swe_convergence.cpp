#include <omega.hpp>
#include <iostream>
#include <vector>

using namespace omega;

static constexpr Real h0 = 1000;
static constexpr Real eta0 = 1;
static constexpr Real g0 = 10;
static constexpr Real f0 = 1e-4;

YAKL_INLINE Real h_exact(Real x, Real y, Real t, Real kx, Real ky, Real omega) {
  return eta0 * cos(kx * x + ky * y - omega * t);
}

YAKL_INLINE Real vx_exact(Real x, Real y, Real t, Real kx, Real ky, Real omega) {
  Real a = kx * x + ky * y - omega * t;
  return eta0 * g0 / (omega * omega - f0 * f0) * (omega * kx * std::cos(a) - f0 * ky * std::sin(a)) ;
}

YAKL_INLINE Real vy_exact(Real x, Real y, Real t, Real kx, Real ky, Real omega) {
  Real a = kx * x + ky * y - omega * t;
  return eta0 * g0 / (omega * omega - f0 * f0) * (omega * ky * std::cos(a) + f0 * kx * std::sin(a)) ;
}

Real run(Int n) {
    Real lx = 1000;
    Real ly = std::sqrt(3) / 2 * lx;
    Real kx = 2 * (2 * pi / lx);
    Real ky = 2 * (2 * pi / ly);
    Real omega = std::sqrt(f0 * f0 + g0 * h0 * (kx * kx + ky * ky));

    PlanarHexagonalMesh mesh(n, n, lx / n);
    LinearShallowWater shallow_water(mesh, h0, f0, g0);
    LSRKStepper stepper(shallow_water);
    
    Real timeend = 20;
    Real cfl = 0.01;
    Real dt = cfl * mesh.dc / std::sqrt(g0 * h0);
    Int numberofsteps = std::ceil(timeend / dt);
    dt = timeend / numberofsteps;
    
    Real1d h("h", mesh.ncells);
    Real1d hexact("hexact", mesh.ncells);
    Real1d v("v", mesh.nedges);
  
    parallel_for("init_h", mesh.ncells, YAKL_LAMBDA (Int icell) {
        Real x = mesh.x_cell(icell);
        Real y = mesh.y_cell(icell);
        h(icell) = h_exact(x, y, 0, kx, ky, omega);
        hexact(icell) = h_exact(x, y, timeend, kx, ky, omega);
    });
    
    parallel_for("init_v", mesh.nedges, YAKL_LAMBDA (Int iedge) {
        Real x = mesh.x_edge(iedge);
        Real y = mesh.y_edge(iedge);
        Real nx = std::cos(mesh.angle_edge(iedge));
        Real ny = std::sin(mesh.angle_edge(iedge));
        Real vx = vx_exact(x, y, 0, kx, ky, omega);
        Real vy = vy_exact(x, y, 0, kx, ky, omega);
        v(iedge) = nx * vx + ny * vy;
    });

    Real en0 = shallow_water.compute_energy(h, v);
    for (Int step = 0; step < numberofsteps; ++step) {
      Real t = step * dt;
      stepper.do_step(t, dt, h, v);
    }
    Real enf = shallow_water.compute_energy(h, v);

    std::cout << "Energy change: " << (enf - en0) / en0 << std::endl;

    parallel_for("compute_error", mesh.ncells, YAKL_LAMBDA (Int icell) {
        hexact(icell) -= h(icell);
        hexact(icell) = std::abs(hexact(icell));
    });
    
    return yakl::intrinsics::maxval(hexact);
}


int main() {
  yakl::init();

  Int nlevels = 1;
  std::vector<Real> err(nlevels);
  Int n = 16;
  
  for (Int l = 0; l < nlevels; ++l) {
    err[l] = run(n);
    n *= 2;
  }

  std::cout << "Inertia gravity wave convergence" << std::endl;
  for (Int l = 0; l < nlevels; ++l) {
    std::cout << l << " " << err[l];
    if (l > 0) {
      std::cout << " " << std::log2(err[l-1] / err[l]);
    }
    std::cout << std::endl;
  }

  yakl::finalize();
}
