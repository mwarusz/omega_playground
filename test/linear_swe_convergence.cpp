#include <omega.hpp>
#include <iostream>
#include <vector>

using namespace omega;

bool check_rate(Real rate, Real expected_rate, Real atol) {
  return std::abs(rate - expected_rate) < atol && !std::isnan(rate);
}

struct InertiaGravityWave {
  Real h0 = 1000;
  Real eta0 = 1;
  Real grav = 9.81;
  Real f0 = 1e-4;
  Real lx = 10000 * 1e3;
  Real ly = std::sqrt(3) / 2 * lx;
  Int mx = 2;
  Int my = 2;
  Real kx = mx * (2 * pi / lx);
  Real ky = my * (2 * pi / ly);
  Real omega = std::sqrt(f0 * f0 + grav * h0 * (kx * kx + ky * ky));

  YAKL_INLINE Real h(Real x, Real y, Real t) const {
    return eta0 * std::cos(kx * x + ky * y - omega * t);
  }

  YAKL_INLINE Real vx(Real x, Real y, Real t) const {
    Real a = kx * x + ky * y - omega * t;
    return eta0 * grav / (omega * omega - f0 * f0) * (omega * kx * std::cos(a) - f0 * ky * std::sin(a)) ;
  }

  YAKL_INLINE Real vy(Real x, Real y, Real t) const {
    Real a = kx * x + ky * y - omega * t;
    return eta0 * grav / (omega * omega - f0 * f0) * (omega * ky * std::cos(a) + f0 * kx * std::sin(a)) ;
  }
};

Real run(Int nx) {
    InertiaGravityWave inertia_gravity_wave;

    PlanarHexagonalMesh mesh(nx, nx, inertia_gravity_wave.lx / nx, 1);
    LinearShallowWater shallow_water(mesh, inertia_gravity_wave.h0,
                                           inertia_gravity_wave.f0,
                                           inertia_gravity_wave.grav);
    LSRKStepper stepper(shallow_water);
    
    Real timeend = 10 * 60 * 60;
    Real cfl = 1.0;
    Real dt = cfl * mesh.dc / std::sqrt(shallow_water.grav * shallow_water.h0);
    Int numberofsteps = std::ceil(timeend / dt);
    dt = timeend / numberofsteps;
    
    Real2d h("h", mesh.ncells, mesh.nlayers);
    Real2d hexact("hexact", mesh.ncells, mesh.nlayers);
    Real2d v("v", mesh.nedges, mesh.nlayers);
  
    parallel_for("init_h", SimpleBounds<2>(mesh.ncells, mesh.nlayers), YAKL_LAMBDA (Int icell, Int k) {
        Real x = mesh.x_cell(icell);
        Real y = mesh.y_cell(icell);
        h(icell, k) = inertia_gravity_wave.h(x, y, 0);
        hexact(icell, k) = inertia_gravity_wave.h(x, y, timeend);
    });
    
    parallel_for("init_v", SimpleBounds<2>(mesh.nedges, mesh.nlayers), YAKL_LAMBDA (Int iedge, Int k) {
        Real x = mesh.x_edge(iedge);
        Real y = mesh.y_edge(iedge);
        Real nx = std::cos(mesh.angle_edge(iedge));
        Real ny = std::sin(mesh.angle_edge(iedge));
        Real vx = inertia_gravity_wave.vx(x, y, 0);
        Real vy = inertia_gravity_wave.vy(x, y, 0);
        v(iedge, k) = nx * vx + ny * vy;
    });

    for (Int step = 0; step < numberofsteps; ++step) {
      Real t = step * dt;
      stepper.do_step(t, dt, h, v);
    }

    parallel_for("compute_error", SimpleBounds<2>(mesh.ncells, mesh.nlayers), YAKL_LAMBDA (Int icell, Int k) {
        hexact(icell, k) -= h(icell, k);
        hexact(icell, k) = std::abs(hexact(icell, k));
    });
    
    return yakl::intrinsics::maxval(hexact);
}


int main() {
  yakl::init();

  Int nlevels = 3;
  Int nx = 16;
  
  std::vector<Real> err(nlevels);
  for (Int l = 0; l < nlevels; ++l) {
    err[l] = run(nx);
    nx *= 2;
  }

  std::vector<Real> rate(nlevels - 1);
  std::cout << "Inertia gravity wave convergence" << std::endl;
  for (Int l = 0; l < nlevels; ++l) {
    std::cout << l << " " << err[l];
    if (l > 0) {
      rate[l - 1] = std::log2(err[l-1] / err[l]); 
      std::cout << " " << rate[l - 1];
    }
    std::cout << std::endl;
  }

  if (!check_rate(rate.back(), 2, 0.05)) {
    throw std::runtime_error("Inertia-gravity wave is not converging at the right rate");
  }

  yakl::finalize();
}
