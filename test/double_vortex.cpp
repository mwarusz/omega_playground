#include <iostream>
#include <memory>
#include <omega.hpp>
#include <string>

using namespace omega;

struct DoubleVortex {
  Real m_g = 9.80616;
  Real m_lx = 5000000;
  Real m_ly = 5000000;
  Real m_coriolis = 0.00006147;
  Real m_h0 = 750;
  Real m_ox = 0.1;
  Real m_oy = 0.1;
  Real m_sigmax = 3. / 40. * m_lx;
  Real m_sigmay = 3. / 40. * m_ly;
  Real m_dh = 75;
  Real m_xc1 = (0.5 - m_ox) * m_lx;
  Real m_yc1 = (0.5 - m_oy) * m_ly;
  Real m_xc2 = (0.5 + m_ox) * m_lx;
  Real m_yc2 = (0.5 + m_oy) * m_ly;

  KOKKOS_INLINE_FUNCTION Real h(Real x, Real y) const {
    using std::exp;
    using std::sin;

    Real xprime1 = m_lx / (pi * m_sigmax) * sin(pi / m_lx * (x - m_xc1));
    Real yprime1 = m_ly / (pi * m_sigmay) * sin(pi / m_ly * (y - m_yc1));
    Real xprime2 = m_lx / (pi * m_sigmax) * sin(pi / m_lx * (x - m_xc2));
    Real yprime2 = m_ly / (pi * m_sigmay) * sin(pi / m_ly * (y - m_yc2));

    return m_h0 - m_dh * (exp(-0.5 * (xprime1 * xprime1 + yprime1 * yprime1)) +
                          exp(-0.5 * (xprime2 * xprime2 + yprime2 * yprime2)) -
                          4. * pi * m_sigmax * m_sigmay / m_lx / m_ly);
  }

  KOKKOS_INLINE_FUNCTION Real vx(Real x, Real y) const {
    using std::exp;
    using std::sin;

    Real xprime1 = m_lx / (pi * m_sigmax) * sin(pi / m_lx * (x - m_xc1));
    Real yprime1 = m_ly / (pi * m_sigmay) * sin(pi / m_ly * (y - m_yc1));
    Real xprime2 = m_lx / (pi * m_sigmax) * sin(pi / m_lx * (x - m_xc2));
    Real yprime2 = m_ly / (pi * m_sigmay) * sin(pi / m_ly * (y - m_yc2));
    Real yprimeprime1 =
        m_ly / (2.0 * pi * m_sigmay) * sin(2.0 * pi / m_ly * (y - m_yc1));
    Real yprimeprime2 =
        m_ly / (2.0 * pi * m_sigmay) * sin(2.0 * pi / m_ly * (y - m_yc2));

    Real vx =
        -m_g * m_dh / m_coriolis / m_sigmay *
        (yprimeprime1 * exp(-0.5 * (xprime1 * xprime1 + yprime1 * yprime1)) +
         yprimeprime2 * exp(-0.5 * (xprime2 * xprime2 + yprime2 * yprime2)));
    return vx;
  }

  KOKKOS_INLINE_FUNCTION Real vy(Real x, Real y) const {
    using std::exp;
    using std::sin;

    Real xprime1 = m_lx / (pi * m_sigmax) * sin(pi / m_lx * (x - m_xc1));
    Real yprime1 = m_ly / (pi * m_sigmay) * sin(pi / m_ly * (y - m_yc1));
    Real xprime2 = m_lx / (pi * m_sigmax) * sin(pi / m_lx * (x - m_xc2));
    Real yprime2 = m_ly / (pi * m_sigmay) * sin(pi / m_ly * (y - m_yc2));
    Real xprimeprime1 =
        m_lx / (2.0 * pi * m_sigmax) * sin(2.0 * pi / m_lx * (x - m_xc1));
    Real xprimeprime2 =
        m_lx / (2.0 * pi * m_sigmax) * sin(2.0 * pi / m_lx * (x - m_xc2));

    Real vy =
        m_g * m_dh / m_coriolis / m_sigmax *
        (xprimeprime1 * exp(-0.5 * (xprime1 * xprime1 + yprime1 * yprime1)) +
         xprimeprime2 * exp(-0.5 * (xprime2 * xprime2 + yprime2 * yprime2)));
    return vy;
  }
};

void run(Int nx, Real cfl) {
  DoubleVortex double_vortex;

  Real dc = double_vortex.m_lx / nx;
  Int ny = std::ceil(double_vortex.m_ly / (dc * std::sqrt(3) / 2));
  auto mesh = std::make_unique<PlanarHexagonalMesh>(nx, ny, dc);

  ShallowWaterParams params;
  params.m_f0 = double_vortex.m_coriolis;
  params.m_grav = double_vortex.m_g;

  ShallowWaterModel shallow_water(mesh.get(), params);

  ShallowWaterState state(mesh.get(), params);

  LSRKStepper stepper(shallow_water);

  Real timeend = 200 * 1000;
  Real dt = cfl * mesh->m_dc / std::sqrt(params.m_grav * double_vortex.m_h0);
  Int numberofsteps = std::ceil(timeend / dt);
  dt = timeend / numberofsteps;

  auto &h_cell = state.m_h_cell;
  OMEGA_SCOPE(x_cell, mesh->m_x_cell);
  OMEGA_SCOPE(y_cell, mesh->m_y_cell);
  parallel_for(
      "init_h", MDRangePolicy<2>({0, 0}, {mesh->m_ncells, mesh->m_nlayers}),
      KOKKOS_LAMBDA(Int icell, Int k) {
        Real x = x_cell(icell);
        Real y = y_cell(icell);
        h_cell(icell, k) = double_vortex.h(x, y);
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
        Real vx = double_vortex.vx(x, y);
        Real vy = double_vortex.vy(x, y);
        vn_edge(iedge, k) = nx * vx + ny * vy;
      });

  Real mass0 = mass_integral(state, shallow_water);
  Real cir0 = circulation_integral(state, shallow_water);
  Real en0 = energy_integral(state, shallow_water);

  // std::cout << "Initial h: " << yakl::intrinsics::minval(h_cell) << " "
  //           << yakl::intrinsics::maxval(h_cell) << std::endl;
  // std::cout << "Initial vn: " << yakl::intrinsics::minval(vn_edge) << " "
  //           << yakl::intrinsics::maxval(vn_edge) << std::endl;

  for (Int step = 0; step < numberofsteps; ++step) {
    Real t = step * dt;
    stepper.do_step(t, dt, state);
  }

  // std::cout << "Final h: " << yakl::intrinsics::minval(h_cell) << " "
  //           << yakl::intrinsics::maxval(h_cell) << std::endl;
  // std::cout << "Final vn: " << yakl::intrinsics::minval(vn_edge) << " "
  //           << yakl::intrinsics::maxval(vn_edge) << std::endl;

  Real massf = mass_integral(state, shallow_water);
  Real cirf = circulation_integral(state, shallow_water);
  Real enf = energy_integral(state, shallow_water);

  Real mass_change = (massf - mass0) / mass0;
  Real cir_change = (cirf - cir0) / cir0;
  Real en_change = (enf - en0) / en0;

  std::cout << "Mass change: " << mass_change << std::endl;
  std::cout << "Circulation change: " << cir_change << std::endl;
  std::cout << "Energy change: " << en_change << std::endl;

  if (std::abs(mass_change) > 5e-15) {
    throw std::runtime_error("Mass conservation check failed");
  }
  if (std::abs(cir_change) > 5e-15) {
    throw std::runtime_error("Circulation conservation check failed");
  }
  if (std::abs(en_change) > 1e-10) {
    throw std::runtime_error("Energy conservation check failed");
  }
}

int main(int argc, char *argv[]) {
  Kokkos::initialize();

  Int nx = 25;
  Real cfl = 0.1;

  if (argc > 1) {
    nx = std::stoi(argv[1]);
  }
  if (argc > 2) {
    cfl = std::stod(argv[2]);
  }

  run(nx, cfl);

  Kokkos::finalize();
}
