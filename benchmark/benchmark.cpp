#include <iostream>
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

  YAKL_INLINE Real h(Real x, Real y) const {
    using std::exp;
    using std::sin;

    Real xprime1 = m_lx / (pi * m_sigmax) * sin(pi / m_lx * (x - m_xc1));
    Real yprime1 = m_ly / (pi * m_sigmay) * sin(pi / m_ly * (y - m_yc1));
    Real xprime2 = m_lx / (pi * m_sigmax) * sin(pi / m_lx * (x - m_xc2));
    Real yprime2 = m_ly / (pi * m_sigmay) * sin(pi / m_ly * (y - m_yc2));
    Real xprimeprime1 =
        m_lx / (2.0 * pi * m_sigmax) * sin(2.0 * pi / m_lx * (x - m_xc1));
    Real yprimeprime1 =
        m_ly / (2.0 * pi * m_sigmay) * sin(2.0 * pi / m_ly * (y - m_yc1));
    Real xprimeprime2 =
        m_lx / (2.0 * pi * m_sigmax) * sin(2.0 * pi / m_lx * (x - m_xc2));
    Real yprimeprime2 =
        m_ly / (2.0 * pi * m_sigmay) * sin(2.0 * pi / m_ly * (y - m_yc2));

    return m_h0 - m_dh * (exp(-0.5 * (xprime1 * xprime1 + yprime1 * yprime1)) +
                          exp(-0.5 * (xprime2 * xprime2 + yprime2 * yprime2)) -
                          4. * pi * m_sigmax * m_sigmay / m_lx / m_ly);
  }

  YAKL_INLINE Real vx(Real x, Real y) const {
    using std::exp;
    using std::sin;

    Real xprime1 = m_lx / (pi * m_sigmax) * sin(pi / m_lx * (x - m_xc1));
    Real yprime1 = m_ly / (pi * m_sigmay) * sin(pi / m_ly * (y - m_yc1));
    Real xprime2 = m_lx / (pi * m_sigmax) * sin(pi / m_lx * (x - m_xc2));
    Real yprime2 = m_ly / (pi * m_sigmay) * sin(pi / m_ly * (y - m_yc2));
    Real xprimeprime1 =
        m_lx / (2.0 * pi * m_sigmax) * sin(2.0 * pi / m_lx * (x - m_xc1));
    Real yprimeprime1 =
        m_ly / (2.0 * pi * m_sigmay) * sin(2.0 * pi / m_ly * (y - m_yc1));
    Real xprimeprime2 =
        m_lx / (2.0 * pi * m_sigmax) * sin(2.0 * pi / m_lx * (x - m_xc2));
    Real yprimeprime2 =
        m_ly / (2.0 * pi * m_sigmay) * sin(2.0 * pi / m_ly * (y - m_yc2));

    Real vx =
        -m_g * m_dh / m_coriolis / m_sigmay *
        (yprimeprime1 * exp(-0.5 * (xprime1 * xprime1 + yprime1 * yprime1)) +
         yprimeprime2 * exp(-0.5 * (xprime2 * xprime2 + yprime2 * yprime2)));
    return vx;
  }

  YAKL_INLINE Real vy(Real x, Real y) const {
    using std::exp;
    using std::sin;

    Real xprime1 = m_lx / (pi * m_sigmax) * sin(pi / m_lx * (x - m_xc1));
    Real yprime1 = m_ly / (pi * m_sigmay) * sin(pi / m_ly * (y - m_yc1));
    Real xprime2 = m_lx / (pi * m_sigmax) * sin(pi / m_lx * (x - m_xc2));
    Real yprime2 = m_ly / (pi * m_sigmay) * sin(pi / m_ly * (y - m_yc2));
    Real xprimeprime1 =
        m_lx / (2.0 * pi * m_sigmax) * sin(2.0 * pi / m_lx * (x - m_xc1));
    Real yprimeprime1 =
        m_ly / (2.0 * pi * m_sigmay) * sin(2.0 * pi / m_ly * (y - m_yc1));
    Real xprimeprime2 =
        m_lx / (2.0 * pi * m_sigmax) * sin(2.0 * pi / m_lx * (x - m_xc2));
    Real yprimeprime2 =
        m_ly / (2.0 * pi * m_sigmay) * sin(2.0 * pi / m_ly * (y - m_yc2));

    Real vy =
        m_g * m_dh / m_coriolis / m_sigmax *
        (xprimeprime1 * exp(-0.5 * (xprime1 * xprime1 + yprime1 * yprime1)) +
         xprimeprime2 * exp(-0.5 * (xprime2 * xprime2 + yprime2 * yprime2)));
    return vy;
  }
};

void run(Int nx, Int nlayers, Int ntracers, Int nsteps) {
  DoubleVortex double_vortex;

  Real dc = double_vortex.m_lx / nx;
  Int ny = nx;
  PlanarHexagonalMesh mesh(nx, ny, dc, nlayers);

  ShallowWaterParams params;
  params.m_ntracers = ntracers;
  params.m_f0 = double_vortex.m_coriolis;
  params.m_grav = double_vortex.m_g;
  // just to turn these on
  params.m_visc_del2 = 1e-5;
  params.m_visc_del4 = 1e-5;
  params.m_eddy_diff2 = 1e-5;
  params.m_eddy_diff4 = 1e-5;

  ShallowWaterModel shallow_water(mesh, params);

  ShallowWaterState state(shallow_water);

  LSRKStepper stepper(shallow_water, 1);

  Real cfl = 0.1;
  Real dt =
      cfl * mesh.m_dc / std::sqrt(shallow_water.m_grav * double_vortex.m_h0);
  Int numberofsteps = nsteps;

  auto &h_cell = state.m_h_cell;
  auto &tr_cell = state.m_tr_cell;
  parallel_for(
      "init_h", SimpleBounds<2>(mesh.m_ncells, mesh.m_nlayers),
      YAKL_LAMBDA(Int icell, Int k) {
        Real x = mesh.m_x_cell(icell);
        Real y = mesh.m_y_cell(icell);
        h_cell(icell, k) = double_vortex.h(x, y);
      });

  if (ntracers > 0) {
    parallel_for(
        "init_tr", SimpleBounds<3>(ntracers, mesh.m_ncells, mesh.m_nlayers),
        YAKL_LAMBDA(Int l, Int icell, Int k) { tr_cell(l, icell, k) = 1000; });
  }

  auto &vn_edge = state.m_vn_edge;
  parallel_for(
      "init_vn", SimpleBounds<2>(mesh.m_nedges, mesh.m_nlayers),
      YAKL_LAMBDA(Int iedge, Int k) {
        Real x = mesh.m_x_edge(iedge);
        Real y = mesh.m_y_edge(iedge);
        Real nx = std::cos(mesh.m_angle_edge(iedge));
        Real ny = std::sin(mesh.m_angle_edge(iedge));
        Real vx = double_vortex.vx(x, y);
        Real vy = double_vortex.vy(x, y);
        vn_edge(iedge, k) = nx * vx + ny * vy;
      });

  for (Int step = 0; step < numberofsteps; ++step) {
    Real t = step * dt;
    stepper.do_step(t, dt, state);
  }

  std::cout << "Final h: " << yakl::intrinsics::minval(h_cell) << " "
            << yakl::intrinsics::maxval(h_cell) << std::endl;
  std::cout << "Final vn: " << yakl::intrinsics::minval(vn_edge) << " "
            << yakl::intrinsics::maxval(vn_edge) << std::endl;
  std::cout << "Final tr: " << yakl::intrinsics::minval(tr_cell) << " "
            << yakl::intrinsics::maxval(tr_cell) << std::endl;
}

int main(int argc, char *argv[]) {
  yakl::init();

  Int nx = argc > 1 ? std::stoi(argv[1]) : 64;
  Int nlayers = argc > 2 ? std::stod(argv[2]) : 64;
  Int ntracers = argc > 3 ? std::stod(argv[3]) : 1;
  Int nsteps = argc > 4 ? std::stod(argv[4]) : 10;

  run(nx, nlayers, ntracers, nsteps);

  yakl::finalize();
}
