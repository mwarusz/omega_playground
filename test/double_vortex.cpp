#include <iostream>
#include <omega.hpp>
#include <string>

using namespace omega;

struct DoubleVortex {
  Real g = 9.80616;
  Real lx = 5000000;
  Real ly = 5000000;
  Real coriolis = 0.00006147;
  Real h0 = 750;
  Real ox = 0.1;
  Real oy = 0.1;
  Real sigmax = 3. / 40. * lx;
  Real sigmay = 3. / 40. * ly;
  Real dh = 75;
  Real xc1 = (0.5 - ox) * lx;
  Real yc1 = (0.5 - oy) * ly;
  Real xc2 = (0.5 + ox) * lx;
  Real yc2 = (0.5 + oy) * ly;

  YAKL_INLINE Real h(Real x, Real y) const {
    using std::exp;
    using std::sin;

    Real xprime1 = lx / (pi * sigmax) * sin(pi / lx * (x - xc1));
    Real yprime1 = ly / (pi * sigmay) * sin(pi / ly * (y - yc1));
    Real xprime2 = lx / (pi * sigmax) * sin(pi / lx * (x - xc2));
    Real yprime2 = ly / (pi * sigmay) * sin(pi / ly * (y - yc2));
    Real xprimeprime1 =
        lx / (2.0 * pi * sigmax) * sin(2.0 * pi / lx * (x - xc1));
    Real yprimeprime1 =
        ly / (2.0 * pi * sigmay) * sin(2.0 * pi / ly * (y - yc1));
    Real xprimeprime2 =
        lx / (2.0 * pi * sigmax) * sin(2.0 * pi / lx * (x - xc2));
    Real yprimeprime2 =
        ly / (2.0 * pi * sigmay) * sin(2.0 * pi / ly * (y - yc2));

    return h0 - dh * (exp(-0.5 * (xprime1 * xprime1 + yprime1 * yprime1)) +
                      exp(-0.5 * (xprime2 * xprime2 + yprime2 * yprime2)) -
                      4. * pi * sigmax * sigmay / lx / ly);
  }

  YAKL_INLINE Real vx(Real x, Real y) const {
    using std::exp;
    using std::sin;

    Real xprime1 = lx / (pi * sigmax) * sin(pi / lx * (x - xc1));
    Real yprime1 = ly / (pi * sigmay) * sin(pi / ly * (y - yc1));
    Real xprime2 = lx / (pi * sigmax) * sin(pi / lx * (x - xc2));
    Real yprime2 = ly / (pi * sigmay) * sin(pi / ly * (y - yc2));
    Real xprimeprime1 =
        lx / (2.0 * pi * sigmax) * sin(2.0 * pi / lx * (x - xc1));
    Real yprimeprime1 =
        ly / (2.0 * pi * sigmay) * sin(2.0 * pi / ly * (y - yc1));
    Real xprimeprime2 =
        lx / (2.0 * pi * sigmax) * sin(2.0 * pi / lx * (x - xc2));
    Real yprimeprime2 =
        ly / (2.0 * pi * sigmay) * sin(2.0 * pi / ly * (y - yc2));

    Real vx =
        -g * dh / coriolis / sigmay *
        (yprimeprime1 * exp(-0.5 * (xprime1 * xprime1 + yprime1 * yprime1)) +
         yprimeprime2 * exp(-0.5 * (xprime2 * xprime2 + yprime2 * yprime2)));
    return vx;
  }

  YAKL_INLINE Real vy(Real x, Real y) const {
    using std::exp;
    using std::sin;

    Real xprime1 = lx / (pi * sigmax) * sin(pi / lx * (x - xc1));
    Real yprime1 = ly / (pi * sigmay) * sin(pi / ly * (y - yc1));
    Real xprime2 = lx / (pi * sigmax) * sin(pi / lx * (x - xc2));
    Real yprime2 = ly / (pi * sigmay) * sin(pi / ly * (y - yc2));
    Real xprimeprime1 =
        lx / (2.0 * pi * sigmax) * sin(2.0 * pi / lx * (x - xc1));
    Real yprimeprime1 =
        ly / (2.0 * pi * sigmay) * sin(2.0 * pi / ly * (y - yc1));
    Real xprimeprime2 =
        lx / (2.0 * pi * sigmax) * sin(2.0 * pi / lx * (x - xc2));
    Real yprimeprime2 =
        ly / (2.0 * pi * sigmay) * sin(2.0 * pi / ly * (y - yc2));

    Real vy =
        g * dh / coriolis / sigmax *
        (xprimeprime1 * exp(-0.5 * (xprime1 * xprime1 + yprime1 * yprime1)) +
         xprimeprime2 * exp(-0.5 * (xprime2 * xprime2 + yprime2 * yprime2)));
    return vy;
  }
};

void run(Int nx, Real cfl) {
  DoubleVortex double_vortex;
  Real dc = double_vortex.lx / nx;
  Int ny = std::ceil(double_vortex.ly / (dc * std::sqrt(3) / 2));
  PlanarHexagonalMesh mesh(nx, ny, dc);

  ShallowWaterParams params;
  params.f0 = double_vortex.coriolis;
  params.grav = double_vortex.g;
  ShallowWater shallow_water(mesh, params);

  LSRKStepper stepper(shallow_water);

  Real timeend = 200 * 1000;
  Real dt = cfl * mesh.dc / std::sqrt(shallow_water.grav * double_vortex.h0);
  Int numberofsteps = std::ceil(timeend / dt);
  dt = timeend / numberofsteps;

  Real2d h_cell("h_cell", mesh.ncells, mesh.nlayers);
  Real2d vn_edge("vn_edge", mesh.nedges, mesh.nlayers);

  parallel_for(
      "init_h", SimpleBounds<2>(mesh.ncells, mesh.nlayers),
      YAKL_LAMBDA(Int icell, Int k) {
        Real x = mesh.x_cell(icell);
        Real y = mesh.y_cell(icell);
        h_cell(icell, k) = double_vortex.h(x, y);
      });

  parallel_for(
      "init_vn", SimpleBounds<2>(mesh.nedges, mesh.nlayers),
      YAKL_LAMBDA(Int iedge, Int k) {
        Real x = mesh.x_edge(iedge);
        Real y = mesh.y_edge(iedge);
        Real nx = std::cos(mesh.angle_edge(iedge));
        Real ny = std::sin(mesh.angle_edge(iedge));
        Real vx = double_vortex.vx(x, y);
        Real vy = double_vortex.vy(x, y);
        vn_edge(iedge, k) = nx * vx + ny * vy;
      });

  Real mass0 = shallow_water.mass_integral(h_cell);
  Real cir0 = shallow_water.circulation_integral(vn_edge);
  Real en0 = shallow_water.energy_integral(h_cell, vn_edge);

  std::cout << "Initial h: " << yakl::intrinsics::minval(h_cell) << " "
            << yakl::intrinsics::maxval(h_cell) << std::endl;
  std::cout << "Initial vn: " << yakl::intrinsics::minval(vn_edge) << " "
            << yakl::intrinsics::maxval(vn_edge) << std::endl;

  for (Int step = 0; step < numberofsteps; ++step) {
    Real t = step * dt;
    stepper.do_step(t, dt, h_cell, vn_edge);
  }

  std::cout << "Final h: " << yakl::intrinsics::minval(h_cell) << " "
            << yakl::intrinsics::maxval(h_cell) << std::endl;
  std::cout << "Final vn: " << yakl::intrinsics::minval(vn_edge) << " "
            << yakl::intrinsics::maxval(vn_edge) << std::endl;

  Real massf = shallow_water.mass_integral(h_cell);
  Real cirf = shallow_water.circulation_integral(vn_edge);
  Real enf = shallow_water.energy_integral(h_cell, vn_edge);

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
  yakl::init();

  Int nx = 25;
  Real cfl = 0.1;

  if (argc > 1) {
    nx = std::stoi(argv[1]);
  }
  if (argc > 2) {
    cfl = std::stod(argv[2]);
  }

  run(nx, cfl);

  yakl::finalize();
}
