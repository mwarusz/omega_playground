#include <omega.hpp>
#include <iostream>
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
    using std::sin;
    using std::exp;

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
    using std::sin;
    using std::exp;

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
    using std::sin;
    using std::exp;

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

    ShallowWater shallow_water(mesh, double_vortex.coriolis, double_vortex.g);

    LSRKStepper stepper(shallow_water);
    
    Real timeend = 200 * 1000;
    Real dt = cfl * mesh.dc / std::sqrt(shallow_water.grav * double_vortex.h0);
    Int numberofsteps = std::ceil(timeend / dt);
    dt = timeend / numberofsteps;
    
    Real1d h("h", mesh.ncells);
    Real1d v("v", mesh.nedges);
  
    parallel_for("init_h", mesh.ncells, YAKL_LAMBDA (Int icell) {
        Real x = mesh.x_cell(icell);
        Real y = mesh.y_cell(icell);
        h(icell) = double_vortex.h(x, y);
    });
    
    parallel_for("init_v", mesh.nedges, YAKL_LAMBDA (Int iedge) {
        Real x = mesh.x_edge(iedge);
        Real y = mesh.y_edge(iedge);
        Real nx = std::cos(mesh.angle_edge(iedge));
        Real ny = std::sin(mesh.angle_edge(iedge));
        Real vx = double_vortex.vx(x, y);
        Real vy = double_vortex.vy(x, y);
        v(iedge) = nx * vx + ny * vy;
    });

    Real mass0 = shallow_water.mass_integral(h);
    Real cir0 = shallow_water.circulation_integral(v);
    Real en0 = shallow_water.energy_integral(h, v);

    for (Int step = 0; step < numberofsteps; ++step) {
      Real t = step * dt;
      stepper.do_step(t, dt, h, v);
    }

    Real massf = shallow_water.mass_integral(h);
    Real cirf = shallow_water.circulation_integral(v);
    Real enf = shallow_water.energy_integral(h, v);
    
    std::cout << "Mass change: " << (massf - mass0) / mass0 << std::endl;
    std::cout << "Circulation change: " << (cirf - cir0) / cir0 << std::endl;
    std::cout << "Energy change: " << (enf - en0) / en0 << std::endl;
}


int main(int argc, char* argv[]) {
  yakl::init();

  Int nx = 32;
  Real cfl = 1;

  if (argc > 1) {
    nx = std::stoi(argv[1]);
  }
  if (argc > 2) {
    cfl = std::stod(argv[2]);
  }

  run(nx, cfl);

  yakl::finalize();
}
