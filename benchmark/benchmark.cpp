#include <chrono>
#include <iostream>
#include <memory>
#include <omega.hpp>
#include <string>
#ifdef BENCHMARK_PROFILE_CUDA
#include <cuda_profiler_api.h>
#endif

using namespace omega;

struct DoubleVortex {
  Real m_g = 9.80616;
  Real m_lx = 5000000;
  Real m_ly = 5000000 * std::sqrt(3) / 2;
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

void run(Int nx, Int nlayers, Int ntracers, Int nsteps) {
  DoubleVortex double_vortex;

  Real dc = double_vortex.m_lx / nx;
  Int ny = nx;
  auto mesh = std::make_unique<PlanarHexagonalMesh>(nx, ny, dc, nlayers);

  ShallowWaterParams params;
  params.m_ntracers = ntracers;
  params.m_f0 = double_vortex.m_coriolis;
  params.m_grav = double_vortex.m_g;
  // just to turn these on
  params.m_visc_del2 = 1e-5;
  params.m_visc_del4 = 1e-5;
  params.m_eddy_diff2 = 1e-5;
  params.m_eddy_diff4 = 1e-5;

  ShallowWaterModel shallow_water(mesh.get(), params);

  ShallowWaterState state(mesh.get(), params);

  LSRKStepper stepper(shallow_water, nsteps == 1 ? 1 : 5);

  Real cfl = 0.1;
  Real dt = cfl * mesh->m_dc / std::sqrt(params.m_grav * double_vortex.m_h0);
  Int numberofsteps = nsteps;

  auto &h_cell = state.m_h_cell;
  auto &tr_cell = state.m_tr_cell;
  OMEGA_SCOPE(x_cell, mesh->m_x_cell);
  OMEGA_SCOPE(y_cell, mesh->m_y_cell);
  parallel_for(
      "init_h", MDRangePolicy<2>({0, 0}, {mesh->m_ncells, mesh->m_nlayers}),
      KOKKOS_LAMBDA(Int icell, Int k) {
        Real x = x_cell(icell);
        Real y = y_cell(icell);
        h_cell(icell, k) = double_vortex.h(x, y);
      });

  if (ntracers > 0) {
    parallel_for(
        "init_tr",
        MDRangePolicy<3>({0, 0, 0},
                         {ntracers, mesh->m_ncells, mesh->m_nlayers}),
        KOKKOS_LAMBDA(Int l, Int icell, Int k) {
          tr_cell(l, icell, k) = 1000;
        });
  }

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

  stepper.do_step(0, dt, state);

#ifdef BENCHMARK_PROFILE_CUDA
  cudaProfilerStart();
#endif

  timer_start("time_integration");

  Kokkos::fence();
  auto ts = std::chrono::steady_clock::now();
  for (Int step = 0; step < numberofsteps; ++step) {
    Real t = (step + 1) * dt;
    stepper.do_step(t, dt, state);
  }
  Kokkos::fence();
  auto te = std::chrono::steady_clock::now();
  auto time_loop_second = std::chrono::duration<double>(te - ts).count();

  timer_end("time_integration");

#ifdef BENCHMARK_PROFILE_CUDA
  cudaProfilerStop();
#endif

  // std::cout << "Final h: " << yakl::intrinsics::minval(h_cell) << " "
  //           << yakl::intrinsics::maxval(h_cell) << std::endl;
  // std::cout << "Final vn: " << yakl::intrinsics::minval(vn_edge) << " "
  //           << yakl::intrinsics::maxval(vn_edge) << std::endl;
  // if (ntracers > 0) {
  //   std::cout << "Final tr: " << yakl::intrinsics::minval(tr_cell) << " "
  //             << yakl::intrinsics::maxval(tr_cell) << std::endl;
  // }

  std::cerr << time_loop_second << std::endl;
}

int main(int argc, char *argv[]) {
  Kokkos::initialize();

  Int nx = argc > 1 ? std::stoi(argv[1]) : 64;
  Int nlayers = argc > 2 ? std::stod(argv[2]) : 64;
  Int ntracers = argc > 3 ? std::stod(argv[3]) : 1;
  Int nsteps = argc > 4 ? std::stod(argv[4]) : 10;

  run(nx, nlayers, ntracers, nsteps);

  Kokkos::finalize();
}
