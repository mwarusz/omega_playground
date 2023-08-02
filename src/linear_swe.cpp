#include <hexagonal_periodic_mesh.hpp>
#include <iostream>
#include <vector>

static constexpr Real h0 = 1000;
static constexpr Real eta0 = 1;
static constexpr Real g0 = 10;
static constexpr Real f0 = 1;

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

void compute_tendency(
    Real1d htend, Real1d vtend,
    Real1d h, Real1d v,
    const HexagonalPeriodicMesh &mesh)  {

  parallel_for("compute_h_tendency", mesh.ncells, YAKL_LAMBDA (Int icell) {
      Real accum = -0;
      for (Int j = 0; j < mesh.nedges_on_cell(icell); ++j) {
        Int iedge = mesh.edges_on_cell(icell, j);
        accum += mesh.dv_edge(iedge) * mesh.orient_on_cell(icell, j) * v(iedge);
      }
      htend(icell) += -h0 * accum / mesh.area_cell(icell);
  });

  parallel_for("compute_v_tendency", mesh.nedges, YAKL_LAMBDA (Int iedge) {
      Real vt = -0;
      for (Int j = 0; j < mesh.nedges_on_edge(iedge); ++j) {
        Int iedge2 = mesh.edges_on_edge(iedge, j);
        vt += mesh.weights_on_edge(iedge, j) * mesh.dv_edge(iedge2) * v(iedge2);
      }
      vt /= mesh.dc_edge(iedge);

      Int icell0 = mesh.cells_on_edge(iedge, 0);
      Int icell1 = mesh.cells_on_edge(iedge, 1);
      Real grad_h = (h(icell1) - h(icell0)) / mesh.dc_edge(iedge);

      vtend(iedge) += f0 * vt - g0 * grad_h;
  });
}

void run(Int n) {
    Real lx = 1000;
    Real ly = std::sqrt(3) / 2 * lx;
    Real kx = 2 * (2 * pi / lx);
    Real ky = 2 * (2 * pi / ly);
    Real omega = std::sqrt(f0 * f0 + g0 * h0 * (kx * kx + ky * ky));

    HexagonalPeriodicMesh mesh(n, n, lx / n);
    
    Real timeend = 1;
    Real cfl = 0.2;
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
    
    Real1d htend("htend", mesh.ncells);
    Real1d vtend("vtend", mesh.nedges);
    yakl::memset(htend, 0);
    yakl::memset(vtend, 0);


    const Int nstages = 5;
    Real rka[] = {
        0., -567301805773. / 1357537059087., -2404267990393. / 2016746695238.,
        -3550918686646. / 2091501179385., -1275806237668. / 842570457699.};

    Real rkb[] = {
        1432997174477. / 9575080441755., 5161836677717. / 13612068292357.,
        1720146321549. / 2090206949498., 3134564353537. / 4481467310338.,
        2277821191437. / 14882151754819.};

    for (Int step = 0; step < numberofsteps; ++step) {
      for (Int stage = 0; stage < nstages; ++stage) {
        
        parallel_for("lsrk1_h", mesh.ncells, YAKL_LAMBDA (Int icell) {
            htend(icell) *= rka[stage];
        });
        parallel_for("lsrk1_v", mesh.nedges, YAKL_LAMBDA (Int iedge) {
            vtend(iedge) *= rka[stage];
        });

        compute_tendency(htend, vtend, h, v, mesh);
        
        parallel_for("lsrk2_h", mesh.ncells, YAKL_LAMBDA (Int icell) {
            h(icell) += dt * rkb[stage] * htend(icell);
        });
        parallel_for("lsrk2_v", mesh.nedges, YAKL_LAMBDA (Int iedge) {
            v(iedge) += dt * rkb[stage] * vtend(iedge);
        });
      }
    }

    parallel_for("compute_error", mesh.ncells, YAKL_LAMBDA (Int icell) {
        hexact(icell) -= h(icell);
        hexact(icell) = std::abs(hexact(icell));
    });
    
    std::cout << yakl::intrinsics::maxval(h) << std::endl;
    std::cout << yakl::intrinsics::maxval(v) << std::endl;
    std::cout << yakl::intrinsics::maxval(hexact) << std::endl;
}


int main() {
  yakl::init();
  run(32);
  yakl::finalize();
}
