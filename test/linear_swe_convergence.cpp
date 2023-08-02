#include <omega.hpp>
#include <iostream>
#include <vector>

using namespace omega;

static constexpr Real h0 = 1000;
static constexpr Real eta0 = 1;
static constexpr Real g0 = 10;
static constexpr Real f0 = 1e-4;

Real compute_energy(Real1d h, Real1d v, const PlanarHexagonalMesh &mesh) {
  Real1d cell_tmp("cell_tmp", mesh.ncells);
  parallel_for("compute_energy_1", mesh.ncells, YAKL_LAMBDA (Int icell) {
      Real K = 0;
      for (Int j = 0; j < mesh.nedges_on_cell(icell); ++j) {
        Int iedge = mesh.edges_on_cell(icell, j);
        Real area_edge = mesh.dv_edge(iedge) * mesh.dc_edge(iedge);
        K += area_edge * v(iedge) * v(iedge) / 4;
      }
      K /= mesh.area_cell(icell);

      //cell_tmp(icell) = mesh.area_cell(icell) * (g0 * h(icell) * h(icell) / 2 + h0 * K);
      cell_tmp(icell) = mesh.area_cell(icell) * (g0 * h(icell) * h(icell) / 2);
  });
  
  Real1d edge_tmp("edge_tmp", mesh.nedges);
  parallel_for("compute_energy_2", mesh.nedges, YAKL_LAMBDA (Int iedge) {
      Real vt = -0;
      for (Int j = 0; j < mesh.nedges_on_edge(iedge); ++j) {
        Int iedge2 = mesh.edges_on_edge(iedge, j);
        vt += mesh.weights_on_edge(iedge, j) * mesh.dv_edge(iedge2) * v(iedge2);
      }
      vt /= mesh.dc_edge(iedge);

      edge_tmp(iedge) = h0 * v(iedge) * v(iedge) * mesh.dc_edge(iedge) * mesh.dv_edge(iedge) / 2;
  });

  return yakl::intrinsics::sum(cell_tmp) + yakl::intrinsics::sum(edge_tmp);
  //return yakl::intrinsics::sum(cell_tmp);
}

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
    LinearShallowWater sw(mesh, h0, f0, g0);
    
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

    Real en0 = compute_energy(h, v, mesh);

    for (Int step = 0; step < numberofsteps; ++step) {
      for (Int stage = 0; stage < nstages; ++stage) {
        
        parallel_for("lsrk1_h", mesh.ncells, YAKL_LAMBDA (Int icell) {
            htend(icell) *= rka[stage];
        });
        parallel_for("lsrk1_v", mesh.nedges, YAKL_LAMBDA (Int iedge) {
            vtend(iedge) *= rka[stage];
        });

        sw.compute_tendency(htend, vtend, h, v);
        
        parallel_for("lsrk2_h", mesh.ncells, YAKL_LAMBDA (Int icell) {
            h(icell) += dt * rkb[stage] * htend(icell);
        });
        parallel_for("lsrk2_v", mesh.nedges, YAKL_LAMBDA (Int iedge) {
            v(iedge) += dt * rkb[stage] * vtend(iedge);
        });
      }
    }
    
    Real enf = compute_energy(h, v, mesh);
    std::cout << "Energy change: " << (enf - en0) / en0 << std::endl;

    parallel_for("compute_error", mesh.ncells, YAKL_LAMBDA (Int icell) {
        hexact(icell) -= h(icell);
        hexact(icell) = std::abs(hexact(icell));
    });
    
    //std::cout << yakl::intrinsics::maxval(h) << std::endl;
    //std::cout << yakl::intrinsics::maxval(v) << std::endl;
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
