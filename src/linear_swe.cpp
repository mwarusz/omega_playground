#include <hexagonal_periodic_mesh.hpp>
#include <iostream>
#include <vector>

static constexpr Real h0 = 1;
static constexpr Real g0 = 10;
static constexpr Real f0 = 0;

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
      htend(icell) = -h0 * accum / mesh.area_cell(icell);
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
      Real grad_h = g0 * (h(icell1) - h(icell0)) / mesh.dc_edge(iedge);

      vtend(iedge) = -f0 * vt - g0 * grad_h;
  });
}

void run(Int n) {
    HexagonalPeriodicMesh mesh(n, n);
    
    Real1d h("h", mesh.ncells);
    Real1d v("v", mesh.nedges);
  
    parallel_for("init_h", mesh.ncells, YAKL_LAMBDA (Int icell) {
        h(icell) = h0;
    });
    
    parallel_for("init_v", mesh.nedges, YAKL_LAMBDA (Int iedge) {
        v(iedge) = 0;
    });
    
    Real1d htend("htend", mesh.ncells);
    Real1d vtend("vtend", mesh.nedges);
    yakl::memset(htend, 0);
    yakl::memset(vtend, 0);

    Real timeend = 1;
    Real cfl = 0.2;
    Real dt = cfl * mesh.dc / std::sqrt(g0 * h0);
    Int numberofsteps = std::ceil(timeend / dt);
    dt = timeend / numberofsteps;

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

    std::cout << yakl::intrinsics::maxval(h) << std::endl;
    std::cout << yakl::intrinsics::maxval(v) << std::endl;
}


int main() {
  yakl::init();
  run(32);
  yakl::finalize();
}
