#include <iostream>
#include <omega.hpp>
#include <vector>

using namespace omega;

bool check_rate(Real rate, Real expected_rate, Real atol) {
  return std::abs(rate - expected_rate) < atol && !std::isnan(rate);
}

struct ManufacturedSolution {
  Real m_grav = 9.80616;
  Real m_f0 = 1e-4;
  Real m_lx = 10000 * 1e3;
  Real m_ly = std::sqrt(3) / 2 * m_lx;
  Real m_eta0 = 1;
  Real m_h0 = 1000;
  Int m_mx = 2;
  Int m_my = 2;
  Real m_kx = m_mx * (2 * pi / m_lx);
  Real m_ky = m_my * (2 * pi / m_ly);
  Real m_omega = std::sqrt(m_grav * m_h0 * (m_kx * m_kx + m_ky * m_ky));

  YAKL_INLINE Real h(Real x, Real y, Real t) const {
    return m_h0 + m_eta0 * std::sin(m_kx * x + m_ky * y - m_omega * t);
  }

  YAKL_INLINE Real vx(Real x, Real y, Real t) const {
    return m_eta0 * std::cos(m_kx * x + m_ky * y - m_omega * t);
  }

  YAKL_INLINE Real vy(Real x, Real y, Real t) const {
    return m_eta0 * std::cos(m_kx * x + m_ky * y - m_omega * t);
  }

  YAKL_INLINE Real h_tend(Real x, Real y, Real t) const {
    using std::cos;
    using std::sin;

    Real phi = m_kx * x + m_ky * y - m_omega * t;
    return m_eta0 * (-m_h0 * (m_kx + m_ky) * sin(phi) - m_omega * cos(phi) +
                     m_eta0 * (m_kx + m_ky) * cos(2 * phi));
  }

  YAKL_INLINE Real vx_tend(Real x, Real y, Real t) const {
    using std::cos;
    using std::sin;

    Real phi = m_kx * x + m_ky * y - m_omega * t;
    return m_eta0 * ((-m_f0 + m_grav * m_kx) * cos(phi) + m_omega * sin(phi) -
                     m_eta0 * (m_kx + m_ky) * sin(2 * phi) / 2);
  }

  YAKL_INLINE Real vy_tend(Real x, Real y, Real t) const {
    using std::cos;
    using std::sin;

    Real phi = m_kx * x + m_ky * y - m_omega * t;
    return m_eta0 * ((m_f0 + m_grav * m_ky) * cos(phi) + m_omega * sin(phi) -
                     m_eta0 * (m_kx + m_ky) * sin(2 * phi) / 2);
  }
};

struct ManufacturedShallowWaterModel : ShallowWaterModel {
  ManufacturedSolution m_manufactured_solution;

  ManufacturedShallowWaterModel(
      PlanarHexagonalMesh &mesh,
      const ManufacturedSolution &manufactured_solution)
      : ShallowWaterModel(mesh,
                          ShallowWaterParams{manufactured_solution.m_f0,
                                             manufactured_solution.m_grav, 0}),
        m_manufactured_solution(manufactured_solution) {}

  void additional_tendency(Real2d h_tend_cell, Real2d vn_tend_edge,
                           RealConst2d h_cell, RealConst2d vn_edge,
                           Real t) const override {
    using std::cos;
    using std::sin;

    YAKL_SCOPE(manufactured_solution, m_manufactured_solution);

    YAKL_SCOPE(x_cell, m_mesh->m_x_cell);
    YAKL_SCOPE(y_cell, m_mesh->m_y_cell);
    YAKL_SCOPE(max_level_cell, m_mesh->m_max_level_cell);
    parallel_for(
        "manufactured_htend", m_mesh->m_ncells, YAKL_LAMBDA(Int icell) {
          for (Int k = 0; k < max_level_cell(icell); ++k) {
            Real x = x_cell(icell);
            Real y = y_cell(icell);
            h_tend_cell(icell, k) += manufactured_solution.h_tend(x, y, t);
          }
        });

    YAKL_SCOPE(x_edge, m_mesh->m_x_edge);
    YAKL_SCOPE(y_edge, m_mesh->m_y_edge);
    YAKL_SCOPE(angle_edge, m_mesh->m_angle_edge);
    YAKL_SCOPE(max_level_edge_top, m_mesh->m_max_level_edge_top);
    parallel_for(
        "manufactured_vtend", m_mesh->m_nedges, YAKL_LAMBDA(Int iedge) {
          for (Int k = 0; k < max_level_edge_top(iedge); ++k) {
            Real x = x_edge(iedge);
            Real y = y_edge(iedge);

            Real nx = std::cos(angle_edge(iedge));
            Real ny = std::sin(angle_edge(iedge));

            Real vx_tend = manufactured_solution.vx_tend(x, y, t);
            Real vy_tend = manufactured_solution.vy_tend(x, y, t);

            vn_tend_edge(iedge, k) += nx * vx_tend + ny * vy_tend;
          }
        });
  }
};

Real run(Int n) {
  ManufacturedSolution manufactured_solution;

  PlanarHexagonalMesh mesh(n, n, manufactured_solution.m_lx / n);

  ManufacturedShallowWaterModel shallow_water(mesh, manufactured_solution);

  ShallowWaterState state(shallow_water);

  RK4Stepper stepper(shallow_water);

  Real timeend = 10 * 60 * 60;
  Real dt_per_km = 3;
  Real dt = dt_per_km * mesh.m_dc / 1e3;
  Int numberofsteps = std::ceil(timeend / dt);
  dt = timeend / numberofsteps;

  auto &h_cell = state.m_h_cell;
  Real2d hexact_cell("hexact_cell", mesh.m_ncells, mesh.m_nlayers);
  parallel_for(
      "init_h", SimpleBounds<2>(mesh.m_ncells, mesh.m_nlayers),
      YAKL_LAMBDA(Int icell, Int k) {
        Real x = mesh.m_x_cell(icell);
        Real y = mesh.m_y_cell(icell);
        h_cell(icell, k) = manufactured_solution.h(x, y, 0);
        hexact_cell(icell, k) = manufactured_solution.h(x, y, timeend);
      });

  auto &vn_edge = state.m_vn_edge;
  parallel_for(
      "init_vn", SimpleBounds<2>(mesh.m_nedges, mesh.m_nlayers),
      YAKL_LAMBDA(Int iedge, Int k) {
        Real x = mesh.m_x_edge(iedge);
        Real y = mesh.m_y_edge(iedge);
        Real nx = std::cos(mesh.m_angle_edge(iedge));
        Real ny = std::sin(mesh.m_angle_edge(iedge));
        Real vx = manufactured_solution.vx(x, y, 0);
        Real vy = manufactured_solution.vy(x, y, 0);
        vn_edge(iedge, k) = nx * vx + ny * vy;
      });

  for (Int step = 0; step < numberofsteps; ++step) {
    Real t = step * dt;
    stepper.do_step(t, dt, state);
  }

  parallel_for(
      "compute_error", SimpleBounds<2>(mesh.m_ncells, mesh.m_nlayers),
      YAKL_LAMBDA(Int icell, Int k) {
        hexact_cell(icell, k) -= h_cell(icell, k);
        hexact_cell(icell, k) *= hexact_cell(icell, k);
      });

  return std::sqrt(yakl::intrinsics::sum(hexact_cell) /
                   (mesh.m_nx * mesh.m_ny));
}

int main() {
  yakl::init();

  Int nlevels = 2;
  std::vector<Real> err(nlevels);
  Int nx = 50;

  for (Int l = 0; l < nlevels; ++l) {
    err[l] = run(nx);
    nx *= 2;
  }

  if (nlevels > 1) {
    std::vector<Real> rate(nlevels - 1);
    std::cout << "Manufactured solution convergence" << std::endl;
    for (Int l = 0; l < nlevels; ++l) {
      std::cout << l << " " << err[l];
      if (l > 0) {
        rate[l - 1] = std::log2(err[l - 1] / err[l]);
        std::cout << " " << rate[l - 1];
      }
      std::cout << std::endl;
    }

    if (!check_rate(rate.back(), 2, 0.05)) {
      throw std::runtime_error(
          "Manufactured solution is not converging at the right rate");
    }
  }

  yakl::finalize();
}
