#include <iostream>
#include <omega.hpp>
#include <vector>

using namespace omega;

bool check_rate(Real rate, Real expected_rate, Real atol) {
  return std::abs(rate - expected_rate) < atol && !std::isnan(rate);
}

Real error_gradient(const PlanarHexagonalMesh &mesh) {
  using std::cos;
  using std::sin;

  Real Lx = mesh.m_period_x;
  Real Ly = mesh.m_period_y;

  Real1d input_field("input_field", mesh.m_ncells);
  Real1d grad_field("grad_field", mesh.m_nedges);
  Real1d exact_grad_field("exact_grad_field", mesh.m_nedges);

  parallel_for(
      RangePolicy(0, mesh.m_ncells), KOKKOS_LAMBDA(Int icell) {
        Real x = mesh.m_x_cell(icell);
        Real y = mesh.m_y_cell(icell);
        input_field(icell) = sin(2 * pi * x / Lx) * sin(2 * pi * y / Ly);
      });

  parallel_for(
      RangePolicy(0, mesh.m_nedges), KOKKOS_LAMBDA(Int iedge) {
        Int icell0 = mesh.m_cells_on_edge(iedge, 0);
        Int icell1 = mesh.m_cells_on_edge(iedge, 1);

        grad_field(iedge) =
            (input_field(icell1) - input_field(icell0)) / mesh.m_dc_edge(iedge);

        Real nx = cos(mesh.m_angle_edge(iedge));
        Real ny = sin(mesh.m_angle_edge(iedge));

        Real x = mesh.m_x_edge(iedge);
        Real y = mesh.m_y_edge(iedge);

        Real grad_x = 2 * pi / Lx * cos(2 * pi * x / Lx) * sin(2 * pi * y / Ly);
        Real grad_y = 2 * pi / Ly * sin(2 * pi * x / Lx) * cos(2 * pi * y / Ly);

        exact_grad_field(iedge) = nx * grad_x + ny * grad_y;

        exact_grad_field(iedge) -= grad_field(iedge);
        exact_grad_field(iedge) = std::abs(exact_grad_field(iedge));
      });

  Real errf;
  parallel_reduce(
      RangePolicy(0, mesh.m_nedges),
      KOKKOS_LAMBDA(Int iedge, Real & accum) {
        accum = std::max(exact_grad_field(iedge), accum);
      },
      Kokkos::Max<Real>(errf));

  return errf;
}

Real error_divergence(const PlanarHexagonalMesh &mesh) {
  using std::cos;
  using std::sin;

  Real Lx = mesh.m_period_x;
  Real Ly = mesh.m_period_y;

  Real1d input_field("input_field", mesh.m_nedges);
  Real1d div_field("div_field", mesh.m_ncells);
  Real1d exact_div_field("exact_div_field", mesh.m_ncells);

  parallel_for(
      RangePolicy(0, mesh.m_nedges), KOKKOS_LAMBDA(Int iedge) {
        Real nx = cos(mesh.m_angle_edge(iedge));
        Real ny = sin(mesh.m_angle_edge(iedge));

        Real x = mesh.m_x_edge(iedge);
        Real y = mesh.m_y_edge(iedge);

        Real v_x = sin(2 * pi * x / Lx) * cos(2 * pi * y / Ly);
        Real v_y = cos(2 * pi * x / Lx) * sin(2 * pi * y / Ly);

        input_field(iedge) = nx * v_x + ny * v_y;
      });

  parallel_for(
      RangePolicy(0, mesh.m_ncells), KOKKOS_LAMBDA(Int icell) {
        Real accum = -0;
        for (Int j = 0; j < mesh.m_nedges_on_cell(icell); ++j) {
          Int jedge = mesh.m_edges_on_cell(icell, j);
          accum += mesh.m_dv_edge(jedge) * mesh.m_edge_sign_on_cell(icell, j) *
                   input_field(jedge);
        }
        div_field(icell) = accum / mesh.m_area_cell(icell);

        Real x = mesh.m_x_cell(icell);
        Real y = mesh.m_y_cell(icell);
        exact_div_field(icell) = 2 * pi * (1. / Lx + 1. / Ly) *
                                 cos(2 * pi * x / Lx) * cos(2 * pi * y / Ly);

        exact_div_field(icell) -= div_field(icell);
        exact_div_field(icell) = std::abs(exact_div_field(icell));
      });

  Real errf;
  parallel_reduce(
      RangePolicy(0, mesh.m_ncells),
      KOKKOS_LAMBDA(Int icell, Real & accum) {
        accum = std::max(exact_div_field(icell), accum);
      },
      Kokkos::Max<Real>(errf));
  return errf;
}

Real error_curl(const PlanarHexagonalMesh &mesh) {
  using std::cos;
  using std::sin;

  Real Lx = mesh.m_period_x;
  Real Ly = mesh.m_period_y;

  Real1d input_field("input_field", mesh.m_nedges);

  Real1d curl_field("curl_field", mesh.m_nvertices);
  Real1d exact_curl_field("exact_curl_field", mesh.m_nvertices);

  parallel_for(
      RangePolicy(0, mesh.m_nedges), KOKKOS_LAMBDA(Int iedge) {
        Real nx = cos(mesh.m_angle_edge(iedge));
        Real ny = sin(mesh.m_angle_edge(iedge));

        Real x = mesh.m_x_edge(iedge);
        Real y = mesh.m_y_edge(iedge);

        Real v_x = sin(2 * pi * x / Lx) * cos(2 * pi * y / Ly);
        Real v_y = cos(2 * pi * x / Lx) * sin(2 * pi * y / Ly);

        input_field(iedge) = nx * v_x + ny * v_y;
      });

  parallel_for(
      RangePolicy(0, mesh.m_nvertices), KOKKOS_LAMBDA(Int ivertex) {
        Real accum = -0;
        for (Int j = 0; j < 3; ++j) {
          Int jedge = mesh.m_edges_on_vertex(ivertex, j);
          accum += mesh.m_dc_edge(jedge) *
                   mesh.m_edge_sign_on_vertex(ivertex, j) * input_field(jedge);
        }
        curl_field(ivertex) = accum / mesh.m_area_triangle(ivertex);

        Real x = mesh.m_x_vertex(ivertex);
        Real y = mesh.m_y_vertex(ivertex);
        exact_curl_field(ivertex) = 2 * pi * (-1. / Lx + 1. / Ly) *
                                    sin(2 * pi * x / Lx) * sin(2 * pi * y / Ly);

        exact_curl_field(ivertex) -= curl_field(ivertex);
        exact_curl_field(ivertex) = std::abs(exact_curl_field(ivertex));
      });

  Real errf;
  parallel_reduce(
      RangePolicy(0, mesh.m_nvertices),
      KOKKOS_LAMBDA(Int ivertex, Real & accum) {
        accum = std::max(exact_curl_field(ivertex), accum);
      },
      Kokkos::Max<Real>(errf));
  return errf;
}

Real error_reconstruction(const PlanarHexagonalMesh &mesh) {
  using std::cos;
  using std::sin;

  Real Lx = mesh.m_period_x;
  Real Ly = mesh.m_period_y;

  Real1d input_field("input_field", mesh.m_nedges);

  Real1d recon_field("recon_field", mesh.m_nedges);
  Real1d exact_recon_field("exact_recon_field", mesh.m_nedges);

  parallel_for(
      RangePolicy(0, mesh.m_nedges), KOKKOS_LAMBDA(Int iedge) {
        Real nx = cos(mesh.m_angle_edge(iedge));
        Real ny = sin(mesh.m_angle_edge(iedge));

        Real tx = -ny;
        Real ty = nx;

        Real x = mesh.m_x_edge(iedge);
        Real y = mesh.m_y_edge(iedge);

        Real v_x = sin(2 * pi * x / Lx) * cos(2 * pi * y / Ly);
        Real v_y = cos(2 * pi * x / Lx) * sin(2 * pi * y / Ly);

        input_field(iedge) = nx * v_x + ny * v_y;
        exact_recon_field(iedge) = tx * v_x + ty * v_y;
      });

  parallel_for(
      RangePolicy(0, mesh.m_nedges), KOKKOS_LAMBDA(Int iedge) {
        Int n = mesh.m_nedges_on_edge(iedge);
        Real accum = -0;
        for (Int j = 0; j < n; ++j) {
          Int iedge2 = mesh.m_edges_on_edge(iedge, j);
          accum += mesh.m_weights_on_edge(iedge, j) * input_field(iedge2);
        }
        recon_field(iedge) = accum;

        exact_recon_field(iedge) -= recon_field(iedge);
        exact_recon_field(iedge) = std::abs(exact_recon_field(iedge));
      });

  Real errf;
  parallel_reduce(
      RangePolicy(0, mesh.m_nedges),
      KOKKOS_LAMBDA(Int iedge, Real & accum) {
        accum = std::max(exact_recon_field(iedge), accum);
      },
      Kokkos::Max<Real>(errf));
  return errf;
}

void run(Int nlevels) {
  Int nx = 6;
  Real atol = 0.05;

  std::vector<Real> grad_err(nlevels);
  std::vector<Real> div_err(nlevels);
  std::vector<Real> curl_err(nlevels);
  std::vector<Real> recon_err(nlevels);

  for (Int l = 0; l < nlevels; ++l) {
    PlanarHexagonalMesh mesh(nx, nx);

    grad_err[l] = error_gradient(mesh);
    div_err[l] = error_divergence(mesh);
    curl_err[l] = error_curl(mesh);
    recon_err[l] = error_reconstruction(mesh);

    nx *= 2;
  }

  std::cout << "Gradient convergence" << std::endl;
  std::vector<Real> grad_rate(nlevels - 1);
  for (Int l = 0; l < nlevels; ++l) {
    std::cout << l << " " << grad_err[l];
    if (l > 0) {
      grad_rate[l - 1] = std::log2(grad_err[l - 1] / grad_err[l]);
      std::cout << " " << grad_rate[l - 1];
    }
    std::cout << std::endl;
  }
  if (!check_rate(grad_rate.back(), 2, atol)) {
    throw std::runtime_error("Gradient is not converging at the right rate");
  }

  std::cout << "Divergence convergence" << std::endl;
  std::vector<Real> div_rate(nlevels - 1);
  for (Int l = 0; l < nlevels; ++l) {
    std::cout << l << " " << div_err[l];
    if (l > 0) {
      div_rate[l - 1] = std::log2(div_err[l - 1] / div_err[l]);
      std::cout << " " << div_rate[l - 1];
    }
    std::cout << std::endl;
  }
  if (!check_rate(div_rate.back(), 2, atol)) {
    throw std::runtime_error("Divergence is not converging at the right rate");
  }

  std::cout << "Curl convergence" << std::endl;
  std::vector<Real> curl_rate(nlevels - 1);
  for (Int l = 0; l < nlevels; ++l) {
    std::cout << l << " " << curl_err[l];
    if (l > 0) {
      curl_rate[l - 1] = std::log2(curl_err[l - 1] / curl_err[l]);
      std::cout << " " << curl_rate[l - 1];
    }
    std::cout << std::endl;
  }
  if (!check_rate(curl_rate.back(), 1, atol)) {
    throw std::runtime_error("Curl is not converging at the right rate");
  }

  std::cout << "Reconstruction convergence" << std::endl;
  std::vector<Real> recon_rate(nlevels - 1);
  for (Int l = 0; l < nlevels; ++l) {
    std::cout << l << " " << recon_err[l];
    if (l > 0) {
      recon_rate[l - 1] = std::log2(recon_err[l - 1] / recon_err[l]);
      std::cout << " " << recon_rate[l - 1];
    }
    std::cout << std::endl;
  }
  if (!check_rate(recon_rate.back(), 2, atol)) {
    throw std::runtime_error(
        "Reconstruction is not converging at the right rate");
  }
}

int main() {
  Kokkos::initialize();
  Int nlevels = 5;
  run(nlevels);
  Kokkos::finalize();
}
