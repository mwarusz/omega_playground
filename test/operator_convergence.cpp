#include <iostream>
#include <omega.hpp>
#include <vector>

using namespace omega;

struct Error {
  Real linf;
  Real l2;
};

bool check_rate(Real rate, Real expected_rate, Real atol) {
  return std::abs(rate - expected_rate) < atol && !std::isnan(rate);
}

Error error_gradient(const PlanarHexagonalMesh &mesh) {
  using std::cos;
  using std::sin;

  Real Lx = mesh.m_period_x;
  Real Ly = mesh.m_period_y;

  Real1d input_field("input_field", mesh.m_ncells);

  Real1d linf_error("linf_error", mesh.m_nedges);
  Real1d l2_error("l2_error", mesh.m_nedges);

  parallel_for(
      mesh.m_ncells, YAKL_LAMBDA(Int icell) {
        Real x = mesh.m_x_cell(icell);
        Real y = mesh.m_y_cell(icell);
        input_field(icell) = sin(2 * pi * x / Lx) * sin(2 * pi * y / Ly);
      });

  parallel_for(
      mesh.m_nedges, YAKL_LAMBDA(Int iedge) {
        Int icell0 = mesh.m_cells_on_edge(iedge, 0);
        Int icell1 = mesh.m_cells_on_edge(iedge, 1);

        Real grad_edge =
            (input_field(icell1) - input_field(icell0)) / mesh.m_dc_edge(iedge);

        Real nx = cos(mesh.m_angle_edge(iedge));
        Real ny = sin(mesh.m_angle_edge(iedge));

        Real x = mesh.m_x_edge(iedge);
        Real y = mesh.m_y_edge(iedge);

        Real grad_x = 2 * pi / Lx * cos(2 * pi * x / Lx) * sin(2 * pi * y / Ly);
        Real grad_y = 2 * pi / Ly * sin(2 * pi * x / Lx) * cos(2 * pi * y / Ly);

        Real exact_grad_edge = nx * grad_x + ny * grad_y;

        linf_error(iedge) = std::abs(grad_edge - exact_grad_edge);
        Real area_edge = mesh.m_dc_edge(iedge) * mesh.m_dv_edge(iedge) / 2;
        l2_error(iedge) = area_edge * linf_error(iedge) * linf_error(iedge);
      });
  return {yakl::intrinsics::maxval(linf_error),
          std::sqrt(yakl::intrinsics::sum(l2_error))};
}

Error error_divergence(const PlanarHexagonalMesh &mesh) {
  using std::cos;
  using std::sin;

  Real Lx = mesh.m_period_x;
  Real Ly = mesh.m_period_y;

  Real1d input_field("input_field", mesh.m_nedges);
  Real1d linf_error("linf_error", mesh.m_ncells);
  Real1d l2_error("l2_error", mesh.m_ncells);

  parallel_for(
      mesh.m_nedges, YAKL_LAMBDA(Int iedge) {
        Real nx = cos(mesh.m_angle_edge(iedge));
        Real ny = sin(mesh.m_angle_edge(iedge));

        Real x = mesh.m_x_edge(iedge);
        Real y = mesh.m_y_edge(iedge);

        Real v_x = sin(2 * pi * x / Lx) * cos(2 * pi * y / Ly);
        Real v_y = cos(2 * pi * x / Lx) * sin(2 * pi * y / Ly);

        input_field(iedge) = nx * v_x + ny * v_y;
      });

  parallel_for(
      mesh.m_ncells, YAKL_LAMBDA(Int icell) {
        Real accum = -0;
        for (Int j = 0; j < mesh.m_nedges_on_cell(icell); ++j) {
          Int jedge = mesh.m_edges_on_cell(icell, j);
          accum += mesh.m_dv_edge(jedge) * mesh.m_edge_sign_on_cell(icell, j) *
                   input_field(jedge);
        }
        Real div_num = accum / mesh.m_area_cell(icell);

        Real x = mesh.m_x_cell(icell);
        Real y = mesh.m_y_cell(icell);
        Real div_exact = 2 * pi * (1. / Lx + 1. / Ly) * cos(2 * pi * x / Lx) *
                         cos(2 * pi * y / Ly);

        linf_error(icell) = std::abs(div_num - div_exact);
        l2_error(icell) =
            mesh.m_area_cell(icell) * linf_error(icell) * linf_error(icell);
      });

  return {yakl::intrinsics::maxval(linf_error),
          std::sqrt(yakl::intrinsics::sum(l2_error))};
}

Error error_curl(const PlanarHexagonalMesh &mesh) {
  using std::cos;
  using std::sin;

  Real Lx = mesh.m_period_x;
  Real Ly = mesh.m_period_y;

  Real1d input_field("input_field", mesh.m_nedges);

  Real1d linf_error("linf_error", mesh.m_nvertices);
  Real1d l2_error("l2_error", mesh.m_nvertices);

  parallel_for(
      mesh.m_nedges, YAKL_LAMBDA(Int iedge) {
        Real nx = cos(mesh.m_angle_edge(iedge));
        Real ny = sin(mesh.m_angle_edge(iedge));

        Real x = mesh.m_x_edge(iedge);
        Real y = mesh.m_y_edge(iedge);

        Real v_x = sin(2 * pi * x / Lx) * cos(2 * pi * y / Ly);
        Real v_y = cos(2 * pi * x / Lx) * sin(2 * pi * y / Ly);

        input_field(iedge) = nx * v_x + ny * v_y;
      });

  parallel_for(
      mesh.m_nvertices, YAKL_LAMBDA(Int ivertex) {
        Real accum = -0;
        for (Int j = 0; j < 3; ++j) {
          Int jedge = mesh.m_edges_on_vertex(ivertex, j);
          accum += mesh.m_dc_edge(jedge) *
                   mesh.m_edge_sign_on_vertex(ivertex, j) * input_field(jedge);
        }
        Real curl_num = accum / mesh.m_area_triangle(ivertex);

        Real x = mesh.m_x_vertex(ivertex);
        Real y = mesh.m_y_vertex(ivertex);
        Real curl_exact = 2 * pi * (-1. / Lx + 1. / Ly) * sin(2 * pi * x / Lx) *
                          sin(2 * pi * y / Ly);

        linf_error(ivertex) = std::abs(curl_num - curl_exact);
        l2_error(ivertex) = mesh.m_area_triangle(ivertex) *
                            linf_error(ivertex) * linf_error(ivertex);
      });

  return {yakl::intrinsics::maxval(linf_error),
          std::sqrt(yakl::intrinsics::sum(l2_error))};
}

Error error_reconstruction(const PlanarHexagonalMesh &mesh) {
  using std::cos;
  using std::sin;

  Real Lx = mesh.m_period_x;
  Real Ly = mesh.m_period_y;

  Real1d input_field("input_field", mesh.m_nedges);

  Real1d linf_error("linf_error", mesh.m_nedges);
  Real1d l2_error("l2_error", mesh.m_nedges);

  parallel_for(
      mesh.m_nedges, YAKL_LAMBDA(Int iedge) {
        Real nx = cos(mesh.m_angle_edge(iedge));
        Real ny = sin(mesh.m_angle_edge(iedge));

        Real x = mesh.m_x_edge(iedge);
        Real y = mesh.m_y_edge(iedge);

        Real v_x = sin(2 * pi * x / Lx) * cos(2 * pi * y / Ly);
        Real v_y = cos(2 * pi * x / Lx) * sin(2 * pi * y / Ly);

        input_field(iedge) = nx * v_x + ny * v_y;
      });

  parallel_for(
      mesh.m_nedges, YAKL_LAMBDA(Int iedge) {
        Int n = mesh.m_nedges_on_edge(iedge);
        Real accum = -0;
        for (Int j = 0; j < n; ++j) {
          Int iedge2 = mesh.m_edges_on_edge(iedge, j);
          accum += mesh.m_weights_on_edge(iedge, j) * input_field(iedge2);
        }
        Real recon_num = accum;

        Real nx = cos(mesh.m_angle_edge(iedge));
        Real ny = sin(mesh.m_angle_edge(iedge));

        Real tx = -ny;
        Real ty = nx;

        Real x = mesh.m_x_edge(iedge);
        Real y = mesh.m_y_edge(iedge);

        Real v_x = sin(2 * pi * x / Lx) * cos(2 * pi * y / Ly);
        Real v_y = cos(2 * pi * x / Lx) * sin(2 * pi * y / Ly);

        Real recon_exact = tx * v_x + ty * v_y;

        linf_error(iedge) = std::abs(recon_num - recon_exact);
        Real area_edge = mesh.m_dc_edge(iedge) * mesh.m_dv_edge(iedge) / 2;
        l2_error(iedge) = area_edge * linf_error(iedge) * linf_error(iedge);
      });

  return {yakl::intrinsics::maxval(linf_error),
          std::sqrt(yakl::intrinsics::sum(l2_error))};
}

void run(Int nlevels) {
  Int nx = 6;
  Real atol = 0.05;

  std::vector<Error> grad_err(nlevels);
  std::vector<Error> div_err(nlevels);
  std::vector<Error> curl_err(nlevels);
  std::vector<Error> recon_err(nlevels);

  for (Int l = 0; l < nlevels; ++l) {
    PlanarHexagonalMesh mesh(nx, nx);

    grad_err[l] = error_gradient(mesh);
    div_err[l] = error_divergence(mesh);
    curl_err[l] = error_curl(mesh);
    recon_err[l] = error_reconstruction(mesh);

    nx *= 2;
  }

  std::cout << "Gradient convergence" << std::endl;
  std::vector<Error> grad_rate(nlevels - 1);
  for (Int l = 0; l < nlevels; ++l) {
    std::cout << l << " " << grad_err[l].linf << " " << grad_err[l].l2;
    if (l > 0) {
      grad_rate[l - 1].linf =
          std::log2(grad_err[l - 1].linf / grad_err[l].linf);
      grad_rate[l - 1].l2 = std::log2(grad_err[l - 1].l2 / grad_err[l].l2);
      std::cout << " " << grad_rate[l - 1].linf << " " << grad_rate[l - 1].l2;
    }
    std::cout << std::endl;
  }
  if (!check_rate(grad_rate.back().linf, 2, atol)) {
    throw std::runtime_error("Gradient is not converging at the right rate");
  }
  if (!check_rate(grad_rate.back().l2, 2, atol)) {
    throw std::runtime_error("Gradient is not converging at the right rate");
  }

  std::cout << "Divergence convergence" << std::endl;
  std::vector<Error> div_rate(nlevels - 1);
  for (Int l = 0; l < nlevels; ++l) {
    std::cout << l << " " << div_err[l].linf << " " << div_err[l].l2;
    if (l > 0) {
      div_rate[l - 1].linf = std::log2(div_err[l - 1].linf / div_err[l].linf);
      div_rate[l - 1].l2 = std::log2(div_err[l - 1].l2 / div_err[l].l2);
      std::cout << " " << div_rate[l - 1].linf << " " << div_rate[l - 1].l2;
    }
    std::cout << std::endl;
  }
  if (!check_rate(div_rate.back().linf, 2, atol)) {
    throw std::runtime_error("Divergence is not converging at the right rate");
  }
  if (!check_rate(div_rate.back().l2, 2, atol)) {
    throw std::runtime_error("Divergence is not converging at the right rate");
  }

  std::cout << "Curl convergence" << std::endl;
  std::vector<Error> curl_rate(nlevels - 1);
  for (Int l = 0; l < nlevels; ++l) {
    std::cout << l << " " << curl_err[l].linf << " " << curl_err[l].l2;
    if (l > 0) {
      curl_rate[l - 1].linf =
          std::log2(curl_err[l - 1].linf / curl_err[l].linf);
      curl_rate[l - 1].l2 = std::log2(curl_err[l - 1].l2 / curl_err[l].l2);
      std::cout << " " << curl_rate[l - 1].linf << " " << curl_rate[l - 1].l2;
    }
    std::cout << std::endl;
  }
  if (!check_rate(curl_rate.back().linf, 1, atol)) {
    throw std::runtime_error("Curl is not converging at the right rate");
  }
  if (!check_rate(curl_rate.back().l2, 1, atol)) {
    throw std::runtime_error("Curl is not converging at the right rate");
  }

  std::cout << "Reconstruction convergence" << std::endl;
  std::vector<Error> recon_rate(nlevels - 1);
  for (Int l = 0; l < nlevels; ++l) {
    std::cout << l << " " << recon_err[l].linf << " " << recon_err[l].l2;
    if (l > 0) {
      recon_rate[l - 1].linf =
          std::log2(recon_err[l - 1].linf / recon_err[l].linf);
      recon_rate[l - 1].l2 = std::log2(recon_err[l - 1].l2 / recon_err[l].l2);
      std::cout << " " << recon_rate[l - 1].linf << " " << recon_rate[l - 1].l2;
    }
    std::cout << std::endl;
  }
  if (!check_rate(recon_rate.back().linf, 2, atol)) {
    throw std::runtime_error(
        "Reconstruction is not converging at the right rate");
  }
  if (!check_rate(recon_rate.back().l2, 2, atol)) {
    throw std::runtime_error(
        "Reconstruction is not converging at the right rate");
  }
}

int main() {
  yakl::init();
  Int nlevels = 5;
  run(nlevels);
  yakl::finalize();
}
