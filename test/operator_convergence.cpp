#include <omega.hpp>
#include <iostream>
#include <vector>

using namespace omega;

Real error_gradient(const PlanarHexagonalMesh &mesh) {
  Real Lx = mesh.period_x;
  Real Ly = mesh.period_y;

  Real1d cell_field("cell_field", mesh.ncells);
  Real1d edge_field("edge_field", mesh.nedges);
  Real1d exact_edge_field("exact_edge_field", mesh.nedges);
  
  parallel_for(mesh.ncells, YAKL_LAMBDA (Int icell) {
      Real x = mesh.x_cell(icell);
      Real y = mesh.y_cell(icell);
      cell_field(icell) = sin(2 * pi * x / Lx) * sin(2 * pi * y / Ly);
  });

  parallel_for(mesh.nedges, YAKL_LAMBDA (Int iedge) {
      Int icell0 = mesh.cells_on_edge(iedge, 0);
      Int icell1 = mesh.cells_on_edge(iedge, 1);

      edge_field(iedge) = (cell_field(icell1) - cell_field(icell0)) / mesh.dc_edge(iedge);

      Real nx = cos(mesh.angle_edge(iedge));
      Real ny = sin(mesh.angle_edge(iedge));

      Real x = mesh.x_edge(iedge);
      Real y = mesh.y_edge(iedge);

      Real grad_x = 2 * pi / Lx * cos(2 * pi * x / Lx) * sin(2 * pi * y / Ly);
      Real grad_y = 2 * pi / Ly * sin(2 * pi * x / Lx) * cos(2 * pi * y / Ly); 

      exact_edge_field(iedge) = nx * grad_x + ny * grad_y;
      
      exact_edge_field(iedge) -= edge_field(iedge);
      exact_edge_field(iedge) = abs(exact_edge_field(iedge));
  });
  return yakl::intrinsics::maxval(exact_edge_field);
}

Real error_divergence(const PlanarHexagonalMesh &mesh) {
  Real Lx = mesh.period_x;
  Real Ly = mesh.period_y;

  Real1d cell_field("cell_field", mesh.ncells);
  Real1d exact_cell_field("exact_cell_field", mesh.ncells);
  Real1d edge_field("edge_field", mesh.nedges);
  
  parallel_for(mesh.nedges, YAKL_LAMBDA (Int iedge) {
      Real nx = cos(mesh.angle_edge(iedge));
      Real ny = sin(mesh.angle_edge(iedge));

      Real x = mesh.x_edge(iedge);
      Real y = mesh.y_edge(iedge);

      Real v_x = sin(2 * pi * x / Lx) * cos(2 * pi * y / Ly);
      Real v_y = cos(2 * pi * x / Lx) * sin(2 * pi * y / Ly); 

      edge_field(iedge) = nx * v_x + ny * v_y;
  });

  parallel_for(mesh.ncells, YAKL_LAMBDA (Int icell) {
      Real accum = -0;
      for (Int j = 0; j < mesh.nedges_on_cell(icell); ++j) {
        Int iedge = mesh.edges_on_cell(icell, j);
        accum += mesh.dv_edge(iedge) * mesh.orient_on_cell(icell, j) * edge_field(iedge);
      }
      cell_field(icell) = accum / mesh.area_cell(icell);

      Real x = mesh.x_cell(icell);
      Real y = mesh.y_cell(icell);
      exact_cell_field(icell) = 2 * pi * (1. / Lx + 1. / Ly) * cos(2 * pi * x / Lx) * cos(2 * pi * y / Ly);

      exact_cell_field(icell) -= cell_field(icell);
      exact_cell_field(icell) = abs(exact_cell_field(icell));
  });

  return yakl::intrinsics::maxval(exact_cell_field);
}

Real error_curl(const PlanarHexagonalMesh &mesh) {
  Real Lx = mesh.period_x;
  Real Ly = mesh.period_y;

  // input
  Real1d edge_field("edge_field", mesh.nedges);
  
  // output
  Real1d vertex_field("vertex_field", mesh.nvertices);
  Real1d exact_vertex_field("exact_vertex_field", mesh.nvertices);
  
  parallel_for(mesh.nedges, YAKL_LAMBDA (Int iedge) {
      Real nx = cos(mesh.angle_edge(iedge));
      Real ny = sin(mesh.angle_edge(iedge));

      Real x = mesh.x_edge(iedge);
      Real y = mesh.y_edge(iedge);

      Real v_x = sin(2 * pi * x / Lx) * cos(2 * pi * y / Ly);
      Real v_y = cos(2 * pi * x / Lx) * sin(2 * pi * y / Ly); 

      edge_field(iedge) = nx * v_x + ny * v_y;
  });

  parallel_for(mesh.nvertices, YAKL_LAMBDA (Int ivertex) {
      Real accum = -0;
      for (Int j = 0; j < 3; ++j) {
        Int iedge = mesh.edges_on_vertex(ivertex, j);
        accum += mesh.dc_edge(iedge) * mesh.orient_on_vertex(ivertex, j) * edge_field(iedge);
      }
      vertex_field(ivertex) = accum / mesh.area_triangle(ivertex);

      Real x = mesh.x_vertex(ivertex);
      Real y = mesh.y_vertex(ivertex);
      exact_vertex_field(ivertex) = 2 * pi * (-1. / Lx + 1. / Ly) * sin(2 * pi * x / Lx) * sin(2 * pi * y / Ly);

      exact_vertex_field(ivertex) -= vertex_field(ivertex);
      exact_vertex_field(ivertex) = abs(exact_vertex_field(ivertex));
  });

  return yakl::intrinsics::maxval(exact_vertex_field);
}

Real error_reconstruction(const PlanarHexagonalMesh &mesh) {
  Real Lx = mesh.period_x;
  Real Ly = mesh.period_y;

  Real1d edge_field_in("edge_field_in", mesh.nedges);
  Real1d edge_field_out("edge_field_out", mesh.nedges);
  Real1d exact_edge_field("exact_edge_field", mesh.nedges);
  
  parallel_for(mesh.nedges, YAKL_LAMBDA (Int iedge) {
      Real nx = cos(mesh.angle_edge(iedge));
      Real ny = sin(mesh.angle_edge(iedge));

      Real tx = -ny;
      Real ty = nx;

      Real x = mesh.x_edge(iedge);
      Real y = mesh.y_edge(iedge);

      Real v_x = sin(2 * pi * x / Lx) * cos(2 * pi * y / Ly);
      Real v_y = cos(2 * pi * x / Lx) * sin(2 * pi * y / Ly); 

      edge_field_in(iedge) = nx * v_x + ny * v_y;
      exact_edge_field(iedge) = tx * v_x + ty * v_y;
  });

  parallel_for(mesh.nedges, YAKL_LAMBDA (Int iedge) {
      Int n = mesh.nedges_on_edge(iedge);
      Real accum = -0;
      for (Int j = 0; j < n; ++j) {
        Int iedge2 = mesh.edges_on_edge(iedge, j);
        accum += mesh.weights_on_edge(iedge, j) * mesh.dv_edge(iedge2) * edge_field_in(iedge2);
      }
      edge_field_out(iedge) = accum / mesh.dc_edge(iedge);

      exact_edge_field(iedge) -= edge_field_out(iedge);
      exact_edge_field(iedge) = abs(exact_edge_field(iedge));
  });

  return yakl::intrinsics::maxval(exact_edge_field);
}

void run(Int nlevels) {
  Int n = 6;
  
  std::vector<Real> grad_err(nlevels);
  std::vector<Real> div_err(nlevels);
  std::vector<Real> curl_err(nlevels);
  std::vector<Real> recon_err(nlevels);
  
  for (Int l = 0; l < nlevels; ++l) {
    PlanarHexagonalMesh mesh(n, n);

    grad_err[l] = error_gradient(mesh);
    div_err[l] = error_divergence(mesh);
    curl_err[l] = error_curl(mesh);
    recon_err[l] = error_reconstruction(mesh);

    n *= 2;
  }

  std::cout << "Gradient convergence" << std::endl;
  for (Int l = 0; l < nlevels; ++l) {
    std::cout << l << " " << grad_err[l];
    if (l > 0) {
      std::cout << " " << std::log2(grad_err[l-1] / grad_err[l]);
    }
    std::cout << std::endl;
  }
  
  std::cout << "Divergence convergence" << std::endl;
  for (Int l = 0; l < nlevels; ++l) {
    std::cout << l << " " << div_err[l];
    if (l > 0) {
      std::cout << " " << std::log2(div_err[l-1] / div_err[l]);
    }
    std::cout << std::endl;
  }
  
  std::cout << "Curl convergence" << std::endl;
  for (Int l = 0; l < nlevels; ++l) {
    std::cout << l << " " << curl_err[l];
    if (l > 0) {
      std::cout << " " << std::log2(curl_err[l-1] / curl_err[l]);
    }
    std::cout << std::endl;
  }

  std::cout << "Reconstruction convergence" << std::endl;
  for (Int l = 0; l < nlevels; ++l) {
    std::cout << l << " " << recon_err[l];
    if (l > 0) {
      std::cout << " " << std::log2(recon_err[l-1] / recon_err[l]);
    }
    std::cout << std::endl;
  }
}

int main() {
  yakl::init();
  run(5);
  yakl::finalize();
}
