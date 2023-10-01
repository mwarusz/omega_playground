#include <iostream>
#include <omega.hpp>
#include <vector>

using namespace omega;

constexpr Real shrink_factor = 120;
constexpr Real earth_radius = 6.37122e6 / shrink_factor;
constexpr Real day = 24 * 60 * 60 / shrink_factor;

bool check_rate(Real rate, Real expected_rate, Real atol) {
  return std::abs(rate - expected_rate) < atol && !std::isnan(rate);
}

struct CosineBell {
  Real m_u0 = 2 * pi * earth_radius / (12 * day);
  Real m_alpha = pi / 4;
  Real m_lon_c = 3 * pi / 2;
  Real m_lat_c = 0;
  Real m_R = earth_radius / 3;
  Real m_h0 = 1000;

  YAKL_INLINE Real h(Real lon, Real lat) const { return 1; }

  YAKL_INLINE Real tr(Real lon, Real lat) const {
    using std::acos;
    using std::cos;
    using std::sin;
    Real r = earth_radius * acos(sin(m_lat_c) * sin(lat) +
                                 cos(m_lat_c) * cos(lat) * cos(lon - m_lon_c));
    return r < m_R ? (m_h0 / 2) * (1 + cos(pi * r / m_R)) : 0;
  }

  YAKL_INLINE Real psi(Real lon, Real lat) const {
    using std::cos;
    using std::sin;
    return -earth_radius * m_u0 *
           (sin(lat) * cos(m_alpha) - cos(lon) * cos(lat) * sin(m_alpha));
  }
};

Real run(Int l) {
  CosineBell cosine_bell;

  std::string mesh_file;
  if (l == 0) {
    mesh_file = "../meshes/mesh_cvt_5.nc";
  } else {
    mesh_file = "../meshes/mesh_cvt_6.nc";
  }

  auto mesh = std::make_unique<FileMesh>(mesh_file);
  mesh->rescale_radius(earth_radius);

  ShallowWaterParams params;
  params.m_disable_h_tendency = true;
  params.m_disable_vn_tendency = true;
  params.m_ntracers = 1;

  ShallowWaterModel shallow_water(mesh.get(), params);

  ShallowWaterState state(shallow_water);

  LSRKStepper stepper(shallow_water);

  Real timeend = 12 * day;
  Real cfl = 0.6;
  Real min_dc_edge = yakl::intrinsics::minval(mesh->m_dc_edge);
  Real dt = cfl * min_dc_edge / cosine_bell.m_u0;
  Int numberofsteps = std::ceil(timeend / dt);
  dt = timeend / numberofsteps;

  auto &h_cell = state.m_h_cell;
  auto &tr_cell = state.m_tr_cell;
  Real3d tr_exact_cell("tr_exact_cell", params.m_ntracers, mesh->m_ncells,
                       mesh->m_nlayers);
  YAKL_SCOPE(lon_cell, mesh->m_lon_cell);
  YAKL_SCOPE(lat_cell, mesh->m_lat_cell);
  parallel_for(
      "init_h_and_tr", SimpleBounds<2>(mesh->m_ncells, mesh->m_nlayers),
      YAKL_LAMBDA(Int icell, Int k) {
        Real lon = lon_cell(icell);
        Real lat = lat_cell(icell);
        h_cell(icell, k) = cosine_bell.h(lon, lat);
        tr_cell(0, icell, k) = cosine_bell.tr(lon, lat);
        tr_exact_cell(0, icell, k) = cosine_bell.tr(lon, lat);
      });

  YAKL_SCOPE(lon_vertex, mesh->m_lon_vertex);
  YAKL_SCOPE(lat_vertex, mesh->m_lat_vertex);
  Real1d psi_vertex("psi_vertex", mesh->m_nvertices);
  parallel_for(
      "init_psi", mesh->m_nvertices, YAKL_LAMBDA(Int ivertex) {
        Real lon = lon_vertex(ivertex);
        Real lat = lat_vertex(ivertex);
        psi_vertex(ivertex) = cosine_bell.psi(lon, lat);
      });

  YAKL_SCOPE(vertices_on_edge, mesh->m_vertices_on_edge);
  YAKL_SCOPE(dv_edge, mesh->m_dv_edge);
  auto &vn_edge = state.m_vn_edge;
  parallel_for(
      "init_vn", SimpleBounds<2>(mesh->m_nedges, mesh->m_nlayers),
      YAKL_LAMBDA(Int iedge, Int k) {
        Int jvertex0 = vertices_on_edge(iedge, 0);
        Int jvertex1 = vertices_on_edge(iedge, 1);
        vn_edge(iedge, k) =
            -(psi_vertex(jvertex1) - psi_vertex(jvertex0)) / dv_edge(iedge);
      });

  for (Int step = 0; step < numberofsteps; ++step) {
    Real t = step * dt;
    stepper.do_step(t, dt, state);
  }
  std::cout << "cosine bell extrema: " << yakl::intrinsics::minval(tr_cell)
            << " " << yakl::intrinsics::maxval(tr_cell) << std::endl;

  YAKL_SCOPE(area_cell, mesh->m_area_cell);
  parallel_for(
      "compute_error", SimpleBounds<2>(mesh->m_ncells, mesh->m_nlayers),
      YAKL_LAMBDA(Int icell, Int k) {
        tr_cell(0, icell, k) -= tr_exact_cell(0, icell, k);
        tr_cell(0, icell, k) *= area_cell(icell) * tr_cell(0, icell, k);
        tr_exact_cell(0, icell, k) *=
            area_cell(icell) * tr_exact_cell(0, icell, k);
      });

  return std::sqrt(yakl::intrinsics::sum(tr_cell) /
                   yakl::intrinsics::sum(tr_exact_cell));
}

int main() {
  yakl::init();

  Int nlevels = 2;

  std::vector<Real> err(nlevels);
  for (Int l = 0; l < nlevels; ++l) {
    err[l] = run(l);
  }

  if (nlevels > 1) {
    std::vector<Real> rate(nlevels - 1);
    std::cout << "Cosine bell convergence" << std::endl;
    for (Int l = 0; l < nlevels; ++l) {
      std::cout << l << " " << err[l];
      if (l > 0) {
        rate[l - 1] = std::log2(err[l - 1] / err[l]);
        std::cout << " " << rate[l - 1];
      }
      std::cout << std::endl;
    }

    // if (!check_rate(rate.back(), 2, 0.05)) {
    //   throw std::runtime_error(
    //       "Tracer advection is not converging at the right rate");
    // }
  }

  yakl::finalize();
}
