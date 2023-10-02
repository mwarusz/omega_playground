#include <iostream>
#include <omega.hpp>
#include <vector>

using namespace omega;

constexpr Real shrink_factor = 1;
constexpr Real grav = 9.8016;
constexpr Real earth_radius = 6.37122e6 / shrink_factor;
constexpr Real day = 24 * 60 * 60 / shrink_factor;
constexpr Real omg = 7.292e-5 * shrink_factor;

struct SteadyZonal {
  Real m_u0 = 2 * pi * earth_radius / (12 * day);
  Real m_alpha = pi / 4;
  Real m_h0 = 2.94e4 / grav;
  
  YAKL_INLINE Real coriolis(Real lon, Real lat) const {
    using std::cos;
    using std::sin;
    return 2 * omg * (-cos(lon) * cos(lat) * sin(m_alpha) + sin(lat) * cos(m_alpha));
  }

  YAKL_INLINE Real h(Real lon, Real lat) const {
    using std::cos;
    using std::sin;
    Real tmp = -cos(lat) * cos(lon) * sin(m_alpha) + sin(lat) * cos(m_alpha);
    return m_h0 - (earth_radius * omg * m_u0 + m_u0 * m_u0 / 2) * tmp * tmp / grav;
  }

  YAKL_INLINE Real psi(Real lon, Real lat) const {
    using std::cos;
    using std::sin;
    return -earth_radius * m_u0 *
           (sin(lat) * cos(m_alpha) - cos(lon) * cos(lat) * sin(m_alpha));
  }
};

Real run(Int l) {
  SteadyZonal steady_zonal;

  std::string mesh_file;
  if (l == 0) {
    mesh_file = "../meshes/mesh_cvt_5.nc";
  } else {
    mesh_file = "../meshes/mesh_cvt_6.nc";
  }

  auto mesh = std::make_unique<FileMesh>(mesh_file);
  mesh->rescale_radius(earth_radius);

  ShallowWaterParams params;
  params.m_grav = grav;
  ShallowWaterModel shallow_water(mesh.get(), params);

  ShallowWaterState state(shallow_water);

  LSRKStepper stepper(shallow_water);

  Real timeend = day;
  Real cfl = 0.6;
  Real min_dc_edge = yakl::intrinsics::minval(mesh->m_dc_edge);
  Real dt = cfl * min_dc_edge / (steady_zonal.m_u0 + std::sqrt(grav * steady_zonal.m_h0));
  Int numberofsteps = std::ceil(timeend / dt);
  dt = timeend / numberofsteps;

  auto &h_cell = state.m_h_cell;
  Real2d h_exact_cell("h_exact_cell", mesh->m_ncells, mesh->m_nlayers);
  YAKL_SCOPE(lon_cell, mesh->m_lon_cell);
  YAKL_SCOPE(lat_cell, mesh->m_lat_cell);
  parallel_for(
      "init_h", SimpleBounds<2>(mesh->m_ncells, mesh->m_nlayers),
      YAKL_LAMBDA(Int icell, Int k) {
        Real lon = lon_cell(icell);
        Real lat = lat_cell(icell);
        h_cell(icell, k) = steady_zonal.h(lon, lat);
        h_exact_cell(icell, k) = steady_zonal.h(lon, lat);
      });

  YAKL_SCOPE(lon_vertex, mesh->m_lon_vertex);
  YAKL_SCOPE(lat_vertex, mesh->m_lat_vertex);
  YAKL_SCOPE(f_vertex, shallow_water.m_f_vertex);
  Real1d psi_vertex("psi_vertex", mesh->m_nvertices);
  parallel_for(
      "init_psi_and_f", mesh->m_nvertices, YAKL_LAMBDA(Int ivertex) {
        Real lon = lon_vertex(ivertex);
        Real lat = lat_vertex(ivertex);
        psi_vertex(ivertex) = steady_zonal.psi(lon, lat);
        f_vertex(ivertex) = steady_zonal.coriolis(lon, lat);
      });

  YAKL_SCOPE(vertices_on_edge, mesh->m_vertices_on_edge);
  YAKL_SCOPE(dv_edge, mesh->m_dv_edge);
  YAKL_SCOPE(lon_edge, mesh->m_lon_edge);
  YAKL_SCOPE(lat_edge, mesh->m_lat_edge);
  YAKL_SCOPE(f_edge, shallow_water.m_f_edge);
  auto &vn_edge = state.m_vn_edge;
  parallel_for(
      "init_vn_and_f", SimpleBounds<2>(mesh->m_nedges, mesh->m_nlayers),
      YAKL_LAMBDA(Int iedge, Int k) {
        Int jvertex0 = vertices_on_edge(iedge, 0);
        Int jvertex1 = vertices_on_edge(iedge, 1);
        vn_edge(iedge, k) =
            -(psi_vertex(jvertex1) - psi_vertex(jvertex0)) / dv_edge(iedge);
        Real lon = lon_edge(iedge);
        Real lat = lat_edge(iedge);
        f_edge(iedge) = steady_zonal.coriolis(lon, lat);
      });

  for (Int step = 0; step < numberofsteps; ++step) {
    Real t = step * dt;
    stepper.do_step(t, dt, state);
  }
  std::cout << "h extrema: " << yakl::intrinsics::minval(h_cell)
            << " " << yakl::intrinsics::maxval(h_cell) << std::endl;

  YAKL_SCOPE(area_cell, mesh->m_area_cell);
  parallel_for(
      "compute_error", SimpleBounds<2>(mesh->m_ncells, mesh->m_nlayers),
      YAKL_LAMBDA(Int icell, Int k) {
        h_cell(icell, k) -= h_exact_cell(icell, k);
        h_cell(icell, k) *= area_cell(icell) * h_cell(icell, k);
        h_exact_cell(icell, k) *=
            area_cell(icell) * h_exact_cell(icell, k);
      });

  return std::sqrt(yakl::intrinsics::sum(h_cell) /
                   yakl::intrinsics::sum(h_exact_cell));
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
  }

  yakl::finalize();
}
