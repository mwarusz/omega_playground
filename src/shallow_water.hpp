#pragma once

#include <common.hpp>
#include <planar_hexagonal_mesh.hpp>

namespace omega {

enum class AddMode { replace, increment };

struct ShallowWaterParams {
  Real f0;
  Real grav = 9.81;
  Real drag_coeff = 0;
};

struct LinearShallowWaterParams : ShallowWaterParams {
  Real h0;
};

struct ShallowWaterBase {
  PlanarHexagonalMesh *mesh;
  Real grav;
  Real1d f_vertex;
  Real1d f_edge;

  virtual void compute_auxiliary_variables(RealConst2d h_cell,
                                           RealConst2d vn_edge) const {}

  virtual void compute_h_tendency(Real2d vn_tend_edge, RealConst2d h_cell,
                                  RealConst2d vn_edge,
                                  AddMode add_mode) const = 0;
  virtual void compute_vn_tendency(Real2d h_tend_cell, RealConst2d h_cell,
                                   RealConst2d vn_edge,
                                   AddMode add_mode) const = 0;
  virtual void additional_tendency(Real2d h_tend_cell, Real2d vn_tend_edge,
                                   RealConst2d h_cell, RealConst2d vn_edge,
                                   Real t) const {}
  virtual Real mass_integral(RealConst2d h_cell) const;
  virtual Real circulation_integral(RealConst2d vn_edge) const;
  virtual Real energy_integral(RealConst2d h_cell,
                               RealConst2d vn_edge) const = 0;

  void compute_tendency(Real2d h_tend_cell, Real2d vn_tend_edge,
                        RealConst2d h_cell, RealConst2d vn_edge, Real t,
                        AddMode add_mode = AddMode::replace) const {
    yakl::timer_start("compute_tendency");

    yakl::timer_start("compute_auxiliary_variables");
    compute_auxiliary_variables(h_cell, vn_edge);
    yakl::timer_stop("compute_auxiliary_variables");

    yakl::timer_start("h_tendency");
    compute_h_tendency(h_tend_cell, h_cell, vn_edge, add_mode);
    yakl::timer_stop("h_tendency");

    yakl::timer_start("vn_tendency");
    compute_vn_tendency(vn_tend_edge, h_cell, vn_edge, add_mode);
    yakl::timer_stop("vn_tendency");

    additional_tendency(h_tend_cell, vn_tend_edge, h_cell, vn_edge, t);

    yakl::timer_stop("compute_tendency");
  }

  ShallowWaterBase(PlanarHexagonalMesh &mesh, const ShallowWaterParams &params);
};

struct ShallowWater : ShallowWaterBase {
  Real drag_coeff;
  Real2d h_flux_edge;
  Real2d h_mean_edge;
  Real2d h_drag_edge;

  Real2d rcirc_vertex;
  Real2d rvort_vertex;

  Real2d rvort_cell;
  Real2d ke_cell;
  Real2d vt_edge;

  Real2d norm_rvort_vertex;
  Real2d norm_f_vertex;
  Real2d norm_rvort_edge;
  Real2d norm_f_edge;
  Real2d norm_rvort_cell;

  void compute_auxiliary_variables(RealConst2d h_cell,
                                   RealConst2d vn_edge) const override;

  void compute_h_tendency(Real2d vn_tend_edge, RealConst2d h_cell,
                          RealConst2d vn_edge, AddMode add_mode) const override;
  void compute_vn_tendency(Real2d h_tend_cell, RealConst2d h_cell,
                           RealConst2d vn_edge,
                           AddMode add_mode) const override;
  Real energy_integral(RealConst2d h, RealConst2d v) const override;

  ShallowWater(PlanarHexagonalMesh &mesh, const ShallowWaterParams &params);
};

struct LinearShallowWater : ShallowWaterBase {
  Real h0;
  void compute_h_tendency(Real2d vn_tend_edge, RealConst2d h_cell,
                          RealConst2d vn_edge, AddMode add_mode) const override;
  void compute_vn_tendency(Real2d h_tend_cell, RealConst2d h_cell,
                           RealConst2d vn_edge,
                           AddMode add_mode) const override;
  Real energy_integral(RealConst2d h_cell, RealConst2d vn_edge) const override;

  LinearShallowWater(PlanarHexagonalMesh &mesh,
                     const LinearShallowWaterParams &params);
};
} // namespace omega
