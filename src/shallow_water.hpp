#pragma once

#include <common.hpp>
#include <planar_hexagonal_mesh.hpp>

namespace omega {

enum class AddMode { replace, increment };

struct ShallowWaterState {
  Real2d h_cell;
  Real2d vn_edge;
  Real3d tr_cell;
  Int ntracers;

  ShallowWaterState(const PlanarHexagonalMesh &mesh, Int ntracers = 0);
};

struct ShallowWaterParams {
  Real f0;
  Real grav = 9.81;
  Real drag_coeff = 0;
  Real visc_del2 = 0;
  bool disable_h_tendency = false;
  bool disable_vn_tendency = false;
};

struct LinearShallowWaterParams : ShallowWaterParams {
  Real h0;
};

struct ShallowWaterModelBase {
  PlanarHexagonalMesh *mesh;
  Real grav;
  bool disable_h_tendency;
  bool disable_vn_tendency;
  Real1d f_vertex;
  Real1d f_edge;

  virtual void compute_auxiliary_variables(RealConst2d h_cell,
                                           RealConst2d vn_edge,
                                           RealConst3d tr_cell) const {}

  virtual void compute_h_tendency(Real2d h_tend_cell, RealConst2d h_cell,
                                  RealConst2d vn_edge,
                                  AddMode add_mode) const = 0;
  virtual void compute_vn_tendency(Real2d vn_tend_edge, RealConst2d h_cell,
                                   RealConst2d vn_edge,
                                   AddMode add_mode) const = 0;
  virtual void compute_tr_tendency(Real3d tr_tend_cell, RealConst3d tr_cell,
                                   RealConst2d vn_edge,
                                   AddMode add_mode) const = 0;
  virtual void additional_tendency(Real2d h_tend_cell, Real2d vn_tend_edge,
                                   RealConst2d h_cell, RealConst2d vn_edge,
                                   Real t) const {}
  virtual Real mass_integral(RealConst2d h_cell) const;
  virtual Real circulation_integral(RealConst2d vn_edge) const;
  virtual Real energy_integral(RealConst2d h_cell,
                               RealConst2d vn_edge) const = 0;

  void compute_tendency(const ShallowWaterState &tend,
                        const ShallowWaterState &state, Real t,
                        AddMode add_mode = AddMode::replace) const {
    yakl::timer_start("compute_tendency");

    yakl::timer_start("compute_auxiliary_variables");
    compute_auxiliary_variables(state.h_cell, state.vn_edge, state.tr_cell);
    yakl::timer_stop("compute_auxiliary_variables");

    yakl::timer_start("h_tendency");
    if (!disable_h_tendency) {
      compute_h_tendency(tend.h_cell, state.h_cell, state.vn_edge, add_mode);
    }
    yakl::timer_stop("h_tendency");

    yakl::timer_start("vn_tendency");
    if (!disable_vn_tendency) {
      compute_vn_tendency(tend.vn_edge, state.h_cell, state.vn_edge, add_mode);
    }
    yakl::timer_stop("vn_tendency");

    yakl::timer_start("tr_tendency");
    compute_tr_tendency(tend.tr_cell, state.tr_cell, state.vn_edge, add_mode);
    yakl::timer_stop("tr_tendency");

    yakl::timer_start("additional_tendency");
    additional_tendency(tend.h_cell, tend.vn_edge, state.h_cell, state.vn_edge,
                        t);
    yakl::timer_stop("additional_tendency");

    yakl::timer_stop("compute_tendency");
  }

  ShallowWaterModelBase(PlanarHexagonalMesh &mesh,
                        const ShallowWaterState &state,
                        const ShallowWaterParams &params);
};

struct ShallowWaterModel : ShallowWaterModelBase {
  Real drag_coeff;
  Real visc_del2;

  Real2d rvort_cell;
  Real2d ke_cell;
  Real2d div_cell;
  Real2d norm_rvort_cell;
  Real3d norm_tr_cell;

  Real2d h_flux_edge;
  Real2d h_mean_edge;
  Real2d h_drag_edge;
  Real2d vt_edge;
  Real2d norm_rvort_edge;
  Real2d norm_f_edge;

  Real2d rcirc_vertex;
  Real2d rvort_vertex;
  Real2d norm_rvort_vertex;
  Real2d norm_f_vertex;

  void compute_auxiliary_variables(RealConst2d h_cell, RealConst2d vn_edge,
                                   RealConst3d tr_cell) const override;

  void compute_h_tendency(Real2d h_tend_cell, RealConst2d h_cell,
                          RealConst2d vn_edge, AddMode add_mode) const override;
  void compute_vn_tendency(Real2d vn_tend_edge, RealConst2d h_cell,
                           RealConst2d vn_edge,
                           AddMode add_mode) const override;
  void compute_tr_tendency(Real3d tr_tend_cell, RealConst3d tr_cell,
                           RealConst2d vn_edge,
                           AddMode add_mode) const override;
  Real energy_integral(RealConst2d h, RealConst2d v) const override;

  ShallowWaterModel(PlanarHexagonalMesh &mesh, const ShallowWaterState &state,
                    const ShallowWaterParams &params);
};

struct LinearShallowWaterModel : ShallowWaterModelBase {
  Real h0;
  void compute_h_tendency(Real2d h_tend_cell, RealConst2d h_cell,
                          RealConst2d vn_edge, AddMode add_mode) const override;
  void compute_vn_tendency(Real2d vn_tend_edge, RealConst2d h_cell,
                           RealConst2d vn_edge,
                           AddMode add_mode) const override;
  void compute_tr_tendency(Real3d tr_tend_cell, RealConst3d tr_cell,
                           RealConst2d vn_edge,
                           AddMode add_mode) const override {}
  Real energy_integral(RealConst2d h_cell, RealConst2d vn_edge) const override;

  LinearShallowWaterModel(PlanarHexagonalMesh &mesh,
                          const ShallowWaterState &state,
                          const LinearShallowWaterParams &params);
};
} // namespace omega
