#pragma once

#include <common.hpp>
#include <planar_hexagonal_mesh.hpp>

namespace omega {

enum class AddMode { replace, increment };

struct ShallowWaterParams {
  Real m_f0;
  Real m_grav = 9.81;
  Real m_drag_coeff = 0;
  Real m_visc_del2 = 0;
  Real m_visc_del4 = 0;
  Real m_eddy_diff2 = 0;
  Real m_eddy_diff4 = 0;
  Int m_ntracers = 0;
  bool m_disable_h_tendency = false;
  bool m_disable_vn_tendency = false;
};

struct LinearShallowWaterParams : ShallowWaterParams {
  Real m_h0;
};

// fwd
struct ShallowWaterModelBase;

struct ShallowWaterState {
  Real2d m_h_cell;
  Real2d m_vn_edge;
  Real3d m_tr_cell;

  ShallowWaterState(const PlanarHexagonalMesh &mesh, Int ntracers);
  ShallowWaterState(const ShallowWaterModelBase &sw);
};

struct ShallowWaterModelBase {
  PlanarHexagonalMesh *m_mesh;
  Real m_grav;
  bool m_disable_h_tendency;
  bool m_disable_vn_tendency;
  Int m_ntracers;
  Real1d m_f_vertex;
  Real1d m_f_edge;

  virtual void compute_auxiliary_variables(RealConst2d h_cell,
                                           RealConst2d vn_edge,
                                           RealConst3d tr_cell) const;

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
                        AddMode add_mode = AddMode::replace) const;

  ShallowWaterModelBase(PlanarHexagonalMesh &mesh,
                        const ShallowWaterParams &params);
};

struct ShallowWaterModel : ShallowWaterModelBase {
  Real m_drag_coeff;
  Real m_visc_del2;
  Real m_visc_del4;
  Real m_eddy_diff2;
  Real m_eddy_diff4;

  Real2d m_ke_cell;
  Real2d m_div_cell;
  Real3d m_norm_tr_cell;
  // Real2d m_rvort_cell;
  // Real2d m_norm_rvort_cell;

  Real2d m_h_flux_edge;
  Real2d m_h_mean_edge;
  Real2d m_h_drag_edge;
  Real2d m_vt_edge;
  Real2d m_norm_rvort_edge;
  Real2d m_norm_f_edge;

  // Real2d m_rcirc_vertex;
  Real2d m_rvort_vertex;
  Real2d m_norm_rvort_vertex;
  Real2d m_norm_f_vertex;

  void compute_auxiliary_variables(RealConst2d h_cell, RealConst2d vn_edge,
                                   RealConst3d tr_cell) const override;
  void compute_cell_auxiliary_variables(RealConst2d h_cell, RealConst2d vn_edge,
                                        RealConst3d tr_cell) const;
  void compute_edge_auxiliary_variables(RealConst2d h_cell, RealConst2d vn_edge,
                                        RealConst3d tr_cell) const;
  void compute_vertex_auxiliary_variables(RealConst2d h_cell,
                                          RealConst2d vn_edge,
                                          RealConst3d tr_cell) const;

  void compute_h_tendency(Real2d h_tend_cell, RealConst2d h_cell,
                          RealConst2d vn_edge, AddMode add_mode) const override;
  void compute_vn_tendency(Real2d vn_tend_edge, RealConst2d h_cell,
                           RealConst2d vn_edge,
                           AddMode add_mode) const override;
  void compute_tr_tendency(Real3d tr_tend_cell, RealConst3d tr_cell,
                           RealConst2d vn_edge,
                           AddMode add_mode) const override;
  Real energy_integral(RealConst2d h, RealConst2d v) const override;

  ShallowWaterModel(PlanarHexagonalMesh &mesh,
                    const ShallowWaterParams &params);
};

struct LinearShallowWaterModel : ShallowWaterModelBase {
  Real m_h0;
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
                          const LinearShallowWaterParams &params);
};
} // namespace omega
