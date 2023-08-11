#pragma once

#include <common.hpp>
#include <planar_hexagonal_mesh.hpp>

namespace omega {

struct ShallowWaterBase {
  PlanarHexagonalMesh *mesh;
  Real grav = 9.81;
  // TODO: generalize to variable f
  Real f0;

  virtual void compute_h_tendency(Real2d vtend, RealConst2d h,
                                  RealConst2d v) const = 0;
  virtual void compute_v_tendency(Real2d htend, RealConst2d h,
                                  RealConst2d v) const = 0;
  virtual void additional_tendency(Real2d htend, Real2d vtend, RealConst2d h,
                                   RealConst2d v, Real t) const {}
  virtual Real mass_integral(RealConst2d h) const;
  virtual Real circulation_integral(RealConst2d v) const;
  virtual Real energy_integral(RealConst2d h, RealConst2d v) const = 0;

  void compute_tendency(Real2d htend, Real2d vtend, RealConst2d h,
                        RealConst2d v, Real t) const {
    yakl::timer_start("compute_tendency");

    yakl::timer_start("h_tendency");
    compute_h_tendency(htend, h, v);
    yakl::timer_stop("h_tendency");

    yakl::timer_start("v_tendency");
    compute_v_tendency(vtend, h, v);
    yakl::timer_stop("v_tendency");

    additional_tendency(htend, vtend, h, v, t);

    yakl::timer_stop("compute_tendency");
  }

  ShallowWaterBase(PlanarHexagonalMesh &mesh, Real f0);
  ShallowWaterBase(PlanarHexagonalMesh &mesh, Real f0, Real grav);
};

struct ShallowWater : ShallowWaterBase {
  Real2d hflux;
  void compute_h_tendency(Real2d vtend, RealConst2d h,
                          RealConst2d v) const override;
  void compute_v_tendency(Real2d htend, RealConst2d h,
                          RealConst2d v) const override;
  Real energy_integral(RealConst2d h, RealConst2d v) const override;

  ShallowWater(PlanarHexagonalMesh &mesh, Real f0);
  ShallowWater(PlanarHexagonalMesh &mesh, Real f0, Real grav);
};

struct LinearShallowWater : ShallowWaterBase {
  Real h0;
  void compute_h_tendency(Real2d vtend, RealConst2d h,
                          RealConst2d v) const override;
  void compute_v_tendency(Real2d htend, RealConst2d h,
                          RealConst2d v) const override;
  Real energy_integral(RealConst2d h, RealConst2d v) const override;

  LinearShallowWater(PlanarHexagonalMesh &mesh, Real h0, Real f0);
  LinearShallowWater(PlanarHexagonalMesh &mesh, Real h0, Real f0, Real grav);
};
} // namespace omega
