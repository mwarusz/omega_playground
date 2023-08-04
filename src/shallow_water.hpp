#pragma once 

#include <common.hpp>
#include <planar_hexagonal_mesh.hpp>

namespace omega {

struct ShallowWater {
  PlanarHexagonalMesh *mesh;
  Real grav = 9.81;
  // TODO: generalize to variable f
  Real f0;
  
  virtual void compute_h_tendency(Real1d vtend, Real1d h, Real1d v) const;
  virtual void compute_v_tendency(Real1d htend, Real1d h, Real1d v) const;
  virtual void additional_tendency(Real1d htend, Real1d vtend, Real1d h, Real1d v, Real t) const {}
  virtual Real mass_integral(Real1d h) const;
  virtual Real circulation_integral(Real1d v) const;
  virtual Real energy_integral(Real1d h, Real1d v) const;
  
  void compute_tendency(Real1d htend, Real1d vtend, Real1d h, Real1d v, Real t) const {
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

  ShallowWater(PlanarHexagonalMesh &mesh, Real f0);
  ShallowWater(PlanarHexagonalMesh &mesh, Real f0, Real grav);
};

struct LinearShallowWater : ShallowWater {
  Real h0;
  void compute_h_tendency(Real1d vtend, Real1d h, Real1d v) const override;
  void compute_v_tendency(Real1d htend, Real1d h, Real1d v) const override;
  Real energy_integral(Real1d h, Real1d v) const override;

  LinearShallowWater(PlanarHexagonalMesh &mesh, Real h0, Real f0);
  LinearShallowWater(PlanarHexagonalMesh &mesh, Real h0, Real f0, Real grav);
};
}
