#pragma once 

#include <common.hpp>
#include <planar_hexagonal_mesh.hpp>

namespace omega {

struct ShallowWater {
  PlanarHexagonalMesh *mesh;
  Real grav = 9.81;
  Real f0;
  
  virtual void compute_h_tendency(Real1d vtend, Real1d h, Real1d v) const;
  virtual void compute_v_tendency(Real1d htend, Real1d h, Real1d v) const;
  virtual Real compute_energy(Real1d h, Real1d v) const;
  
  void compute_tendency(Real1d htend, Real1d vtend, Real1d h, Real1d v) const {
    compute_h_tendency(htend, h, v);
    compute_v_tendency(vtend, h, v);
  }

  ShallowWater(PlanarHexagonalMesh &mesh, Real f0);
  ShallowWater(PlanarHexagonalMesh &mesh, Real f0, Real grav);
};

struct LinearShallowWater : ShallowWater {
  Real h0;
  void compute_h_tendency(Real1d vtend, Real1d h, Real1d v) const override;
  void compute_v_tendency(Real1d htend, Real1d h, Real1d v) const override;
  Real compute_energy(Real1d h, Real1d v) const override;

  LinearShallowWater(PlanarHexagonalMesh &mesh, Real h0, Real f0);
  LinearShallowWater(PlanarHexagonalMesh &mesh, Real h0, Real f0, Real grav);
};
}
