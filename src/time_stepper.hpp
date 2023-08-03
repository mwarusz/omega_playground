#pragma once 

#include <common.hpp>
#include <shallow_water.hpp>

namespace omega {
  
  struct TimeStepper {
    ShallowWater* shallow_water;
    virtual void do_step(Real dt, Real1d h, Real1d v) const = 0;
    TimeStepper(ShallowWater &shallow_water) : shallow_water(&shallow_water) {}
  };

  struct LSRKStepper : TimeStepper {
    Int nstages = 5;
    std::vector<Real> rka;
    std::vector<Real> rkb;
    Real1d htend;
    Real1d vtend;

    LSRKStepper(ShallowWater &shallow_water);
    void do_step(Real dt, Real1d h, Real1d v) const override;
  };
}
