#pragma once

#include <common.hpp>
#include <shallow_water.hpp>

namespace omega {

struct TimeStepper {
  ShallowWaterModelBase *shallow_water;
  virtual void do_step(Real t, Real dt,
                       const ShallowWaterState &state) const = 0;
  TimeStepper(ShallowWaterModelBase &shallow_water)
      : shallow_water(&shallow_water) {}
};

struct LSRKStepper : TimeStepper {
  Int nstages = 5;
  std::vector<Real> rka;
  std::vector<Real> rkb;
  std::vector<Real> rkc;
  ShallowWaterState tend;

  LSRKStepper(ShallowWaterModelBase &shallow_water);
  void do_step(Real t, Real dt, const ShallowWaterState &state) const override;
};

struct RK4Stepper : TimeStepper {
  Int nstages = 4;
  std::vector<Real> rka;
  std::vector<Real> rkb;
  std::vector<Real> rkc;
  ShallowWaterState tend;
  ShallowWaterState provis_state;
  ShallowWaterState old_state;

  RK4Stepper(ShallowWaterModelBase &shallow_water);
  void do_step(Real t, Real dt, const ShallowWaterState &state) const override;
};
} // namespace omega
