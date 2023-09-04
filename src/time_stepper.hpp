#pragma once

#include <common.hpp>
#include <shallow_water.hpp>

namespace omega {

struct TimeStepper {
  ShallowWaterBase *shallow_water;
  virtual void do_step(Real t, Real dt, Real2d h, Real2d v) const = 0;
  TimeStepper(ShallowWaterBase &shallow_water)
      : shallow_water(&shallow_water) {}
};

struct LSRKStepper : TimeStepper {
  Int nstages = 5;
  std::vector<Real> rka;
  std::vector<Real> rkb;
  std::vector<Real> rkc;
  Real2d htend;
  Real2d vtend;

  LSRKStepper(ShallowWaterBase &shallow_water);
  void do_step(Real t, Real dt, Real2d h, Real2d v) const override;
};

struct RK4Stepper : TimeStepper {
  Int nstages = 4;
  std::vector<Real> rka;
  std::vector<Real> rkb;
  std::vector<Real> rkc;
  Real2d htend;
  Real2d vtend;
  Real2d hprovis;
  Real2d vprovis;
  Real2d hold;
  Real2d vold;

  RK4Stepper(ShallowWaterBase &shallow_water);
  void do_step(Real t, Real dt, Real2d h, Real2d v) const override;
};
} // namespace omega
