#pragma once

#include <common.hpp>
#include <shallow_water.hpp>

namespace omega {

struct TimeStepper {
  ShallowWaterModel *m_shallow_water;
  virtual void do_step(Real t, Real dt,
                       const ShallowWaterState &state) const = 0;
  TimeStepper(ShallowWaterModel &shallow_water)
      : m_shallow_water(&shallow_water) {}
};

struct LSRKStepper : TimeStepper {
  Int m_nstages;
  std::vector<Real> m_rka;
  std::vector<Real> m_rkb;
  std::vector<Real> m_rkc;
  ShallowWaterState m_tend;

  LSRKStepper(ShallowWaterModel &shallow_water, Int nstages = 5);
  void do_step(Real t, Real dt, const ShallowWaterState &state) const override;
};

struct RK4Stepper : TimeStepper {
  static constexpr Int nstages = 4;
  std::vector<Real> m_rka;
  std::vector<Real> m_rkb;
  std::vector<Real> m_rkc;
  ShallowWaterState m_tend;
  ShallowWaterState m_provis_state;
  ShallowWaterState m_old_state;

  RK4Stepper(ShallowWaterModel &shallow_water);
  void do_step(Real t, Real dt, const ShallowWaterState &state) const override;
};
} // namespace omega
