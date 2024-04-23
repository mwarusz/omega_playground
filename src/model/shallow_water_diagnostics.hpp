#pragma once

#include "shallow_water_model.hpp"
#include "shallow_water_state.hpp"

namespace omega {
Real energy_integral(const ShallowWaterState &state,
                     const ShallowWaterModel &model);
Real mass_integral(const ShallowWaterState &state,
                   const ShallowWaterModel &model);
Real circulation_integral(const ShallowWaterState &state,
                          const ShallowWaterModel &model);
}; // namespace omega
