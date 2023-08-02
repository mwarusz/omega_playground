#pragma once

#include <YAKL.h>
#include <cmath>

namespace omega {

using Real = double;
using Int = int;

YAKL_INLINE constexpr Real operator""_fp(long double x) {
  return x;
}

constexpr Real pi = M_PI;

using yakl::c::parallel_for;
using yakl::c::SimpleBounds;
using yakl::c::Bounds;
using yakl::SArray;

using Real1d = yakl::Array<Real, 1, yakl::memDevice, yakl::styleC>;
using Real2d = yakl::Array<Real, 2, yakl::memDevice, yakl::styleC>;
using Real3d = yakl::Array<Real, 3, yakl::memDevice, yakl::styleC>;
using Real4d = yakl::Array<Real, 4, yakl::memDevice, yakl::styleC>;

using RealHost1d = yakl::Array<Real, 1, yakl::memHost, yakl::styleC>;
using RealHost2d = yakl::Array<Real, 2, yakl::memHost, yakl::styleC>;
using RealHost3d = yakl::Array<Real, 3, yakl::memHost, yakl::styleC>;
using RealHost4d = yakl::Array<Real, 4, yakl::memHost, yakl::styleC>;

using Int1d = yakl::Array<Int, 1, yakl::memDevice, yakl::styleC>;
using Int2d = yakl::Array<Int, 2, yakl::memDevice, yakl::styleC>;
using Int3d = yakl::Array<Int, 3, yakl::memDevice, yakl::styleC>;
using Int4d = yakl::Array<Int, 4, yakl::memDevice, yakl::styleC>;

using IntHost1d = yakl::Array<Int, 1, yakl::memHost, yakl::styleC>;
using IntHost2d = yakl::Array<Int, 2, yakl::memHost, yakl::styleC>;
using IntHost3d = yakl::Array<Int, 3, yakl::memHost, yakl::styleC>;
using IntHost4d = yakl::Array<Int, 4, yakl::memHost, yakl::styleC>;
}
