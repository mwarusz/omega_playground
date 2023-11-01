#pragma once

#include <YAKL.h>
#include <cmath>

namespace omega {

using Real = double;
using Int = int;

YAKL_INLINE constexpr Real operator""_fp(long double x) { return x; }

constexpr Int vector_length = OMEGA_VECTOR_LENGTH;

constexpr Real pi = M_PI;

constexpr Int maxedges = 6;

using yakl::SArray;
using yakl::c::Bounds;
using yakl::c::parallel_for;
using yakl::c::SimpleBounds;

using Real1d = yakl::Array<Real, 1, yakl::memDevice, yakl::styleC>;
using Real2d = yakl::Array<Real, 2, yakl::memDevice, yakl::styleC>;
using Real3d = yakl::Array<Real, 3, yakl::memDevice, yakl::styleC>;
using Real4d = yakl::Array<Real, 4, yakl::memDevice, yakl::styleC>;

using RealConst1d = yakl::Array<Real const, 1, yakl::memDevice, yakl::styleC>;
using RealConst2d = yakl::Array<Real const, 2, yakl::memDevice, yakl::styleC>;
using RealConst3d = yakl::Array<Real const, 3, yakl::memDevice, yakl::styleC>;
using RealConst4d = yakl::Array<Real const, 4, yakl::memDevice, yakl::styleC>;

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
} // namespace omega
