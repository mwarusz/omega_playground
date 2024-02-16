#pragma once

#include <Kokkos_Core.hpp>
#include <cmath>

namespace omega {

using Real = double;
using Int = int;

KOKKOS_INLINE_FUNCTION constexpr Real operator""_fp(long double x) { return x; }

constexpr Int vector_length = OMEGA_VECTOR_LENGTH;

constexpr Real pi = M_PI;

#define OMEGA_SCOPE(a, b) auto &a = b

using Kokkos::deep_copy;
using Kokkos::parallel_for;
using Kokkos::parallel_reduce;
using Kokkos::TeamThreadRange;
using Kokkos::ThreadVectorRange;

using ExecSpace = Kokkos::DefaultExecutionSpace;
using MemSpace = ExecSpace::memory_space;
using Layout = Kokkos::LayoutRight;

using RangePolicy = Kokkos::RangePolicy<ExecSpace>;
using TeamPolicy = Kokkos::TeamPolicy<ExecSpace>;
using TeamMember = TeamPolicy::member_type;

constexpr Int tile1 = 1;
constexpr Int tile2 = 64;

constexpr Int team_size = 1;
constexpr Int vector_size = 64;

template <Int N>
using MDRangePolicy = Kokkos::MDRangePolicy<
    ExecSpace, Kokkos::Rank<N, Kokkos::Iterate::Right, Kokkos::Iterate::Right>>;

using Real1d = Kokkos::View<Real *, Layout, MemSpace>;
using Real2d = Kokkos::View<Real **, Layout, MemSpace>;
using Real3d = Kokkos::View<Real ***, Layout, MemSpace>;
using Real4d = Kokkos::View<Real ****, Layout, MemSpace>;

using RealConst1d = Kokkos::View<Real const *, Layout, MemSpace>;
using RealConst2d = Kokkos::View<Real const **, Layout, MemSpace>;
using RealConst3d = Kokkos::View<Real const ***, Layout, MemSpace>;
using RealConst4d = Kokkos::View<Real const ****, Layout, MemSpace>;

using Int1d = Kokkos::View<Int *, Layout, MemSpace>;
using Int2d = Kokkos::View<Int **, Layout, MemSpace>;
using Int3d = Kokkos::View<Int ***, Layout, MemSpace>;
using Int4d = Kokkos::View<Int ****, Layout, MemSpace>;

using IntConst1d = Kokkos::View<Int const *, Layout, MemSpace>;
using IntConst2d = Kokkos::View<Int const **, Layout, MemSpace>;
using IntConst3d = Kokkos::View<Int const ***, Layout, MemSpace>;
using IntConst4d = Kokkos::View<Int const ****, Layout, MemSpace>;

} // namespace omega
