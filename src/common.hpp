#pragma once

#include <Kokkos_Core.hpp>
#include <Kokkos_SIMD.hpp>
#include <cmath>
#include <iostream>
#include <utility>

namespace omega {

using Real = double;
using Int = int;

KOKKOS_INLINE_FUNCTION constexpr Real operator""_fp(long double x) { return x; }


#ifdef OMEGA_NO_SIMD
#define OMEGA_SIMD_PRAGMA
constexpr Int vector_length = 1;
#else
#define OMEGA_SIMD_PRAGMA _Pragma("omp simd")
constexpr Int vector_length = Kokkos::Experimental::native_simd<Real>::size();
#endif

#ifdef OMEGA_KOKKOS_SIMD
using Vec = Kokkos::Experimental::native_simd<Real>;
using VecTag = Kokkos::Experimental::element_aligned_tag;
#else
using Vec = Real[vector_length];
#endif

constexpr Real pi = M_PI;

#define OMEGA_SCOPE(a, b) const auto &a = b

using Kokkos::deep_copy;
using Kokkos::create_mirror_view;
using Kokkos::parallel_for;
using Kokkos::parallel_reduce;
using Kokkos::TeamThreadRange;
using Kokkos::ThreadVectorRange;

using ExecSpace = Kokkos::DefaultExecutionSpace;
using HostExecSpace = Kokkos::DefaultHostExecutionSpace;
constexpr bool exec_is_gpu =
    !Kokkos::SpaceAccessibility<ExecSpace, Kokkos::HostSpace>::accessible;

using MemSpace = ExecSpace::memory_space;
using HostMemSpace = HostExecSpace::memory_space;
using Layout = Kokkos::LayoutRight;

using RangePolicy = Kokkos::RangePolicy<ExecSpace>;
using TeamPolicy = Kokkos::TeamPolicy<ExecSpace>;
using TeamMember = TeamPolicy::member_type;

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

template <class V>
inline Kokkos::View<typename V::const_data_type, Layout, MemSpace>
const_view(const V &view) {
  return view;
}

template <Int N> struct DefaultTile;

template <> struct DefaultTile<1> {
  static constexpr Int value[] = {64};
};

template <> struct DefaultTile<2> {
  static constexpr Int value[] = {1, 64};
};

template <> struct DefaultTile<3> {
  static constexpr Int value[] = {1, 1, 64};
};

template <Int N, class F>
inline void omega_parallel_for(const std::string &label,
                               Int const (&upper_bounds)[N], const F &f,
                               Int const (&tile)[N] = DefaultTile<N>::value) {
  if constexpr (N == 1) {
    const auto policy = RangePolicy(0, upper_bounds[0]);
    parallel_for(label, policy, f);
  } else {
    const Int lower_bounds[N] = {0};
    const auto policy = MDRangePolicy<N>(lower_bounds, upper_bounds, tile);
    parallel_for(label, policy, f);
  }
}

// without label
template <Int N, class F>
inline void omega_parallel_for(Int const (&upper_bounds)[N], const F &f,
                               Int const (&tile)[N] = DefaultTile<N>::value) {
  omega_parallel_for("", upper_bounds, f, tile);
}

template <Int N, class F, class R>
inline void
omega_parallel_reduce(const std::string &label, Int const (&upper_bounds)[N],
                      const F &f, R &&reducer,
                      Int const (&tile)[N] = DefaultTile<N>::value) {
  if constexpr (N == 1) {
    const auto policy = RangePolicy(0, upper_bounds[0]);
    parallel_reduce(label, policy, f, std::forward<R>(reducer));
  } else {
    const Int lower_bounds[N] = {0};
    const auto policy = MDRangePolicy<N>(lower_bounds, upper_bounds, tile);
    parallel_reduce(label, policy, f, std::forward<R>(reducer));
  }
}

// without label
template <Int N, class F, class R>
inline void
omega_parallel_reduce(Int const (&upper_bounds)[N], const F &f, R &&reducer,
                      Int const (&tile)[N] = DefaultTile<N>::value) {
  omega_parallel_reduce("", upper_bounds, f, std::forward<R>(reducer), tile);
}

constexpr Int team_size = 1;
constexpr Int vector_size = 64;

template <class F>
inline void omega_parallel_for_outer(const std::string &label, Int upper_bound,
                                     const F &f) {
  const auto policy = TeamPolicy(upper_bound, team_size, vector_size);
  parallel_for(label, policy, f);
}

// without label
template <class F>
inline void omega_parallel_for_outer(Int upper_bound, const F &f) {
  omega_parallel_for_outer("", upper_bound, f);
}

template <class F>
inline void omega_parallel_for_inner(Int upper_bound, const F &f,
                                     const TeamMember &team_member) {
  const auto policy = ThreadVectorRange(team_member, upper_bound);
  parallel_for(policy, f);
}

#ifdef OMEGA_USE_CALIPER
#include <caliper/cali.h>
inline void timer_start(char const *label) {
  if constexpr (exec_is_gpu) {
    Kokkos::fence();
  }
  cali_begin_region(label);
}

inline void timer_stop(char const *label) {
  if constexpr (exec_is_gpu) {
    Kokkos::fence();
  }
  cali_end_region(label);
}
#else
inline void timer_start(char const *label) {}
inline void timer_stop(char const *label) {}
#endif

} // namespace omega
