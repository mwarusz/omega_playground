#include <omega.hpp>

using namespace omega;

void ke_grad_func(const Real2d &vn_tend_edge,
	          const RealConst2d &ke_cell,
		  Int nedges,
		  Int nlayers_vec,
                  const KineticEnergyGradOnEdge &ke_grad_edge) {
    omega_parallel_for(
        "compute_vtend3", {nedges, nlayers_vec},
        KOKKOS_LAMBDA(Int iedge, Int kchunk) {
          ke_grad_edge(vn_tend_edge, iedge, kchunk, ke_cell);
        });
}
