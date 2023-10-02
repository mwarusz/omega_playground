#pragma once

#include <common.hpp>
#include <mpas_mesh.hpp>
#include <string>

namespace omega {

struct FileMesh : MPASMesh {

  FileMesh(const std::string &filename, Int nlayers = 1);

  void convert_fortran_indices_to_cxx() const;
  void rescale_radius(Real radius) const;
};
} // namespace omega
