#pragma once

#include "mpas_mesh.hpp"
#include <common.hpp>
#include <netcdf>
#include <string>

namespace omega {

struct FileMesh : MPASMesh {

  FileMesh(const std::string &filename, Int nlayers = 1);
  FileMesh(const netCDF::NcFile &mesh_file, Int nlayers);

  void convert_fortran_indices_to_cxx() const;
  void rescale_radius(Real radius) const;
};

} // namespace omega
