#pragma once

#include <common.hpp>
#include <mpas_mesh.hpp>
#include <string>

namespace omega {

struct FileMesh : MPASMesh {

  FileMesh(const std::string &filename, Int nlayers = 1);
};
} // namespace omega
