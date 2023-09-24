#pragma once

#include <common.hpp>
#include <mpas_mesh.hpp>

namespace omega {

struct FileMesh : MPASMesh {

  FileMesh(const char *filename, Int nlayers = 1);
};
} // namespace omega
