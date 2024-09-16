target_sources(omega PRIVATE mpas_mesh.cpp planar_hexagonal_mesh.cpp)
if (${netCDFCxx_FOUND})
  target_sources(omega PRIVATE file_mesh.cpp)
endif()
