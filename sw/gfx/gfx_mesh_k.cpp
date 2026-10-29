// Compile unit linked into each translated mesh shader: the persistent mesh
// grid that calls the shader body (vp_mesh_main) once per mesh workgroup and
// compacts what it emits. Device-only.

#include "gfx_mesh_k.h"

__kernel void mesh_k(gfx_mesh_arg_t* __UNIFORM__ arg) {
  gfx_mesh::mesh_stage(arg);
}
