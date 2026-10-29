// Compile unit linked into each translated task shader: the persistent task
// grid that calls the shader body (vp_task_main) once per task workgroup.
// Device-only.

#include "gfx_mesh_k.h"

__kernel void task_k(gfx_task_arg_t* __UNIFORM__ arg) {
  gfx_mesh::task_stage(arg);
}
