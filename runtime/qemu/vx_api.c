#include <stdio.h>
#include <stdlib.h>
<<<<<<< HEAD
=======
#include "../vx_api/vx_api.h"
>>>>>>> fpga_synthesis

#ifdef __cplusplus
extern "C" {
#endif

<<<<<<< HEAD
struct pocl_context_t {
  uint32_t num_groups[3];
  uint32_t global_offset[3];
  uint32_t local_size[3];
  uint8_t *printf_buffer;
  uint32_t *printf_buffer_position;
  uint32_t printf_buffer_capacity;
  uint32_t work_dim;
};

=======
>>>>>>> fpga_synthesis
typedef void (*pocl_workgroup_func) (
  void * /* args */,
  void * /* pocl_context */,
  uint32_t /* group_x */,
  uint32_t /* group_y */,
  uint32_t /* group_z */
);

<<<<<<< HEAD
void pocl_spawn(struct pocl_context_t * ctx, const pocl_workgroup_func pfn, void * arguments) {
=======
void pocl_spawn(struct pocl_context_t * ctx, pocl_workgroup_func pfn, const void * args) {
>>>>>>> fpga_synthesis
  uint32_t x, y, z;
  for (z = 0; z < ctx->num_groups[2]; ++z)
    for (y = 0; y < ctx->num_groups[1]; ++y)
      for (x = 0; x < ctx->num_groups[0]; ++x)
        (pfn)(arguments, ctx, x, y, z);
}

#ifdef __cplusplus
}
#endif