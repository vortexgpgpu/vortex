#include <stdio.h>
#include <stdlib.h>
#include "../vx_api/vx_api.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef void (*pocl_workgroup_func) (
  void * /* args */,
  void * /* pocl_context */,
  uint32_t /* group_x */,
  uint32_t /* group_y */,
  uint32_t /* group_z */
);

void pocl_spawn(struct pocl_context_t * ctx, pocl_workgroup_func pfn, const void * args) {
  uint32_t x, y, z;
  for (z = 0; z < ctx->num_groups[2]; ++z)
    for (y = 0; y < ctx->num_groups[1]; ++y)
      for (x = 0; x < ctx->num_groups[0]; ++x)
        (pfn)(arguments, ctx, x, y, z);
}

#ifdef __cplusplus
}
#endif