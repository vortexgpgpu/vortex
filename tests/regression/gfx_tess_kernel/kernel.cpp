#include <vx_spawn2.h>
#include "common.h"
#include "gfx_scan_k.h"
#include "gfx_tess_k.h"
#include "gfx_tess_abi.cpp"

__kernel void kernel_main(kernel_arg_t* __UNIFORM__ arg) {
  const uint32_t step = gridDim.x * blockDim.x;
  const uint32_t g0 = blockIdx.x * blockDim.x + threadIdx.x;
  if (arg->stage == TESS_STAGE_SETUP) {
    auto lv = reinterpret_cast<const float*>(arg->levels_addr);
    auto desc = reinterpret_cast<gfx_tess_patch_t*>(arg->desc_addr);
    auto counts = reinterpret_cast<uint32_t*>(arg->counts_addr);
    for (uint32_t p = g0; p < arg->n; p += step) {
      uint32_t mode, id;
      __builtin_memcpy(&mode, &lv[8 * p + 6], 4);
      __builtin_memcpy(&id, &lv[8 * p + 7], 4);
      gfx_tess_patch_setup(mode, &lv[8 * p], id, &desc[p], &counts[3 * p]);
    }
  } else {
    auto d = reinterpret_cast<const gfx_tess_draw_t*>(arg->draw_addr);
    auto out = reinterpret_cast<uint32_t*>(arg->points_addr);
    for (uint32_t g = g0; g < arg->n; g += step) {
      gfx_tess_eval_point(d, g, &out[4 * g]);
    }
  }
}
