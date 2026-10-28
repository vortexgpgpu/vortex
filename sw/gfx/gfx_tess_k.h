#pragma once

// The tessellator's stitch kernel: tess_stitch_k, one thread per stitch unit
// of a batch of patches (gfx_tess_draw_t), grid-strided. Each unit writes its
// primitives into the batch's primitive list at their place in the reference
// tessellator's order -- its patch's first primitive plus the unit's offset --
// so the list comes out in patch order, and within a patch in the order
// llvmpipe produces, without any thread waiting on another. Device-only (pulls
// vx_spawn2.h).

#include <vx_spawn2.h>
#include <gfx_frontend_abi.h>
#include <gfx_tess.h>

__kernel void tess_stitch_k(gfx_tess_draw_t* __UNIFORM__ arg) {
  const gfx_tess_draw_t& d = *arg;
  const auto scan = reinterpret_cast<const uint32_t*>(d.scan_addr);
  uint32_t total = d.scan_addr ? d.units : d.units * d.patches;
  if (d.units_addr) {
    const uint32_t n = *reinterpret_cast<const volatile uint32_t*>(d.units_addr);
    total = n < total ? n : total;
  }
  auto list = reinterpret_cast<pipe_list_entry_t*>(d.list_addr);
  const uint32_t step = gridDim.x * blockDim.x;
  for (uint32_t u = blockIdx.x * blockDim.x + threadIdx.x; u < total; u += step) {
    uint32_t p, s, vbase, pbase;
    const gfx_tess_patch_t* desc;
    uint32_t id;
    if (scan) {
      const uint32_t k = u + d.first_unit;
      p = gfx_tess::locate(scan, d.patches, 2, k);
      s = k - scan[3 * p + 2];
      vbase = scan[3 * p] - d.first_point;
      pbase = scan[3 * p + 1] - d.first_prim;
      desc = gfx_tess::descriptor(d, p);
      id = desc->prim_id;
    } else {
      p = u / d.units;
      s = u - p * d.units;
      vbase = p * d.points;
      pbase = p * d.prims;
      desc = gfx_tess::descriptor(d, p);
      id = gfx_tess::primitive_id(d, p);
    }
    const gfx_tess::sink dst = { reinterpret_cast<uint32_t*>(list + pbase),
                                 sizeof(pipe_list_entry_t) / 4u, vbase,
                                 id & PIPE_LIST_AUX_MASK };
    gfx_tess::unit(*desc, s, dst);
  }
}
