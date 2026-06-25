#include <vx_spawn2.h>
#include <vx_graphics.h>
#include "common.h"
#include <pipe_frontend.h>   // setup_k + binning_k (-I gfx_setup_kernel)

// gfx_v2 device front end + RASTER fragment kernel in one module: the shared
// pipeline (setup_k / binning_k) produces RASTER's tilebuf + primbuf; this
// trivial fragment kernel writes the covered pixels.

__kernel void kernel_main(frag_arg_t* __UNIFORM__ arg) {
  using namespace vortex::graphics;
  const uint32_t out_color = 0xffffffff;
  vx_rast_begin();
  uint32_t lane = csr_read(VX_CSR_THREAD_ID);
  frag_payload_t* pls = (frag_payload_t*)__local_mem()
                      + (unsigned)vx_warp_id() * (unsigned)vx_num_threads();
  for (;;) {
    unsigned drained = vx_frag_fetch(pls);
    if (drained) return;
    uint32_t pos_mask = pls[lane].pos_mask;
    if (pos_mask == 0) continue;
    uint32_t mask = (pos_mask >> 0) & 0xf;
    uint32_t x    = (pos_mask >> 4) & ((1u << (VX_RASTER_DIM_BITS - 1)) - 1);
    uint32_t y    = (pos_mask >> (4 + (VX_RASTER_DIM_BITS - 1))) & ((1u << (VX_RASTER_DIM_BITS - 1)) - 1);
    for (uint32_t i = 0; i < 4; ++i) {
      if (mask & (1u << i)) {
        uint32_t px = (x << 1) + (i & 1);
        uint32_t py = (y << 1) + (i >> 1);
        auto dst_ptr = reinterpret_cast<uint32_t*>(
            arg->cbuf_addr + px * arg->cbuf_stride + py * arg->cbuf_pitch);
        *dst_ptr = out_color;
      }
    }
  }
}
