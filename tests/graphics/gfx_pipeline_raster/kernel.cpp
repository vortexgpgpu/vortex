#include <vx_spawn2.h>
#include <vx_graphics.h>
#include "common.h"
#include <gfx_frontend_k.h>   // setup_k + binning_k (-I gfx_setup_kernel)

// gfx_v2 device front end + RASTER fragment kernel in one module: the shared
// pipeline (setup_k / binning_k) produces RASTER's tilebuf + primbuf; this
// trivial fragment kernel writes the covered pixels.

// RASTER dispatch (push): straight-line fragment shader, launched once per packed
// fragment wave with this lane's pixel already in its launch registers.
__kernel void kernel_main(frag_arg_t* __UNIFORM__ arg) {
  using namespace vortex::graphics;
  const uint32_t out_color = 0xffffffff;
  frag_payload_t p;
  vx_frag_load(p);
  // A lane the primitive misses is a helper: it must not branch out of the shader
  // (a covered neighbour may still need to shuffle a value out of it), so coverage
  // gates the export, not the control flow.
  if (vx_frag_covered(p)) {
    auto dst_ptr = reinterpret_cast<uint32_t*>(
        arg->cbuf_addr + vx_frag_x(p) * arg->cbuf_stride
                       + vx_frag_y(p) * arg->cbuf_pitch);
    *dst_ptr = out_color;
  }
}
