#pragma once

// Pass-end multisample colour resolve — device kernel entry.
//
// Kept out of gfx_frontend_k.h: this pulls in gfx_sw.h, which refuses to
// compile without GFX_SW_DIVERGENCE_OK and a raised divergence-block guard
// (sw/gfx/libgfx_sw.mk). The front-end header is included by several kernels
// that merge nothing and should not have to carry either. Include this one
// beside it where the resolve is wanted. Device-only (pulls vx_spawn2.h).

#include <vx_spawn2.h>
#include <gfx_frontend_abi.h>   // resolve_arg_t
#include <gfx_sw.h>             // gfx_sw::{om_state_t, om_store, msaa_resolve_color}

// Folds each pixel's `samples` colour samples into one with the box filter the
// software merger already defines, so a resolved pixel matches what those
// samples would have produced had the merger written the pixel once. One thread
// per pixel, grid-strided so any launch geometry covers the target.
//
// The fold runs on the device rather than on the host: what the fixed-function
// units cannot represent is served here, and it leaves the readback carrying one
// texel per pixel rather than `samples` of them, which is where the cost sits on
// a real device.
__kernel void msaa_resolve_k(resolve_arg_t* __UNIFORM__ arg) {
  const uint32_t w = arg->width, h = arg->height, s = arg->samples;
  const uint32_t n = w * h;
  const uint32_t bpp = gfx_sw::om_color_bpp(arg->color_format);
  // msaa_color_addr takes the plane's base, pitch and format through an
  // om_state_t; no other field of it is consulted on this path.
  gfx_sw::om_state_t src{};
  src.cbuf_base    = arg->src_addr;
  src.cbuf_pitch   = w * s * bpp;
  src.color_format = arg->color_format;
  auto dst = static_cast<uintptr_t>(arg->dst_addr);
  uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  uint32_t gstride = gridDim.x * blockDim.x;
  for (uint32_t i = gid; i < n; i += gstride) {
    uint32_t x = i % w, y = i / w;
    gfx_sw::om_store(dst + (uintptr_t)i * bpp, bpp,
                     gfx_sw::msaa_resolve_color(src, s, x, y));
  }
}
