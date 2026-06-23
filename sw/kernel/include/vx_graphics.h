// Copyright © 2019-2023
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Graphics ABI header. Provides on-wire types and pixel helpers via
// <vx_gfx_abi.h> (host and device), plus device-side TEX/OM/RASTER
// intrinsics (gated on __VORTEX__).

#pragma once

#include <vx_gfx_abi.h>
#include <VX_types.h>

///////////////////////////////////////////////////////////////////////////////
// Kernel-only intrinsics for the fixed-function TEX / OM / RASTER units.
// Encodings (CUSTOM1 family):
//   funct3=1, R4-type, funct2=stage : vx_tex          (texture sample)
//   funct3=2, R4-type, funct2=0     : vx_om           (output-merger write)
//   funct3=3, R-type,  funct7=0     : vx_rast         (raster pop)
//   funct3=3, R-type,  funct7=1     : vx_fwd_run      (FWD driver entry, v2)
//   funct3=4, R-type,  funct7=0     : vx_rast_begin   (per-frame trigger)
// Trap as illegal-instruction unless VX_CFG_EXT_TEX_ENABLE /
// VX_CFG_EXT_OM_ENABLE / VX_CFG_EXT_RASTER_ENABLE is set.
///////////////////////////////////////////////////////////////////////////////

#ifdef __VORTEX__

#include <vx_intrinsics.h>

namespace vortex {
namespace graphics {

// Texture sample: (stage, u, v, lod) -> texel
inline unsigned vx_tex(unsigned stage, unsigned u, unsigned v, unsigned lod) {
  unsigned ret;
  __asm__ volatile (".insn r4 %1, 1, %2, %0, %3, %4, %5"
      : "=r"(ret)
      : "i"(RISCV_CUSTOM1), "i"(stage), "r"(u), "r"(v), "r"(lod));
  return ret;
}

// Texture sample on the shared graphics window (single mode). u,v are read from
// the window at slot base `in_slot` (u@in_slot, v@in_slot+1) — stage them with
// vx_rt_set first; `lod` is explicit. The texel lands in the window at `out_slot`
// (read it back with vx_rt_get_after(out_slot, handle)) and is also returned in
// rd as the scoreboard sync handle. `stage` and `out_slot` are compile-time
// constants (they ride funct7). CUSTOM1 funct3=5, R-type.
inline unsigned vx_tex4_single(unsigned stage, unsigned lod, unsigned in_slot, unsigned out_slot) {
  unsigned handle;
  __asm__ volatile (".insn r %1, 5, %2, %0, %3, %4"
      : "=r"(handle)
      : "i"(RISCV_CUSTOM1), "i"((((out_slot) << 2) | ((stage) << 1))), "r"(lod), "r"(in_slot));
  return handle;
}

// Texture sample on the shared graphics window, quad mode (hardware LOD). One
// thread owns a 2x2 quad: u[0..3] at window slots in_slot..in_slot+3, v[0..3] at
// in_slot+4..in_slot+7 (frags 0=(x,y) 1=(x+1,y) 2=(x,y+1) 3=(x+1,y+1)). rs1
// carries the texture dims {logh<<16 | logw}; the unit computes one integer mip
// LOD from the quad derivatives. The four texels land in the window at
// out_slot..out_slot+3 (read them with vx_rt_get_after over that window); rd
// returns the scoreboard sync handle. stage and out_slot are compile-time
// constants (they ride funct7). CUSTOM1 funct3=5, R-type, funct7.mode=1.
inline unsigned vx_tex4_quad(unsigned stage, unsigned logw, unsigned logh,
                             unsigned in_slot, unsigned out_slot) {
  unsigned handle;
  unsigned dims = (logw & 0xffff) | (logh << 16);
  __asm__ volatile (".insn r %1, 5, %2, %0, %3, %4"
      : "=r"(handle)
      : "i"(RISCV_CUSTOM1), "i"((((out_slot) << 2) | ((stage) << 1) | 1u)), "r"(dims), "r"(in_slot));
  return handle;
}

// Output-merger submit on the shared graphics window (vx_om4 — the sole OM op).
// One thread owns a 2x2 quad: color[0..3] at window slots base..base+3, depth[0..3]
// at base+4..base+7 (stage them with vx_rt_set first; frags 0=(x,y) 1=(x+1,y)
// 2=(x,y+1) 3=(x+1,y+1)). `desc` is the vx_rast pos_mask (cov_mask[3:0], quad
// origin qx@[4 +: 14] / qy@[18 +: 13]) with `face` in bit 31. The unit submits
// each covered sub-pixel (pos_x=(qx<<1)|(F&1), pos_y=(qy<<1)|(F>>1)) to the OM
// core. Fire-and-forget (rd=x0). CUSTOM1 funct3=2, R-type.
inline void vx_om4(unsigned desc, unsigned base) {
  __asm__ volatile (".insn r %0, 2, 0, x0, %1, %2"
      :: "i"(RISCV_CUSTOM1), "r"(desc), "r"(base));
}

// Raster pop: returns next quad descriptor from the rasterizer.
inline unsigned vx_rast() {
  unsigned ret;
  __asm__ volatile (".insn r %1, 3, 0, %0, x0, x0"
      : "=r"(ret) : "i"(RISCV_CUSTOM1));
  return ret;
}

// Fragment Work Distributor run (RASTER dispatch v2). Called by the per-core
// "driver" warp: arms the FWD with the fragment-shader context pointer (rs1) and
// BLOCKS until the FWD has drained the rasterizer and all launched fragment waves
// have retired (single-owner epoch, C5). While blocked, the FWD pulls covered
// quads from the rasterizer, packs NUM_THREADS-quad waves, seeds each lane's
// frag_payload_t into the wave's LMEM, and launches the waves onto the core's free
// warp slots. Replaces the vx_rast() poll loop + bcoord CSRs + pos_mask sentinel.
// CUSTOM1 funct3=3 (raster family), funct7=1 — a sub-op of vx_rast (funct7=0).
// Returns a sync handle in rd that lands only when the epoch drains; the caller
// consumes it to stay parked until then (mirrors the vx_tex4 handle pattern —
// the SFU op has no natural stall otherwise).
inline unsigned vx_fwd_run(const void* frag_ctx) {
  unsigned handle;
  __asm__ volatile (".insn r %1, 3, 1, %0, %2, x0"
      : "=r"(handle)
      : "i"(RISCV_CUSTOM1), "r"(frag_ctx) : "memory");
  return handle;
}

// Raster begin: per-frame trigger. Idempotent in hardware (subsequent
// calls during an active fetch are deduped via the raster's
// fetch_triggered state), so multiple warps can call it concurrently
// without a barrier. Must be issued once per frame by at least one
// participating warp before any vx_rast() call.
inline void vx_rast_begin() {
  __asm__ volatile (".insn r %0, 4, 0, x0, x0, x0"
      :: "i"(RISCV_CUSTOM1) : "memory");
}

} // namespace graphics
} // namespace vortex

#endif // __VORTEX__
