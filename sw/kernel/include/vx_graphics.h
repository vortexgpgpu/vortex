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
//   funct3=2, R-type,  funct7=0     : vx_om4          (output-merger, windowed)
//   funct3=5, R-type,  funct7=...   : vx_tex4         (texture sample, windowed)
// RASTER has no kernel op in v2: the raster engine launches the fragment shader
// on-device (push); the payload is in the gfx window at warp launch.
// Trap as illegal-instruction unless VX_CFG_EXT_TEX_ENABLE /
// VX_CFG_EXT_OM_ENABLE / VX_CFG_EXT_RASTER_ENABLE is set.
///////////////////////////////////////////////////////////////////////////////

#ifdef __VORTEX__

#include <vx_intrinsics.h>
#include <vx_gfx_window.h>   // vx_gfx_set / vx_gfx_get* (SETW/GETW window primitives)

namespace vortex {
namespace graphics {

// Texture sample on the shared graphics window (single mode). u,v are read from
// the window at slot base `in_slot` (u@in_slot, v@in_slot+1) — stage them with
// vx_gfx_set first; `lod` is explicit. The texel lands in the window at `out_slot`
// (read it back with vx_gfx_get_after(out_slot, handle)) and is also returned in
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
// out_slot..out_slot+3 (read them with vx_gfx_get_after over that window); rd
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
// at base+4..base+7 (stage them with vx_gfx_set first; frags 0=(x,y) 1=(x+1,y)
// 2=(x,y+1) 3=(x+1,y+1)). `desc` is the raster pos_mask (cov_mask[3:0], quad
// origin qx@[4 +: 14] / qy@[18 +: 13]) with `face` in bit 31. The unit submits
// each covered sub-pixel (pos_x=(qx<<1)|(F&1), pos_y=(qy<<1)|(F>>1)) to the OM
// core. Fire-and-forget (rd=x0). CUSTOM1 funct3=2, R-type.
inline void vx_om4(unsigned desc, unsigned base) {
  __asm__ volatile (".insn r %0, 2, 0, x0, %1, %2"
      :: "i"(RISCV_CUSTOM1), "r"(desc), "r"(base));
}

// RASTER dispatch v2 is PUSH: the raster engine's work distributor launches the
// fragment shader once per covered-quad wave (no pull op). The per-lane payload
// is already staged in this warp's gfx register window (slots
// VX_GFX_FRAG_SLOT_BASE..) at warp launch (FWD-5, zero LMEM/LSU traffic); the FS
// runs straight-line and reads it via the helpers below. `vx_rast_fetch` no
// longer exists.

// VX_GFX_FRAG_SLOT_BASE + the full window slot map live in <vx_gfx_window.h>.
// frag_payload_t layout: word 0 = pos_mask, 1 = pid, 2+axis*4+corner =
// bcoord[axis][corner] (matches RTL VX_gfx_window_pkg + SimX).

// Read one staged frag_payload_t `word` for this lane from the gfx window. The
// payload was seeded at warp launch (no fetch, no scoreboard token to chain on).
// `word` must be a compile-time constant (the slot rides the funct7 immediate).
#define vx_frag_payload(word) \
  vx_gfx_get(VX_GFX_FRAG_SLOT_BASE + (word))

// Load this lane's full staged frag_payload_t from the gfx window into `p`.
// Unrolled so every slot index is a compile-time immediate.
#define vx_frag_load(p) do { \
  (p).pos_mask     = vx_frag_payload(0); \
  (p).pid          = vx_frag_payload(1); \
  (p).bcoord[0][0] = vx_frag_payload(2); \
  (p).bcoord[0][1] = vx_frag_payload(3); \
  (p).bcoord[0][2] = vx_frag_payload(4); \
  (p).bcoord[0][3] = vx_frag_payload(5); \
  (p).bcoord[1][0] = vx_frag_payload(6); \
  (p).bcoord[1][1] = vx_frag_payload(7); \
  (p).bcoord[1][2] = vx_frag_payload(8); \
  (p).bcoord[1][3] = vx_frag_payload(9); \
  (p).bcoord[2][0] = vx_frag_payload(10); \
  (p).bcoord[2][1] = vx_frag_payload(11); \
  (p).bcoord[2][2] = vx_frag_payload(12); \
  (p).bcoord[2][3] = vx_frag_payload(13); \
} while (0)

} // namespace graphics
} // namespace vortex

#endif // __VORTEX__
