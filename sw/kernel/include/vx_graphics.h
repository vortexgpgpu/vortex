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

// RASTER dispatch is PUSH: the raster engine launches the fragment shader once
// per covered-quad wave (no pull op), and the per-lane stamp rides INSIDE that
// launch. The core lands it in the warp's launch registers before the warp is
// activated, so the shader reads its own pixel from a register — no window op,
// no LMEM, no memory traffic.

// This lane's fragment stamp, straight out of its launch registers.
#define vx_frag_posmask() ((uint32_t)csr_read(VX_CSR_FRAG_POSMASK))
#define vx_frag_pid()     ((uint32_t)csr_read(VX_CSR_FRAG_PID))

// Load this lane's fragment stamp {pos_mask, pid} into `p`.
//
// The stamp arrives with the launch — the raster engine packs it into the launch
// message and the core lands it in this warp's launch registers before the warp
// is ever activated — so reading it is two CSR reads, no window op and no memory
// traffic. There is no bcoord payload: the FS recomputes per-corner edge values
// from the primitive edges + the quad origin (decoded from pos_mask).
#define vx_frag_load(p) do { \
  (p).pos_mask = vx_frag_posmask(); \
  (p).pid      = vx_frag_pid(); \
} while (0)

} // namespace graphics
} // namespace vortex

#endif // __VORTEX__
