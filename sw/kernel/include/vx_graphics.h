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
//   funct3=3, R4-type, funct7=mask   : vx_om_export    (fragment export -> OM aperture)
//   funct3=5, R-type,  funct7=...   : vx_tex4         (texture sample, windowed)
// RASTER has no kernel op in v2: the raster engine launches the fragment shader
// on-device (push); the stamp rides inside the launch (read as CSRs).
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

// ── fragment export: the aperture store (gfx_subsystem_redesign §5) ──────────
//
// The shader exports a fragment by STORING to the OM aperture. There is no OM bus
// and no window staging: the cluster's OM steer peels the write off the L1->L2
// trunk and the OM ingress turns it back into a {pos, colour, depth, face}
// request for the unchanged VX_om_core.
//
// The aperture address is SHIFT-ONLY (the pitch is padded to a power of two), so
// the ingress decodes it by bit-slicing instead of dividing:
//     offset = ((face << (XBITS+YBITS)) | (y << XBITS) | x) << RECORD_SHIFT
// XBITS/YBITS/RECORD_SHIFT come from the OM DCRs; the runtime programs them and
// passes them to the kernel, so the shader just shifts and adds.
#define VX_OM_APERTURE_ADDR(xbits, ybits, record_shift, x, y, face) \
  ((VX_MEM_OM_BASE_ADDR) +                                          \
   ((((uint32_t)(face) << ((xbits) + (ybits)))                      \
     | ((uint32_t)(y) << (xbits))                                   \
     | (uint32_t)(x)) << (record_shift)))

// vx_om_export — one fragment. CUSTOM1 funct3=3, R4-type, rd=x0 (posted).
// funct7[1:0] = {has_depth, has_colour}: a shader may emit colour only (the
// common case — early-Z owns the depth test AND the depth write), depth only
// (z-prepass / shadow map), or both (gl_FragDepth). The uop expander turns this
// into one ordinary store per word; the LSU never learns that OM exists.
#define vx_om_export(addr, color, depth, mask)                     \
  __asm__ volatile (".insn r4 %0, 3, %1, x0, %2, %3, %4"           \
      :: "i"(RISCV_CUSTOM1), "i"(mask), "r"(addr), "r"(color), "r"(depth))

// The three record shapes.
#define vx_om_export_color(addr, color)        vx_om_export(addr, color, 0, 1)
#define vx_om_export_depth(addr, depth)        vx_om_export(addr, 0, depth, 2)
#define vx_om_export_both(addr, color, depth)  vx_om_export(addr, color, depth, 3)


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
