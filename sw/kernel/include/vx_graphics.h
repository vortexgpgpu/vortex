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
//   funct3=3, R4-type, funct2=mask   : vx_om_export    (fragment export -> OM aperture)
//   funct3=5, R4-type, funct2=stage  : vx_tex          (texture sample, registers only)
// RASTER has no kernel op: the raster engine launches the fragment shader
// on-device (push); the stamp rides inside the launch (read as CSRs).
// Trap as illegal-instruction unless VX_CFG_EXT_TEX_ENABLE /
// VX_CFG_EXT_OM_ENABLE / VX_CFG_EXT_RASTER_ENABLE is set.
///////////////////////////////////////////////////////////////////////////////

#ifdef __VORTEX__

#include <vx_intrinsics.h>

namespace vortex {
namespace graphics {

// Texture sample. u,v,lod come from registers and the texel is returned in rd —
// the TEX unit takes its operands in registers. `stage` is a compile-time
// constant (it rides funct2).
// CUSTOM1 funct3=5, R4-type.
//
// There is no hardware-LOD form. A lane owns one pixel, so it derives the integer
// mip from its quad neighbours with vx_tex_auto_lod() (<vx_tex_lod.h>) and passes
// the level in. Every lane of the quad must be active for that to work — see
// frag_payload_t on why helper lanes run.
inline unsigned vx_tex(unsigned stage, unsigned u, unsigned v, unsigned lod) {
  unsigned texel;
  __asm__ volatile (".insn r4 %1, 5, %2, %0, %3, %4, %5"
      : "=r"(texel)
      : "i"(RISCV_CUSTOM1), "i"(stage), "r"(u), "r"(v), "r"(lod));
  return texel;
}

// ── fragment export: the aperture store ─────────────────────────────────────
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
#define vx_frag_pos()     ((uint32_t)csr_read(VX_CSR_FRAG_POS))
#define vx_frag_pid()     ((uint32_t)csr_read(VX_CSR_FRAG_PID))

// This lane's pixel, and whether the primitive actually covers it.
#define vx_frag_x(p)       VX_FRAG_POS_X((p).pos)
#define vx_frag_y(p)       VX_FRAG_POS_Y((p).pos)
#define vx_frag_covered(p) VX_FRAG_POS_COVERED((p).pos)

// Load this lane's fragment stamp {pos, pid} into `p`.
//
// The stamp arrives with the launch — the raster engine packs it into the launch
// message and the core lands it in this warp's launch registers before the warp
// is ever activated — so reading it is two CSR reads, no window op and no memory
// traffic. There is no bcoord payload: the FS recomputes per-corner edge values
// from the primitive edges and its own pixel.
#define vx_frag_load(p) do { \
  (p).pos = vx_frag_pos();   \
  (p).pid = vx_frag_pid();   \
} while (0)

} // namespace graphics
} // namespace vortex

#endif // __VORTEX__
