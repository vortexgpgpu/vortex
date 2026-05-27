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

// Canonical graphics ABI header. Two sections:
//   - on-wire types + 8888 pixel helpers (host-visible, always available;
//     defined in <vx_gfx_abi.h>, the shared single-source-of-truth header)
//   - device-side intrinsics for TEX / OM / RASTER (kernel-only, gated
//     on __VORTEX__)
//
// Host code (mesa, runtime, simx) and device code both include this
// header for the shared types. graphics.h in sw/runtime/include/
// builds on top of this and adds the host-only API (Binning, vertex_t,
// DCR address helpers).

#pragma once

#include <vx_gfx_abi.h>
#include <VX_types.h>

///////////////////////////////////////////////////////////////////////////////
// Kernel-only intrinsics for the fixed-function TEX / OM / RASTER units.
// Encodings (CUSTOM1 family):
//   funct3=1, R4-type, funct2=stage : vx_tex          (texture sample)
//   funct3=2, R4-type, funct2=0     : vx_om           (output-merger write)
//   funct3=3, R-type,  funct7=0     : vx_rast         (raster pop)
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

// Output-merger write: (x, y, face, color, depth)
inline void vx_om(unsigned x, unsigned y, unsigned face, unsigned color, unsigned depth) {
  unsigned pos_face = (y << 16) | (x << 1) | face;
  __asm__ volatile (".insn r4 %0, 2, 0, x0, %1, %2, %3"
      :: "i"(RISCV_CUSTOM1), "r"(pos_face), "r"(color), "r"(depth));
}

// Raster pop: returns next quad descriptor from the rasterizer.
inline unsigned vx_rast() {
  unsigned ret;
  __asm__ volatile (".insn r %1, 3, 0, %0, x0, x0"
      : "=r"(ret) : "i"(RISCV_CUSTOM1));
  return ret;
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
