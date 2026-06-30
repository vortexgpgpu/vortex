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

// Graphics register-window access primitives (SETW / GETW). The window is the
// scoreboarded register file the fixed-function units stage operands through:
// TEX reads u/v and writes texels, OM reads color/depth, RASTER stages the frag
// payload, and the RTU stages its hit/object-ray attributes. These ops are the
// shared substrate both vx_graphics.h (TEX/OM/RASTER) and vx_raytrace.h (RTU)
// build on, so they live here — below both — and neither layer depends on the
// other. Encoding: CUSTOM1, funct3=6; funct2 selects (1=SETW, 3=GETW); the slot
// rides funct7[6:2].
//
// NOTE: this packages the window primitives so the overlap between the gfx and
// RTU slot regions is visible and mutually exclusive by convention. The longer-
// term direction (gfx_v2 §3.1 / C4) is per-unit scoreboard-retired windows that
// remove the cross-unit sharing entirely; this header does not entrench it.

#pragma once

#include <vx_intrinsics.h>
#include <VX_types.h>
#include <stdint.h>

// ── Window slot map (the shared 32-slot register window) ─────────────────────
// SETW/GETW carry the slot in funct7[6:2], so the window is exactly 32 slots,
// carved between graphics and the RTU:
//
//   slots 0..7    RTU per-ray window (origin/dir/tmin/tmax -> f0..f7)
//   slots 8..21   GFX frag payload  (VX_GFX_FRAG_SLOT_BASE.., 14 words)
//                 — OVERLAPS the RTU object-ray (8..13) + hit window (14..)
//   slots 8..24   RTU object-ray + hit attributes
//   slot  31      RTU HIT_SBT_IDX  (window is full)
//
// The gfx and RTU regions overlap (slots 8..21). This is correct ONLY under the
// invariant that a warp never has live window state for fragment shading and ray
// tracing at the same time — it is not enforced by hardware. The asserts below
// pin the bounds so an accidental layout change fails to compile; the long-term
// fix (gfx_v2 §3.1 / doctrine C4) is per-unit scoreboard-retired windows that
// remove the overlap by construction.
#define VX_GFX_FRAG_SLOT_BASE 8       // frag_payload_t base (14 words: 0=pos_mask,
                                      // 1=pid, 2+axis*4+corner=bcoord[axis][corner];
                                      // matches RTL VX_gfx_window_pkg + SimX)

// static_assert (C++ keyword) — these headers are only included in C++ kernels;
// portable across gcc and clang, unlike the C11 _Static_assert.
static_assert(VX_GFX_FRAG_SLOT_BASE + 14 <= 32,
              "gfx frag window overflows the 32-slot register window");
#ifdef VX_RT_HIT_SBT_IDX
static_assert(VX_RT_HIT_SBT_IDX < 32,
              "RTU window overflows the 32-slot register window");
#endif

#ifdef __VORTEX__

// SETW — write one window slot (funct2=1; slot in funct7[6:2], value in rs1, no
// rd). Stage an operand (e.g. a TEX u/v or OM color/depth) before the FF op.
#define vx_gfx_set(slot, val) \
  __asm__ volatile (".insn r %0, 6, %1, x0, %2, x0" \
      :: "i"(RISCV_CUSTOM1), "i"(((slot) << 2) | 1), "r"(val))

// GETW — read one window slot into rd (funct2=3; slot in funct7[6:2]). rs1 = x0,
// so NO scoreboard dependency. Use inside a trap-context dispatcher where the
// window is already populated; ordinary code chains a dependency via the *_dep /
// *_after forms below.
#define vx_gfx_get(slot) ({ \
  uint32_t __v; \
  __asm__ volatile (".insn r %1, 6, %2, %0, x0, x1" \
      : "=r"(__v) \
      : "i"(RISCV_CUSTOM1), "i"(((slot) << 2) | 3)); \
  __v; })

// GETW chained on a scoreboard token — same op, but `tok` rides rs1 so the
// scoreboard stalls this read until the producer that returned `tok` has written
// the slot (the SFU ignores rs1's value; the encoded slot still lives in funct7).
// No memory clobber: use when the dependency is purely register-window (e.g. an
// RTU hit-attribute read chained on a vx_rt_wait status token).
#define vx_gfx_get_dep(slot, tok) ({ \
  uint32_t __v; \
  __asm__ volatile (".insn r %1, 6, %2, %0, %3, x1" \
      : "=r"(__v) \
      : "i"(RISCV_CUSTOM1), "i"(((slot) << 2) | 3), "r"(tok)); \
  __v; })

// GETW chained on a token AND a compiler memory barrier. Identical to
// vx_gfx_get_dep plus a "memory" clobber, so a callback-written memory load
// placed after it in source is not hoisted above it; combined with in-order
// issue (the read stalls on `tok` until the producer retires) this guarantees
// the load observes the producer's stores. Use for post-vx_rt_wait attribute
// reads that precede a dependent memory load.
#define vx_gfx_get_after(slot, tok) ({ \
  uint32_t __v; \
  __asm__ volatile (".insn r %1, 6, %2, %0, %3, x1" \
      : "=r"(__v) \
      : "i"(RISCV_CUSTOM1), "i"(((slot) << 2) | 3), "r"(tok) \
      : "memory"); \
  __v; })

#endif // __VORTEX__
