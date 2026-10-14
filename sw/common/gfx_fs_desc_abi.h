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

// gfx_v2 per-draw fragment-shader descriptor / constant table.
//
// A flat, resident i64 table the fragment shader reads through a pointer
// carried in its kernel arg block (slot GFX_FS_ARG_DESC): per-dispatch data
// reached via a resident pointer, never per-thread. Each
// slot i holds the device base address of the fragment stage's constant
// buffer i, indexed the lavapipe way:
//   slot 0 = push constants        (load_ubo(0)/load_push_constant base)
//   slot 1 = descriptor set-0 blob (UBO/SSBO/combined-sampler descriptors)
//   slot k = descriptor set (k-1)  (additional sets)
// Unbound slots are 0. load_ubo(i, off)/load_push_constant read table[i]+off
// directly; load_ssbo dereferences the buffer pointer inside the descriptor
// blob at table[1] (the descriptor's pointers are relocated host->device by
// the host before upload). Nothing in the device or SimX fixed-function path
// interprets this table — it is ordinary shader-read memory — so the contract
// lives between the host arg-block builder (mesa vp_raster) and the generated
// fragment kernel (mesa vp_nir_to_llvm); no RTL/SimX change is required.

#pragma once

#include <stdint.h>
#include <VX_types.h>

// i64 constant-buffer base slots in a stage's resident descriptor table: push
// constants (slot 0) and 8 descriptor sets (slots 1..8 -- lavapipe binds set N at
// constant-buffer index N+1 and exposes 8 bound sets). Kernels index the table by
// constant-buffer index, so only the host side sizes it.
#define GFX_FS_DESC_SLOTS   9u

// FS kernel arg-block slot carrying the descriptor-table device address.
// The HW frag wrapper uses arg slots 0..2 and the SW-raster wrapper 0..8, so
// the table pointer rides slot 9 in both.
#define GFX_FS_ARG_DESC     9u

// MRT: FS kernel arg-block slot carrying the resident gfx_sw_omcolor_t[]
// device address (per-attachment colour/blend/write-mask state) for a draw that
// targets >1 colour attachment. The FS wrapper reads it only when the shader has
// more than one colour output; a 1-RT draw leaves it 0 and keeps the fast path.
#define GFX_FS_ARG_MRT      10u

// FS kernel arg-block slot carrying the packed OM aperture geometry:
//   bits [7:0] = xbits, [15:8] = ybits, [23:16] = record_shift
// A fragment export is a STORE to the OM aperture, and its address is formed by
// bit-slicing (the pitch is padded to a power of two). The driver exports
// colour+depth pairs, whose address names the pixel only:
//   offset = ((y << xbits) | x) << record_shift
// with the attachment and face in the depth word (VX_OM_DEPTH_WORD_*).
// The shift amounts depend on the render-target size, which is a per-draw value,
// so they cannot be baked into the shader at JIT time -- they ride the arg block.
// The host programs the matching VX_DCR_OM_APERTURE_* registers so the cluster's
// OM ingress decodes the same address.
#define GFX_FS_ARG_APERTURE 11u

// FS kernel arg-block slots locating a draw's varyings: the rast_vary_t array
// (gfx_frontend_abi.h), indexed by the same primitive id as the primitive record,
// and the VS output records it points into with their byte stride. The wrapper
// interpolates each fragment input out of the three source records with the
// record's perspective-correct barycentrics; a flat input reads the provoking
// record's word verbatim, so an integer varying keeps its bits.
#define GFX_FS_ARG_VARY     12u
#define GFX_FS_ARG_VSREC    14u
#define GFX_FS_ARG_VSTRIDE  15u

// FS kernel arg-block slot carrying the view index (gl_ViewIndex) of the draw:
// the view a multiview pass is replaying, 0 otherwise. Input-attachment reads
// address their layer with it even outside multiview.
#define GFX_FS_ARG_VIEW     13u

// FS kernel arg-block slots bounding the fragment depth, as raw depth-plane
// values (window z in Q7.24), each GFX_FS_ZPAIR(lo, hi): a fragment whose plane
// depth is outside ZKILL is dropped (depth clipping against the viewport's depth
// range), and the depth it merges is clamped into ZCLAMP (depth clamp), then to
// the attachment's [0, 1].
#define GFX_FS_ARG_ZKILL    16u
#define GFX_FS_ARG_ZCLAMP   17u
// User clip distances in the VS records: GFX_FS_CLIP_PACK(byte offset, count).
// A fragment where any interpolated distance is negative is dropped. 0 = none.
#define GFX_FS_ARG_CLIP     18u
// Which of a primitive's three records a flat input reads (0 or 2).
#define GFX_FS_ARG_PROVOKE  19u
// Rasterization state: GFX_FS_RAST_PACK(pattern, factor, smooth) | GFX_FS_RAST_CENTRE.
// With a non-zero factor a line is stippled: a fragment whose counter
// (SETUP_PRIM_STIPPLE_PLANE) lands on a clear pattern bit is dropped. With smooth
// set, the fragment's coverage (SETUP_PRIM_SMOOTH_*) scales the alpha of every
// colour output. With CENTRE set, the draw is rasterized single-sampled into a
// multisample target (Bresenham and smooth lines): every sample takes the pixel
// centre's coverage.
#define GFX_FS_ARG_RAST     20u
// The gfx_sw_omstate_t a shader with early fragment tests is tested against by
// its wrapper (gfx_ds_early_sw), ahead of the shader. 0 for any other shader.
#define GFX_FS_ARG_EARLY_DS 21u
// A float depth target's depth clamp, as the bits of two floats
// GFX_FS_ZPAIR(lo, hi): -inf/+inf when depth clamp is off. A D32F fragment
// evaluates SETUP_PRIM_ZFLOAT_PLANE and is clamped to this range only.
#define GFX_FS_ARG_ZCLAMPF  22u
#define GFX_FS_ARG_COUNT    23u

#define GFX_FS_ZPAIR(lo, hi) \
   ((uint64_t)(uint32_t)(int32_t)(lo) | ((uint64_t)(uint32_t)(int32_t)(hi) << 32))
#define GFX_FS_RAST_PACK(pattern, factor, smooth) \
   (((uint32_t)(pattern) & 0xffffu) | (((uint32_t)(factor) & 0x1ffu) << 16) \
    | ((smooth) ? GFX_FS_RAST_SMOOTH : 0u))
#define GFX_FS_RAST_FACTOR_MASK 0x01ff0000u
#define GFX_FS_RAST_SMOOTH      0x80000000u
#define GFX_FS_RAST_CENTRE      0x40000000u
#define GFX_FS_CLIP_PACK(off, count) \
   (((uint32_t)(off) & 0xffffu) | (((uint32_t)(count) & 0xffu) << 16))

// The sample count deliberately has NO arg slot. It is a fragment-shader variant
// key, so the value is fixed when the kernel is translated and the emitter bakes
// it in as a constant; a slot would be a second source of truth for a fact the
// variant already decides.

#define GFX_FS_APERTURE_PACK(xbits, ybits, shift) \
   (((uint32_t)(xbits) & 0xffu) | (((uint32_t)(ybits) & 0xffu) << 8) \
    | (((uint32_t)(shift) & 0xffu) << 16))

// Colour attachments the output merger handles, fixed-function and software.
#define GFX_OM_MAX_RT       ((uint32_t)VX_OM_MAX_RT)
