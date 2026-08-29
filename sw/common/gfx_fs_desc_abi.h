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

// i64 constant-buffer base slots in the resident FS descriptor table.
// Ceiling = 8 = push constants (slot 0) + up to 7 descriptor sets (slots 1..7,
// lavapipe binds descriptor set N at constant-buffer index N+1). Raising this
// requires widening the table + the fragment kernel's per-slot reads in lockstep.
#define GFX_FS_DESC_SLOTS   8u

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
// bit-slicing (the pitch is padded to a power of two):
//   offset = ((face << (xbits + ybits)) | (y << xbits) | x) << record_shift
// The shift amounts depend on the render-target size, which is a per-draw value,
// so they cannot be baked into the shader at JIT time -- they ride the arg block.
// The host programs the matching VX_DCR_OM_APERTURE_* registers so the cluster's
// OM ingress decodes the same address.
#define GFX_FS_ARG_APERTURE 11u

// FS kernel arg-block slot carrying the per-primitive flat-varying array: the
// provoking vertex's varying words, copied verbatim, GFX_FS_FLAT_WORDS of them
// per primitive and indexed by the same primitive id the wrapper uses to reach
// the primitive record.
//
// A flat varying cannot travel through the interpolation planes at all. Setup
// premultiplies every plane by 1/w and quantises it to Q7.24, which is defined
// on numbers; a flat varying's bit pattern is not necessarily a number -- every
// integer varying is flat, and a small integer read as a float is a denormal
// that quantises to zero. So the words ride beside the planes rather than
// through them, and the wrapper reads them without arithmetic.
//
// A side array rather than a field in the primitive record: the record is a
// fixed-function layout that RASTER also fetches, and widening it would move
// every draw's stride for a feature most draws do not use.
#define GFX_FS_ARG_FLAT     12u

// Scalar words carried per primitive -- the same twelve the interpolation
// planes hold, in the same order [u,v,r,g,b,a,w0..w5], so a varying's flat word
// is at the lane index it would have interpolated at. Copying all twelve keeps
// the device kernel free of any per-shader mask: which lanes are flat is a
// fact the fragment kernel is compiled with, not one setup has to be told.
#define GFX_FS_FLAT_WORDS   12u

// The sample count deliberately has NO arg slot. It is a fragment-shader variant
// key, so the value is fixed when the kernel is translated and the emitter bakes
// it in as a constant; a slot would be a second source of truth for a fact the
// variant already decides.

#define GFX_FS_APERTURE_PACK(xbits, ybits, shift) \
   (((uint32_t)(xbits) & 0xffu) | (((uint32_t)(ybits) & 0xffu) << 8) \
    | (((uint32_t)(shift) & 0xffu) << 16))

// Maximum colour attachments the MRT output-merger fallback handles (mirror of
// gfx_sw.h VX_OM_MAX_RT). Vulkan requires maxColorAttachments >= 4.
#define GFX_OM_MAX_RT       4u
