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
//
// PRISM CW-BVH4 on-disk format — Phase 4.
//
// Scene-kind 2 (kRtuSceneKindBvh4) routes the walker to this format.
// Backward-compat: scene-kind 0 (TRI_LIST) and 1 (TLAS) still walked
// by the Phase 1-11 flat scanner in rtu_core.cpp.
//
// Design notes
// ------------
// - Each internal node is 64 B (one cache line). Four children per node,
//   8-bit quantized AABBs with per-axis exponent + common origin. This
//   layout is structurally identical to the original vortex-raytracing
//   prototype (rt_core.h:23-52) and to Intel Xe-HPG / Mesa's vk_bvh.h
//   internal-node shape (modulo bit-pack details). Bit-compatibility
//   with vk_bvh.h is a Phase 4-late refinement; for now the host-side
//   BVH fixture builder emits this layout directly.
// - Leaves carry their kind tag in the first byte after the node header
//   so the walker can fan out to TRI / INSTANCE / PROCEDURAL paths
//   from one decode point.
// - Triangle stride (40 B) and TLAS instance stride (64 B) match the
//   flat-list constants in rtu_core.cpp so the existing intersection
//   helpers (`ray_triangle`, `affine_inverse_transform_ray`) drop in
//   unchanged.

#ifndef _VX_RTU_BVH_TYPES_H_
#define _VX_RTU_BVH_TYPES_H_

#include <cstdint>

namespace vortex {

// ---------------------------------------------------------------------
// Node kinds (low 8 bits of `kind` word)
// ---------------------------------------------------------------------
constexpr uint32_t kVxBvhKindInternal   = 0;
constexpr uint32_t kVxBvhKindLeafTri    = 1;
constexpr uint32_t kVxBvhKindLeafInst   = 2;
constexpr uint32_t kVxBvhKindLeafProc   = 3;

constexpr uint32_t kVxBvhKindMask       = 0xffu;          // low byte = kind tag
constexpr uint32_t kVxBvhCountShift     = 8;              // bits 8..15 = num_children / prim_count
constexpr uint32_t kVxBvhCountMask      = 0xffu;

// ---------------------------------------------------------------------
// Child offset encoding (each child_offsets[i] entry)
// ---------------------------------------------------------------------
constexpr uint32_t kVxBvhChildEmpty     = 0;              // entry not used
constexpr uint32_t kVxBvhChildLeafFlag  = 0x80000000u;    // bit 31: 1 = leaf, 0 = internal
constexpr uint32_t kVxBvhChildOffsetMask= 0x7fffffffu;    // bits 0..30: byte offset from BVH root

// ---------------------------------------------------------------------
// Fan-out width. PRISM Phase 4 = 4 (CW-BVH4). Phase 4-late could widen
// to 6 (CW-BVH6, Intel layout) by extending the qaabb / child_offsets
// arrays; the walker is structurally independent of this constant.
// ---------------------------------------------------------------------
constexpr uint32_t kVxBvhWidth          = 4;

// ---------------------------------------------------------------------
// 64-byte internal node. One cache line. Fan-out 4.
//
//   uint32 kind                : bits 0..7 = kVxBvhKindInternal,
//                                bits 8..15 = num_children (1..4)
//   float  origin[3]            : 12 B — common origin for child AABB
//                                 quantization
//   int8   exp[3]               : 3 B — per-axis exponent e such that
//                                 actual_step = 2^e. AABB child bounds
//                                 reconstruct as
//                                   min = origin + qaabb_min * 2^e
//                                   max = origin + qaabb_max * 2^e
//   uint8  pad0                 : 1 B
//   uint32 child_offsets[4]     : 16 B — byte offset from BVH root to
//                                 each child node; bit 31 = is_leaf,
//                                 value 0 = empty (no child)
//   uint8  qaabb_min[4][3]      : 12 B — quantized child mins
//   uint8  qaabb_max[4][3]      : 12 B — quantized child maxs
//
// Total = 4 + 12 + 4 + 16 + 12 + 12 = 60 B; pad with 4 B.
// ---------------------------------------------------------------------
struct VxBvhInternalNode {
  uint32_t kind;
  float    origin[3];
  int8_t   exp[3];
  uint8_t  pad0;
  uint32_t child_offsets[kVxBvhWidth];
  uint8_t  qaabb_min[kVxBvhWidth][3];
  uint8_t  qaabb_max[kVxBvhWidth][3];
  uint8_t  pad1[4];
};
static_assert(sizeof(VxBvhInternalNode) == 64,
              "BVH internal node must be exactly one 64 B cache line");

// ---------------------------------------------------------------------
// 16-byte leaf header. Triangles / instances / AABBs follow inline.
//
//   uint32 kind             : bits 0..7  = kVxBvhKindLeafTri/Inst/Proc
//                             bits 8..15 = prim_count
//   uint32 geometry_index   : Vulkan VK_GEOMETRY_INDEX
//   uint32 flags            : bit 0 = OPAQUE (all prims), bit 1 = forced
//                             non-opaque, bits 8..15 = SBT_IDX
//   uint32 reserved
//
// After this header (offset +16):
//   LeafTri  : VxBvhTri[prim_count]      — 40 B each (matches flat-list)
//   LeafInst : VxBvhInstance[prim_count] — 64 B each (matches flat-list)
//   LeafProc : VxBvhProcAabb[prim_count] — 24 B each
// ---------------------------------------------------------------------
struct VxBvhLeafHeader {
  uint32_t kind;
  uint32_t geometry_index;
  uint32_t flags;
  uint32_t reserved;
};
static_assert(sizeof(VxBvhLeafHeader) == 16,
              "BVH leaf header must be exactly 16 B");

constexpr uint32_t kVxBvhLeafHeaderBytes = 16;

// Per-leaf-flags layout (mirrors the flat-list per-tri flags so the
// existing AHS / CHS / MISS classifier in compute_intersections() can
// drop in unchanged).
constexpr uint32_t kVxBvhLeafFlagOpaque       = 0x1u;
constexpr uint32_t kVxBvhLeafFlagNonOpaque    = 0x2u;
constexpr uint32_t kVxBvhLeafSbtIdxShift      = 8;
constexpr uint32_t kVxBvhLeafSbtIdxMask       = 0xffu;

// ---------------------------------------------------------------------
// 40-byte triangle. Same layout as the flat-list kPhase2TriStride
// triangles so the same Möller-Trumbore tester applies; the
// `flags` word also has the same meaning (OPAQUE, PROCEDURAL,
// SBT_IDX) so per-tri overrides of the leaf-wide flags work.
// ---------------------------------------------------------------------
struct VxBvhTri {
  float    v0[3];     // 12
  float    v1[3];     // 12
  float    v2[3];     // 12
  uint32_t flags;     // 4 — bit 0 OPAQUE, bit 1 PROCEDURAL, bits 8..15 SBT_IDX
};
static_assert(sizeof(VxBvhTri) == 40,
              "BVH triangle must match flat-list 40 B stride");

constexpr uint32_t kVxBvhTriStride = 40;

// ---------------------------------------------------------------------
// 64-byte instance. Same layout as the flat-list TLAS instance, with
// an absolute BLAS root pointer in `blas_root_byte_offset` (offset
// from the SCENE root, not from a private BLAS base — gives us a
// single base address for the whole TLAS+BLAS bundle).
//
//   floats 0..11           : 48 B object→world affine (3x4, row-major)
//   uint32 blas_root_off   : 4 B byte offset to this instance's BLAS
//                            root node from the scene-buffer base
//   uint32 custom_id       : 4 B VK_INSTANCE_CUSTOM_INDEX_KHR
//   uint32 instance_id     : 4 B HW-assigned instance ID
//   uint32 reserved        : 4 B
// ---------------------------------------------------------------------
struct VxBvhInstance {
  float    xform[12];
  uint32_t blas_root_byte_offset;
  uint32_t custom_id;
  uint32_t instance_id;
  uint32_t reserved;
};
static_assert(sizeof(VxBvhInstance) == 64,
              "BVH instance must match flat-list 64 B stride");

constexpr uint32_t kVxBvhInstanceStride = 64;

// ---------------------------------------------------------------------
// 24-byte procedural-leaf AABB record. The walker hits this and yields
// IS (intersection shader) — the kernel's IS does the real
// shape-vs-ray test using whatever data it pre-stashed via the SBT.
// ---------------------------------------------------------------------
struct VxBvhProcAabb {
  float    aabb_min[3];
  float    aabb_max[3];
};
static_assert(sizeof(VxBvhProcAabb) == 24,
              "BVH proc-AABB record must be 24 B");

// ---------------------------------------------------------------------
// Scene header for kRtuSceneKindBvh4 (16 B, parsed by drain_mem_rsp
// like the flat-list headers):
//
//   uint32 root_node_offset : byte offset of the root internal node
//                             from the scene buffer base (typically 16)
//   uint32 scene_kind       : MUST equal kRtuSceneKindBvh4 (= 2)
//   uint32 node_count       : total number of internal nodes (diagnostic)
//   uint32 leaf_count       : total number of leaves (diagnostic)
//
// node_count + leaf_count are not consumed by the walker — they let the
// fixture builder cross-check what it serialized.
// ---------------------------------------------------------------------
struct VxBvhSceneHeader {
  uint32_t root_node_offset;
  uint32_t scene_kind;
  uint32_t node_count;
  uint32_t leaf_count;
};
static_assert(sizeof(VxBvhSceneHeader) == 16,
              "BVH scene header must be 16 B");

}  // namespace vortex

#endif  // _VX_RTU_BVH_TYPES_H_
