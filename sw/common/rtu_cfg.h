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
// Shared host/device RTU acceleration-structure format constants. The
// vortex::raytrace host transcoder (sw/runtime/include/raytrace.h) emits these
// bytes and the SimX walker (sim/simx/rtu/rtu_bvh.h) consumes them, so both
// sides agree on one CW-BVH<W> byte layout. See rtu_isa_v2_proposal.md §5.3.

#ifndef __RTU_CFG_H__
#define __RTU_CFG_H__

#include <stdint.h>

// Scene-kind selector (scene-header word 1; also packed into VX_DCR_RTU_CONFIG).
#define RTU_SCENE_KIND_TRI_LIST  0
#define RTU_SCENE_KIND_TLAS      1
#define RTU_SCENE_KIND_BVH4      2
#define RTU_SCENE_KIND_BVH6      3

// CW-BVH byte layout (matches VxBvhSceneHeader / VxBvhLeafHeader / VxBvhTri).
#define RTU_BVH_SCENE_HDR_BYTES  16   // root_off, scene_kind, node_count, leaf_count
#define RTU_BVH_LEAF_HDR_BYTES   16   // kind|count<<8, geometry_index, flags, prim_base
#define RTU_BVH_TRI_STRIDE       40   // v0[3], v1[3], v2[3], flags

// Internal-node byte layout (matches VxBvhInternalNode / VxBvh6InternalNode in
// the SimX walker, sim/simx/rtu/rtu_bvh.h — which the host cannot include, so
// the shared offsets live here). Common header (kind/origin/exp/pad) is
// identical for both widths; only the per-child array sizes differ:
//   kind         @0   (uint32: low byte INTERNAL, bits 8..15 = num_children)
//   origin[3]    @4   (float : common quantization origin = node min corner)
//   exp[3]       @16  (int8  : per-axis exponent; step = 2^exp)
//   pad0         @19  (uint8)
//   child_off[W] @20  (uint32: byte offset from scene base | leaf flag; 0=empty)
//   qaabb_min    @20+4*W (uint8[W][3]: child min, quantized, rounded DOWN)
//   qaabb_max    @20+7*W (uint8[W][3]: child max, quantized, rounded UP)
// CW-BVH4 = 64 B (one cache line), CW-BVH6 = 96 B.
#define RTU_BVH_NODE4_BYTES      64
#define RTU_BVH_NODE6_BYTES      96
#define RTU_BVH_NODE_ORIGIN_OFF  4
#define RTU_BVH_NODE_EXP_OFF     16
#define RTU_BVH_NODE_CHILD_OFF   20

// child_offsets[] entry encoding (bits 0..30 = byte offset from scene base).
#define RTU_BVH_CHILD_EMPTY      0u
#define RTU_BVH_CHILD_LEAF_FLAG  0x80000000u   // bit 31: 1 = leaf child
#define RTU_BVH_CHILD_OFFSET_MASK 0x7fffffffu

// Per-lane BVH pre-fetch budget (mirrors SimX kRtuMaxBvhSceneBytes); the host
// transcoder refuses to emit a scene larger than the walker can pre-fetch.
#define RTU_BVH_MAX_SCENE_BYTES  16384

// Leaf-header `kind` word: low byte = leaf kind, bits 8.. = primitive count.
#define RTU_BVH_KIND_INTERNAL    0
#define RTU_BVH_KIND_LEAF_TRI    1
#define RTU_BVH_KIND_LEAF_INST   2
#define RTU_BVH_KIND_LEAF_PROC   3
#define RTU_BVH_COUNT_SHIFT      8

// Per-triangle / per-leaf flags word (bit 0 OPAQUE, bit 1 PROCEDURAL,
// bits 8..15 SBT index).
#define RTU_BVH_FLAG_OPAQUE      0x1u
#define RTU_BVH_FLAG_PROCEDURAL  0x2u
#define RTU_BVH_SBT_IDX_SHIFT    8

#ifdef __cplusplus
namespace vortex {
namespace raytrace {

// Host-side triangle: three world-space vertices + the device flags word.
struct host_tri_t {
  float    v0[3];
  float    v1[3];
  float    v2[3];
  uint32_t flags;   // RTU_BVH_FLAG_* | (sbt_idx << RTU_BVH_SBT_IDX_SHIFT)
};

// Host acceleration structure handed to build_bvh_scene(): a flat triangle
// list with one geometry index. The transcoder emits a single-leaf CW-BVH<W>
// scene (root IS the leaf) — a valid, if unpartitioned, BVH.
struct host_bvh_t {
  const host_tri_t* tris;
  uint32_t          tri_count;
  uint32_t          geometry_index;
};

} // namespace raytrace
} // namespace vortex
#endif // __cplusplus

#endif // __RTU_CFG_H__
