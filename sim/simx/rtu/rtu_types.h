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
// PRISM RTU — common types (header-only).
// Layer 1 of the rtu_implementation.md refactor (Option C, 13 files).
//
// This file holds every POD type and constant that's needed across
// the RTU subsystem (bus packets, scene-format constants, per-lane /
// per-slot state structs, math primitives, perf counters). Higher
// layers (rtu_isect, rtu_classifier, rtu_walker, rtu_memory, rtu_unit,
// rtu_core) include this file but no other rtu_*.h cross-references.
//
// All names live in `vortex::rtu` for cross-namespace clarity. The
// pre-existing top-level names (RtuReq, RtuRsp, RtuRspKind, ...) are
// re-exported via using-declarations into `vortex::` for back-compat
// with code outside the RTU subsystem (cluster.cpp, sfu_unit.cpp).

#ifndef _VX_RTU_TYPES_H_
#define _VX_RTU_TYPES_H_

#include <array>
#include <cstdint>
#include <ostream>
#include "instr_trace.h"
#include "constants.h"
#include "types.h"

namespace vortex { namespace rtu {

// ════════════════════════════════════════════════════════════════════
// 1. Bus packet types (Req / Rsp)
// ════════════════════════════════════════════════════════════════════
//
// Two request kinds share the RtuReq channel:
//   TRACE_NEW — vx_rt_trace fires a fresh ray.
//   CB_ACTION — vx_rt_cb_ret releases a parked context with per-lane
//               action codes (ACCEPT/IGNORE/TERMINATE/DONE).

enum class RtuReqKind : uint8_t {
  TRACE_NEW = 0,
  CB_ACTION = 1,
};

// Per-warp request packet. Carries either the per-lane ray descriptor
// snapshot (TRACE_NEW) or the per-lane cb_ret action codes (CB_ACTION).
// Simulator-only fields ride alongside for writeback routing.
struct RtuReq {
  RtuReqKind kind = RtuReqKind::TRACE_NEW;
  uint64_t uuid = 0;
  uint32_t tag  = 0;
  uint32_t tmask_bits = 0;

  // §8.6 async pool: pre-allocated slot index for TRACE_NEW.
  uint32_t slot_idx = 0;

  // Per-lane ray descriptor snapshot (TRACE_NEW only).
  std::array<uint32_t, VX_CFG_NUM_THREADS> scene_root = {};
  std::array<float,    VX_CFG_NUM_THREADS> origin_x   = {};
  std::array<float,    VX_CFG_NUM_THREADS> origin_y   = {};
  std::array<float,    VX_CFG_NUM_THREADS> origin_z   = {};
  std::array<float,    VX_CFG_NUM_THREADS> dir_x      = {};
  std::array<float,    VX_CFG_NUM_THREADS> dir_y      = {};
  std::array<float,    VX_CFG_NUM_THREADS> dir_z      = {};
  std::array<float,    VX_CFG_NUM_THREADS> tmin       = {};
  std::array<float,    VX_CFG_NUM_THREADS> tmax       = {};
  std::array<uint32_t, VX_CFG_NUM_THREADS> flags      = {};
  std::array<uint32_t, VX_CFG_NUM_THREADS> cull_mask  = {};

  // Per-lane cb_ret action codes (CB_ACTION only). One of VX_RT_CB_*.
  std::array<uint32_t, VX_CFG_NUM_THREADS> cb_action  = {};

  // P1 (CB_ACTION only): the IS-computed hit distance, read back from the
  // kernel's VX_RT_HIT_T slot at vx_rt_cb_ret time. On ACCEPT of a
  // procedural (IS) candidate the RtuCore commits this t instead of the
  // pre-IS AABB-entry candidate t.
  std::array<float,    VX_CFG_NUM_THREADS> cb_hit_t   = {};

  // Per-lane RtuCore slot handle (CB_ACTION only) — read from the kernel's
  // VX_RT_CB_HANDLE slot at vx_rt_cb_ret time. Phase 3-A2 reformation may
  // batch lanes from MULTIPLE slots into one virtual warp at CB_YIELD, so
  // the action packet routes per-lane back to the originating slot rather
  // than rely on a single warp-scoped slot id.
  std::array<uint32_t, VX_CFG_NUM_THREADS> cb_handle  = {};

  // SimX-only: routing back to per-core SfuUnit writeback. In RTL these
  // don't exist (the bus arbiter's stored route delivers the response).
  instr_trace_t* trace    = nullptr;
  uint32_t       block_id = 0;
  uint32_t       warp_id  = 0;

  RtuReq() = default;

  friend std::ostream& operator<<(std::ostream& os, const RtuReq& req) {
    os << (req.kind == RtuReqKind::TRACE_NEW ? "TRACE" : "CB_RET")
       << " tag=0x" << std::hex << req.tag << std::dec
       << ", tmask=0x" << std::hex << req.tmask_bits << std::dec
       << " (#" << req.uuid << ")";
    return os;
  }
};

// Two response kinds share the RtuRsp channel:
//   TERMINAL — slot finished (HIT or MISS). Per-lane status + hit attrs.
//   CB_YIELD — slot yielded mid-walk (AHS / IS / CHS / MISS). cb_active_mask
//              marks which lanes need a callback; cb_type / candidate-hit
//              attrs are populated for those lanes.
enum class RtuRspKind : uint8_t {
  TERMINAL = 0,
  CB_YIELD = 1,
};

struct RtuRsp {
  RtuRspKind kind = RtuRspKind::TERMINAL;
  uint64_t uuid = 0;
  uint32_t tag  = 0;

  // Per-lane terminal status + hit attributes.
  std::array<uint32_t, VX_CFG_NUM_THREADS> status            = {};
  std::array<float,    VX_CFG_NUM_THREADS> hit_t             = {};
  std::array<float,    VX_CFG_NUM_THREADS> hit_bary_u        = {};
  std::array<float,    VX_CFG_NUM_THREADS> hit_bary_v        = {};
  std::array<uint32_t, VX_CFG_NUM_THREADS> hit_primitive_id  = {};
  std::array<uint32_t, VX_CFG_NUM_THREADS> hit_instance_id   = {};
  std::array<uint32_t, VX_CFG_NUM_THREADS> hit_geometry_index = {};

  // P1 (proposal §4.2 slots 8..13): object-space ray for the hit/candidate.
  // Written to VX_RT_OBJECT_RAY_* by apply_response (TERMINAL) and
  // apply_callback_payload (CB_YIELD).
  std::array<float,    VX_CFG_NUM_THREADS> obj_o_x = {};
  std::array<float,    VX_CFG_NUM_THREADS> obj_o_y = {};
  std::array<float,    VX_CFG_NUM_THREADS> obj_o_z = {};
  std::array<float,    VX_CFG_NUM_THREADS> obj_d_x = {};
  std::array<float,    VX_CFG_NUM_THREADS> obj_d_y = {};
  std::array<float,    VX_CFG_NUM_THREADS> obj_d_z = {};

  // CB_YIELD only — yielding-lane mask + per-lane callback metadata.
  uint32_t cb_active_mask = 0;
  std::array<uint32_t, VX_CFG_NUM_THREADS> cb_type    = {};
  std::array<uint32_t, VX_CFG_NUM_THREADS> cb_handle  = {};
  std::array<uint32_t, VX_CFG_NUM_THREADS> cb_sbt_idx = {};

  instr_trace_t* trace    = nullptr;
  uint32_t       block_id = 0;
  uint32_t       warp_id  = 0;
  // §8.6: TERMINAL response carries the slot_idx so SfuUnit can look up
  // parked vx_rt_wait traces in wait_parked_ keyed by slot.
  uint32_t       slot_idx = 0;

  RtuRsp() = default;
  RtuRsp(const RtuReq& req)
    : uuid(req.uuid), tag(req.tag),
      trace(req.trace), block_id(req.block_id), warp_id(req.warp_id),
      slot_idx(req.slot_idx) {}

  friend std::ostream& operator<<(std::ostream& os, const RtuRsp& rsp) {
    os << (rsp.kind == RtuRspKind::TERMINAL ? "DONE" : "CB_YIELD")
       << " tag=0x" << std::hex << rsp.tag << std::dec
       << " (#" << rsp.uuid << ")";
    return os;
  }
};

using RtuBusArbiter = TxRxArbiter<RtuReq, RtuRsp>;

// ════════════════════════════════════════════════════════════════════
// 2. Scene-format constants (flat-list and TLAS paths; BVH4 layout is
//    in rtu_bvh.h)
// ════════════════════════════════════════════════════════════════════

constexpr uint64_t kRtuLineMask = ~uint64_t(VX_CFG_MEM_BLOCK_SIZE - 1);

// Max triangles per scene (per-lane fetch budget cap).
constexpr uint32_t kRtuMaxTrisPerScene  = 8;

// Per-triangle stride 40 B = 9 floats (v0/v1/v2 xyz) + uint32 flags.
//   bit  0     = OPAQUE (clear → AHS yield)
//   bit  1     = PROCEDURAL (yield IS instead of AHS)
//   bits 8..15 = SBT_IDX (Phase 3-A2 — keys the kernel's switch(sbt_idx))
constexpr uint32_t kPhase2TriStride       = 40;
constexpr uint32_t kPhase2TriFlagsOff     = 36;
constexpr uint32_t kPhase2TriFlagOpaque   = 0x1u;
constexpr uint32_t kPhase2TriFlagProc     = 0x2u;
constexpr uint32_t kPhase2TriSbtIdxShift  = 8;
constexpr uint32_t kPhase2TriSbtIdxMask   = 0xffu;
constexpr uint32_t kRtuSceneHeaderBytes   = 16;

// Scene-kind tag (second uint32 of every scene header):
//   0 = TRI_LIST  — flat triangle scan (Phase 1-7)
//   1 = TLAS      — flat 1-level TLAS over inline BLAS (Phase 8-11)
//   2 = BVH4      — CW-BVH4 walker (Phase 4 architectural; see rtu_bvh.h)
//   3 = BVH6      — CW-BVH6 walker (Intel Xe-HPG fan-out; shares the
//                   width-generic walker with BVH4, see rtu_bvh.h)
constexpr uint32_t kRtuSceneKindTriList = 0;
constexpr uint32_t kRtuSceneKindTlas    = 1;
constexpr uint32_t kRtuSceneKindBvh4    = 2;
constexpr uint32_t kRtuSceneKindBvh6    = 3;

// True-hardware model: the RTU is built for ONE scene format, selected at
// COMPILE time by VX_CFG_RTU_BVH_WIDTH (0 = flat triangle-list, 4 = CW-BVH4,
// 6 = CW-BVH6). TLAS instancing is an orthogonal compile-time capability
// (VX_CFG_RTU_TLAS_ENABLE), only meaningful with a flat BLAS walker. There is
// no runtime scene_kind dispatch — the configured kind below replaces it.
#ifndef VX_CFG_RTU_BVH_WIDTH
#define VX_CFG_RTU_BVH_WIDTH 4
#endif
#if VX_CFG_RTU_BVH_WIDTH == 0
  #ifdef VX_CFG_RTU_TLAS_ENABLE
    constexpr uint32_t kRtuConfiguredKind = kRtuSceneKindTlas;
  #else
    constexpr uint32_t kRtuConfiguredKind = kRtuSceneKindTriList;
  #endif
#elif VX_CFG_RTU_BVH_WIDTH == 6
  constexpr uint32_t kRtuConfiguredKind = kRtuSceneKindBvh6;
#else
  constexpr uint32_t kRtuConfiguredKind = kRtuSceneKindBvh4;
#endif

// TLAS instance record (64 B). Lives inline after the scene header for
// "TLAS + inline BLAS" layout.
//   floats 0..11   = 3x4 affine transform (rows r0|r1|r2), object→world
//   uint32 [48..52) = blas_byte_offset
//   uint32 [52..56) = custom_id (Vulkan VK_INSTANCE_CUSTOM_INDEX_KHR)
//   uint32 [56..60) = cull_mask (low byte = Vulkan instance mask;
//                     walker skips the instance if
//                     (instance_mask & ray.cull_mask) == 0). A 0 here
//                     means "no ray hits this instance" per Vulkan,
//                     so scene generators must set 0xff for the
//                     no-culling default.
//   uint32 [60..64) = reserved
constexpr uint32_t kRtuInstanceStride       = 64;
constexpr uint32_t kRtuInstanceBlasOffOff   = 48;
constexpr uint32_t kRtuInstanceCustomIdOff  = 52;
constexpr uint32_t kRtuInstanceCullMaskOff  = 56;

// Per-TLAS instance-count cap.
constexpr uint32_t kRtuMaxInstancesPerTlas = 4;

// Worst-case scene bytes (TLAS with kRtuMaxInstancesPerTlas instances
// sharing one BLAS that holds kRtuMaxTrisPerScene tris).
constexpr uint32_t kRtuMaxTriListBytes =
    kRtuSceneHeaderBytes + kRtuMaxTrisPerScene * kPhase2TriStride;
constexpr uint32_t kRtuMaxTlasSceneBytes =
    kRtuSceneHeaderBytes + kRtuMaxInstancesPerTlas * kRtuInstanceStride
    + kRtuSceneHeaderBytes + kRtuMaxTrisPerScene * kPhase2TriStride;
// CW-BVH4/6 scenes (scene_kind 2/3) are walked from a per-lane pre-fetch
// of the whole acceleration structure (rtu_memory.cpp), so the line budget
// must cover a real — if modest — mesh BVH. 16 KB holds a few-hundred-tri
// mesh (e.g. tests/raytracing/rt_raycast). Demand-fetch (issuing node reads
// mid-walk instead of pre-fetching) is the HW-faithful way to lift this cap
// for large scenes; see proposal §8.5.1.
constexpr uint32_t kRtuMaxBvhSceneBytes = 16384;
constexpr uint32_t kRtuMaxSceneBytes =
    (kRtuMaxBvhSceneBytes > kRtuMaxTriListBytes
         ? kRtuMaxBvhSceneBytes
         : kRtuMaxTriListBytes) > kRtuMaxTlasSceneBytes
        ? (kRtuMaxBvhSceneBytes > kRtuMaxTriListBytes
               ? kRtuMaxBvhSceneBytes
               : kRtuMaxTriListBytes)
        : kRtuMaxTlasSceneBytes;
// Account for worst-case alignment (byte_off = LINE_SIZE - 1).
constexpr uint32_t kRtuMaxLinesPerLane =
    (kRtuMaxSceneBytes + VX_CFG_MEM_BLOCK_SIZE - 1 + (VX_CFG_MEM_BLOCK_SIZE - 1))
    / VX_CFG_MEM_BLOCK_SIZE;

// Bytes of a TRI_LIST scene with `triangle_count` triangles (incl. header).
inline uint32_t tri_list_bytes(uint32_t triangle_count) {
  return kRtuSceneHeaderBytes + triangle_count * kPhase2TriStride;
}

// Bytes of a worst-case TLAS scene with `instance_count` instances.
inline uint32_t tlas_bytes(uint32_t instance_count) {
  return kRtuSceneHeaderBytes + instance_count * kRtuInstanceStride
       + kRtuSceneHeaderBytes + kRtuMaxTrisPerScene * kPhase2TriStride;
}

// Number of cache lines needed to cover `bytes` of scene starting at
// `byte_off` within the first cache line.
inline uint32_t lines_for_bytes(uint32_t byte_off, uint32_t bytes) {
  uint32_t end_off = byte_off + bytes;
  uint32_t n = (end_off + VX_CFG_MEM_BLOCK_SIZE - 1) / VX_CFG_MEM_BLOCK_SIZE;
  if (n > kRtuMaxLinesPerLane) n = kRtuMaxLinesPerLane;
  return n;
}

// Back-compat alias for the TRI_LIST path.
inline uint32_t lines_for_scene(uint32_t byte_off, uint32_t triangle_count) {
  return lines_for_bytes(byte_off, tri_list_bytes(triangle_count));
}

// ════════════════════════════════════════════════════════════════════
// 3. Math primitives (intersection helpers in rtu_isect.{h,cpp} use these)
// ════════════════════════════════════════════════════════════════════

struct Vec3 {
  float x, y, z;
  Vec3 operator-(const Vec3& o) const { return {x-o.x, y-o.y, z-o.z}; }
};

inline Vec3 cross(const Vec3& a, const Vec3& b) {
  return { a.y*b.z - a.z*b.y, a.z*b.x - a.x*b.z, a.x*b.y - a.y*b.x };
}

inline float dot(const Vec3& a, const Vec3& b) {
  return a.x*b.x + a.y*b.y + a.z*b.z;
}

// ════════════════════════════════════════════════════════════════════
// 4. Slot pool state (consumed by RtuCore::Impl, walkers, MemoryEngine)
// ════════════════════════════════════════════════════════════════════

enum class SlotState : uint8_t {
  RESERVED,         // §8.6: allocated by allocate_slot() but req not
                    // drained yet — drain_requests transitions to ISSUE.
  ISSUE,            // need to issue mem reads for active lanes
  AWAIT,            // mem reads outstanding
  COMPUTE,          // ready to run ray-triangle intersection
  IN_QUEUE,         // Phase 3-A2: yielded lanes pushed onto ahs_queue_;
                    // slot stays here until CB_ACTION drains every
                    // cb_pending lane, then transitions to RESP.
  RESP,             // terminal status ready to emit
  EMITTED           // §8.6: TERMINAL sent; awaits free_slot()
};

struct LaneState {
  bool   active = false;
  bool   hit    = false;            // a *committed* hit (best so far)
  float  hit_t  = 0.f;
  float  hit_u  = 0.f;
  float  hit_v  = 0.f;
  uint32_t hit_prim = 0;
  // Vulkan gl_GeometryIndexEXT (slot 23). BVH4/6 leaves carry it in the
  // leaf header; the walker stashes the committed/candidate leaf's value
  // here so emit_completions / CB_YIELD can report it. Flat-list scenes
  // have no per-geometry split, so it stays 0 there.
  uint32_t hit_geometry  = 0;
  uint32_t cand_geometry = 0;
  // Phase 2/3-A2: candidate hit + yield state. When a non-opaque
  // triangle intersects, we stash its attrs here; the lane's
  // QueueEntry holds an index back into the slot so the CB_ACTION
  // drain can route commit/discard to (slot, lane).
  bool   cb_pending      = false;
  uint32_t cb_type       = 0;
  uint32_t sbt_idx       = 0;
  float  cand_t          = 0.f;
  float  cand_u          = 0.f;
  float  cand_v          = 0.f;
  uint32_t cand_prim     = 0;
  // P1 (proposal §4.2 slots 8..13): object-space ray captured at BLAS
  // entry. hit_obj_* is the committed hit's object ray (read by a CHS via
  // VX_RT_OBJECT_RAY_*); cand_obj_* is the yield candidate's object ray
  // (read by an AHS/IS). For top-level / TriList (no-instance) hits this
  // equals the world ray.
  float  hit_obj_o[3]    = {0.f, 0.f, 0.f};
  float  hit_obj_d[3]    = {0.f, 0.f, 0.f};
  float  cand_obj_o[3]   = {0.f, 0.f, 0.f};
  float  cand_obj_d[3]   = {0.f, 0.f, 0.f};
  // Phase 4 multi-line scene fetch:
  //   line 0 always carries the header. After parse, lines_needed grows
  //   to the per-scene byte budget. line_filled[i] / line_issued[i]
  //   track per-line state; the slot transitions to COMPUTE once every
  //   active lane reports lines_filled == lines_needed.
  std::array<bool,    kRtuMaxLinesPerLane> line_filled = {};
  std::array<bool,    kRtuMaxLinesPerLane> line_issued = {};
  std::array<std::array<uint8_t, VX_CFG_MEM_BLOCK_SIZE>, kRtuMaxLinesPerLane> line_data = {};
  uint32_t line_byte_off = 0;
  uint32_t lines_needed  = 1;
  uint32_t lines_filled  = 0;
  uint32_t lines_issued  = 0;
  uint32_t triangle_count = 0;
  uint32_t instance_count = 0;
  uint32_t hit_instance_id = 0;
  bool     header_parsed  = false;
  // Phase 4 (BVH4): byte offset of root node from scene-buffer base.
  uint32_t bvh_root_offset = 0;
};

struct Slot {
  bool      in_use = false;
  SlotState state  = SlotState::ISSUE;
  RtuReq    req;
  std::array<LaneState, VX_CFG_NUM_THREADS> lanes = {};
  uint32_t  pending_mem = 0;
  // §8.9 coherency gather: 3-bit octant signature.
  uint8_t   coh_signature = 0;
  // §8.7 SIMD-PE cycle accounting. The orchestrator walks the slot
  // once on its first tick in COMPUTE, accumulates the BoxPe/TriPe
  // cycle cost across all lanes' tests, stashes the post-compute
  // state (RESP or IN_QUEUE), then holds the slot in COMPUTE while
  // compute_cycles_remaining decrements per tick. When it reaches
  // 0 the slot advances to next_state_after_compute. Without §8.7
  // the walker effectively ran in zero cycles — every box/tri test
  // was free in SimX wall-clock. SystemC translation: maps to a
  // counter inside the BVH-walker SC_MODULE that stalls slot
  // advancement until the box/tri pipes have drained.
  uint32_t  compute_cycles_remaining = 0;
  bool      walk_done = false;
  SlotState next_state_after_compute = SlotState::RESP;
};

// Phase 3-A2 shader queue entry. One per yielded (slot, lane). The
// reformation pass groups entries by (warp_id, sbt_idx) and dispatches
// up to SIMD_WIDTH lanes per CB_YIELD.
struct QueueEntry {
  uint32_t slot_idx;
  uint32_t warp_id;
  uint8_t  lane;
  uint32_t sbt_idx;
  uint32_t cb_type;
  float    cand_t, cand_u, cand_v;
  uint32_t cand_prim;
  uint32_t cand_geometry;   // gl_GeometryIndexEXT of the candidate leaf
  // P1: candidate object-space ray (slots 8..13) carried to the CB_YIELD
  // so the AHS/IS dispatcher can read VX_RT_OBJECT_RAY_*.
  float    cand_obj_o[3];
  float    cand_obj_d[3];
};

// ════════════════════════════════════════════════════════════════════
// 5. Performance counters (surfaced via RtuCore::perf_stats())
// ════════════════════════════════════════════════════════════════════

struct PerfStats {
  // Phase 1 baseline.
  uint64_t rays_issued = 0;
  uint64_t rays_hit    = 0;
  uint64_t rays_miss   = 0;
  uint64_t mem_reads   = 0;
  // §8.9 BVH4 walker observability.
  uint64_t bvh_nodes_fetched     = 0;
  uint64_t bvh_leaves_fetched    = 0;
  uint64_t bvh_instance_descents = 0;
  uint64_t bvh_box_tests         = 0;
  uint64_t bvh_tri_tests         = 0;
  // §8.9 Callback-pipeline counters.
  uint64_t ahs_callbacks       = 0;
  uint64_t chs_callbacks       = 0;
  uint64_t miss_callbacks      = 0;
  uint64_t is_callbacks        = 0;
  uint64_t reformation_yields  = 0;
  // §8.9 Coherency gather.
  uint64_t coherency_hits      = 0;
  uint64_t coherency_misses    = 0;
  // §8.7 SIMD-PE cycle accounting. walker_cycles_total counts the
  // BoxPe + TriPe pipeline cycles charged across the lifetime of
  // every COMPUTE phase; pre-§8.7 SimX charged 0 (walker was
  // free). walker_busy_ticks counts ticks where at least one slot
  // was draining compute_cycles_remaining — gives an immediate
  // sense of how saturated the PEs were.
  uint64_t walker_cycles_total = 0;
  uint64_t walker_busy_ticks   = 0;

  PerfStats& operator+=(const PerfStats& rhs) {
    rays_issued            += rhs.rays_issued;
    rays_hit               += rhs.rays_hit;
    rays_miss              += rhs.rays_miss;
    mem_reads              += rhs.mem_reads;
    bvh_nodes_fetched      += rhs.bvh_nodes_fetched;
    bvh_leaves_fetched     += rhs.bvh_leaves_fetched;
    bvh_instance_descents  += rhs.bvh_instance_descents;
    bvh_box_tests          += rhs.bvh_box_tests;
    bvh_tri_tests          += rhs.bvh_tri_tests;
    ahs_callbacks          += rhs.ahs_callbacks;
    chs_callbacks          += rhs.chs_callbacks;
    miss_callbacks         += rhs.miss_callbacks;
    is_callbacks           += rhs.is_callbacks;
    reformation_yields     += rhs.reformation_yields;
    coherency_hits         += rhs.coherency_hits;
    coherency_misses       += rhs.coherency_misses;
    walker_cycles_total    += rhs.walker_cycles_total;
    walker_busy_ticks      += rhs.walker_busy_ticks;
    return *this;
  }
};

}}  // namespace vortex::rtu

// ════════════════════════════════════════════════════════════════════
// Back-compat re-exports — code outside vortex::rtu (cluster.cpp,
// sfu_unit.cpp, scheduler.cpp) historically uses vortex::RtuReq etc.
// Keep those names alive in the parent vortex:: namespace until a
// follow-up pass migrates all call sites to qualified vortex::rtu::*.
// ════════════════════════════════════════════════════════════════════

namespace vortex {
  using RtuReqKind    = ::vortex::rtu::RtuReqKind;
  using RtuRspKind    = ::vortex::rtu::RtuRspKind;
  using RtuReq        = ::vortex::rtu::RtuReq;
  using RtuRsp        = ::vortex::rtu::RtuRsp;
  using RtuBusArbiter = ::vortex::rtu::RtuBusArbiter;
}

#endif  // _VX_RTU_TYPES_H_
