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
//
// This file holds every POD type and constant that's needed across
// the RTU subsystem (bus packets, scene-format constants, per-lane /
// per-slot state structs, math primitives, perf counters). Higher
// layers (rtu_isect, rtu_classifier, rtu_walker, rtu_memory, rtu_unit,
// rtu_core) include this file but no other rtu_*.h cross-references.
//
// All names live in `vortex::rtu` for cross-namespace clarity. The
// top-level names (RtuReq, RtuRsp, RtuRspKind, ...) are re-exported
// via using-declarations into `vortex::` for code outside the RTU
// subsystem (cluster.cpp, sfu_unit.cpp).

#ifndef _VX_RTU_TYPES_H_
#define _VX_RTU_TYPES_H_

#include <array>
#include <cstddef>   // offsetof (flat<->BVH layout cross-check)
#include <cstdint>
#include <ostream>
#include <unordered_map>
#include "instr_trace.h"
#include "constants.h"
#include "types.h"
#include "rtu_bvh.h"  // VxBvhInstance — bound to the flat offsets below

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

  // Async pool: pre-allocated slot index for TRACE_NEW.
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

  // CB_ACTION only: the IS-computed hit distance, read back from the
  // kernel's VX_RT_HIT_T slot at vx_rt_cb_ret time. On ACCEPT of a
  // procedural (IS) candidate the RtuCore commits this t instead of the
  // pre-IS AABB-entry candidate t.
  std::array<float,    VX_CFG_NUM_THREADS> cb_hit_t   = {};
  // The intersection shader's hitAttribute, carried by the same verdict. Like
  // cb_hit_t it is committed only when the shader ACCEPTs.
  std::array<uint32_t, VX_CFG_NUM_THREADS> cb_attr    = {};

  // Per-lane RtuCore slot handle (CB_ACTION only) — read from the kernel's
  // VX_RT_CB_HANDLE slot at vx_rt_cb_ret time. Same-warp reformation may
  // batch lanes from MULTIPLE slots into one virtual warp at CB_YIELD, so
  // the action packet routes per-lane back to the originating slot rather
  // than rely on a single warp-scoped slot id.
  std::array<uint32_t, VX_CFG_NUM_THREADS> cb_handle  = {};

  // SimX-only: routing back to per-core SfuUnit writeback.
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
  std::array<uint32_t, VX_CFG_NUM_THREADS> hit_attr          = {};
  std::array<float,    VX_CFG_NUM_THREADS> hit_bary_u        = {};
  std::array<float,    VX_CFG_NUM_THREADS> hit_bary_v        = {};
  std::array<uint32_t, VX_CFG_NUM_THREADS> hit_primitive_id  = {};
  std::array<uint32_t, VX_CFG_NUM_THREADS> hit_instance_id   = {};
  std::array<uint32_t, VX_CFG_NUM_THREADS> hit_instance_custom = {};
  std::array<uint32_t, VX_CFG_NUM_THREADS> hit_geometry_index = {};

  // Object-space ray for the hit/candidate.
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
  // TERMINAL response carries the slot_idx so SfuUnit can look up
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

// One fetched cache line.
using LineBuf = std::array<uint8_t, VX_CFG_MEM_BLOCK_SIZE>;

// Max triangles per scene (flat-list walker cap).
constexpr uint32_t kRtuMaxTrisPerScene  = 8;

// Per-triangle stride 40 B = 9 floats (v0/v1/v2 xyz) + uint32 flags.
//   bit  0     = OPAQUE (clear → AHS yield)
//   bit  1     = PROCEDURAL (yield IS instead of AHS)
//   bits 8..15 = SBT_IDX (keys the kernel's switch(sbt_idx))
constexpr uint32_t kPhase2TriStride       = 40;
constexpr uint32_t kPhase2TriFlagsOff     = 36;
constexpr uint32_t kPhase2TriFlagOpaque   = 0x1u;
constexpr uint32_t kPhase2TriFlagProc     = 0x2u;
constexpr uint32_t kPhase2TriSbtIdxShift  = 8;
constexpr uint32_t kPhase2TriSbtIdxMask   = 0xffu;
constexpr uint32_t kRtuSceneHeaderBytes   = 16;

// Scene-kind tag (second uint32 of every scene header):
//   0 = TRI_LIST  — flat triangle scan
//   1 = TLAS      — flat 1-level TLAS over inline BLAS
//   2 = BVH4      — CW-BVH4 walker (see rtu_bvh.h)
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

// The context array: concurrent ray traversals. One context is one ray's live
// BVH walk — its stack, its fetch buffer, its hit record. Decoupled from the
// SIMD width, but never below it: a full-width trace must be able to bind all
// its lanes at once or it could never start (see the all-or-nothing rule in
// RtuCore::promote_slots).
static_assert(VX_CFG_RTU_NUM_CTX >= VX_CFG_NUM_THREADS,
              "RTU_NUM_CTX must cover a full-width trace");

// The slot pool: resident traces. A slot stages one trace's rays, its control
// state and its result record, from issue to terminal record. Slots partition
// statically across the cores one RTU serves, so a core with zero slots could
// never issue a trace at all.

// The MSHR file that would merge duplicate in-flight node fetches onto one
// entry. It is not implemented: a fetch is one request per context, with
// per-context tags. Refuse a non-zero depth rather than model a table no
// configuration can select, which is how the two descriptions drift apart.
static_assert(VX_CFG_RTU_MERGE_DEPTH == 0,
              "node-fetch merging is not implemented");

// Cores served by one RTU, and the per-core slot partition.
constexpr uint32_t kRtuCoresPerRtu = VX_CFG_SOCKET_SIZE / VX_CFG_NUM_RTU_CORES;
constexpr uint32_t kRtuSlotsPerCore = VX_CFG_RTU_NUM_SLOTS / kRtuCoresPerRtu;
static_assert(kRtuSlotsPerCore >= 1,
              "a core with no slot of its own could never issue a trace");

// The traversal front end is one shared machine — select a runnable context,
// then execute one step of it — serving the WHOLE context array, so the RTU
// advances one context by one FSM state at a time however many contexts it
// holds. That serialisation, not the PE arrays, is what a traversal step costs,
// and it is the ceiling every other knob is measured against.
//
// A step costs two cycles, except a step that reads the fetched node image: that
// one pays a third to stage the image out of the node RAM before it can be
// decoded. So a step's cost is kRtuPhasesPerStep, plus one if it reads the image.
constexpr uint32_t kRtuPhasesPerStep  = 2;
constexpr uint32_t kRtuPhasesImageAdd = 1;

// FSM states one traversal step costs, counted off the scheduler's own state
// list. An internal node runs REQ / RSP / DISPATCH / WAIT / PUSH / POP plus one
// FEED per child streamed to the box PE; a triangle leaf runs REQ / RSP /
// DISPATCH / POP plus a FEED and a WAIT per triangle; an instance descent runs
// the record fetch, the transform and the object-space reciprocal.
constexpr uint32_t kRtuStatesPerNode = 6;
constexpr uint32_t kRtuStatesPerBox  = 1;   // CS_FEED, one child per state
constexpr uint32_t kRtuStatesPerLeaf = 4;
constexpr uint32_t kRtuStatesPerTri  = 2;   // CS_TRI_FEED + CS_TRI_WAIT
constexpr uint32_t kRtuStatesPerInst = 6;
constexpr uint32_t kRtuStatesPerRay  = 3;   // CS_SETUP, CS_HDR_REQ, CS_HDR_WAIT

// Of those, the ones that decode the fetched image: the node/leaf response and
// its dispatch, the instance record, and the scene header. The box feed, the
// stack push/pop, the request states and the PE waits all run from state the
// select already latched, so they never pay the staging cycle.
constexpr uint32_t kRtuImageStatesPerNode = 2;   // RSP + DISPATCH
constexpr uint32_t kRtuImageStatesPerBox  = 0;
constexpr uint32_t kRtuImageStatesPerLeaf = 2;   // RSP + DISPATCH
constexpr uint32_t kRtuImageStatesPerTri  = 0;
constexpr uint32_t kRtuImageStatesPerInst = 1;   // instance record
constexpr uint32_t kRtuImageStatesPerRay  = 1;   // scene header

// Per-ray setup span waited before traversal: the reciprocal (1/dir) pipeline
// depth. Charged once per ray so the per-ray setup latency is accounted for
// alongside the box/tri PE cycles.
constexpr uint32_t kRtuSetupLatency = 17;   // reciprocal pipe depth
constexpr uint32_t kRtuFdivLat      = 17;   // reciprocal pipe depth
constexpr uint32_t kRtuLatencyFma   = 9;    // FMA pipe depth
// Per-instance transform latency = 4 * FMA pipe depth = 36: an (ro-t) subtract
// at FMA depth, then a 3-deep dot product. Charged per TLAS instance descent
// in the SimX cost model.
constexpr uint32_t kRtuXformLatency = 36;   // 4 * FMA pipe depth

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
//                     no-culling default. Bits 15..8 carry the instance
//                     flags byte (VkGeometryInstanceFlagBits); the low byte
//                     alone gates culling.
//   uint32 [60..64) = reserved
constexpr uint32_t kRtuInstanceStride       = 64;
constexpr uint32_t kRtuInstanceBlasOffOff   = 48;
constexpr uint32_t kRtuInstanceCustomIdOff  = 52;
constexpr uint32_t kRtuInstanceCullMaskOff  = 56;

// Layout guard: bind the flat-list TLAS byte offsets above to the CW-BVH
// `VxBvhInstance` struct (rtu_bvh.h) so the two 64 B instance-record layouts —
// selected at runtime by scene_kind (flat-list vs BVH4/6) — cannot silently
// diverge (a change to one layout would compile clean and only surface as a
// wrong-cull/instance-id bug on the other scene path).
//   Shared prefix [0..56): MUST match (same stride, blas-offset, custom-id).
static_assert(kRtuInstanceStride      == kVxBvhInstanceStride,   "flat/BVH instance stride diverged");
static_assert(kRtuInstanceBlasOffOff  == kVxBvhInstanceBlasOff,  "flat/BVH BLAS-offset diverged");
static_assert(kRtuInstanceCustomIdOff == kVxBvhInstanceCustomOff, "flat/BVH custom-id diverged");
//   Tail [56..64): INTENTIONALLY divergent — flat cull_mask@56 (no instance_id);
//   BVH instance_id@56 + cull_mask@60. Pin both ends so neither drifts unnoticed.
static_assert(kRtuInstanceCullMaskOff == 56, "flat TLAS cull_mask offset drifted");
static_assert(kVxBvhInstanceIdOff     == 56, "BVH instance_id offset drifted");
static_assert(kVxBvhInstanceCullOff   == 60, "BVH cull_mask offset drifted");

// VkGeometryInstanceFlagBits (low byte) packed into the reserved second byte
// (bits 15..8) of the cull_mask word — cull_mask uses only its low byte. Kept
// out of the cull-overlap test (which masks 0xff) and composed with the ray /
// per-tri classifier by classify_tri_hit.
constexpr uint32_t kRtuInstanceFlagsShift      = 8;    // within cull_mask word
constexpr uint32_t kRtuInstanceFlagsMask       = 0xffu;
constexpr uint32_t kRtuInstanceFlagTriCullDis  = 0x1u; // TRIANGLE_FACING_CULL_DISABLE
constexpr uint32_t kRtuInstanceFlagTriFlip     = 0x2u; // TRIANGLE_FLIP_FACING
constexpr uint32_t kRtuInstanceFlagForceOpaque = 0x4u; // FORCE_OPAQUE
constexpr uint32_t kRtuInstanceFlagForceNoOpq  = 0x8u; // FORCE_NO_OPAQUE

// Per-TLAS instance-count cap.
constexpr uint32_t kRtuMaxInstancesPerTlas = 4;

// The scene is fetched on demand — one line at a time, as the walk discovers
// it — so there is no whole-structure line budget to size. A malformed
// acceleration structure is bounded by the walker's node-visit ceiling instead.

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
  RESERVED,         // claimed at trace issue; its rays have not arrived yet
  READY,            // rays complete; waiting for contexts
  WALKING,          // contexts bound; the rays are traversing
  IN_QUEUE,         // yielded lanes queued for a callback; the slot stays here
                    // until CB_ACTION drains every cb_pending lane
  RESP,             // terminal record ready to emit
  EMITTED           // TERMINAL sent; awaits free_slot()
};

// One traversal context: the live BVH walk of a single ray. The scheduler binds
// contexts to a slot's active lanes at promote and releases them when the walk
// ends, so a 4-ray trace costs four contexts, not a warp's worth.
enum class CtxState : uint8_t {
  IDLE,             // unbound
  PROBE,            // has every byte it needs; step the walk
  PE,               // box / tri tests draining through the shared PEs
  REQ,              // wants a line; contends for the memory front end
  WAIT,             // its line is in flight
  DONE              // walk complete; its result is written back to the slot
};

struct LaneState {
  bool   active = false;
  bool   hit    = false;            // a *committed* hit (best so far)
  float  hit_t  = 0.f;
  uint32_t hit_attr = 0;   // the accepted candidate's hitAttribute
  float  hit_u  = 0.f;
  float  hit_v  = 0.f;
  uint32_t hit_prim = 0;
  // Vulkan gl_GeometryIndexEXT (slot 23). BVH4/6 leaves carry it in the
  // leaf header; the walker stashes the committed/candidate leaf's value
  // here so emit_completions / CB_YIELD can report it. Flat-list scenes
  // have no per-geometry split, so it stays 0 there.
  uint32_t hit_geometry  = 0;
  uint32_t cand_geometry = 0;
  uint32_t cand_instance = 0;
  uint32_t cand_custom   = 0;
  // Candidate hit + yield state. When a non-opaque
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
  // Object-space ray captured at BLAS entry. hit_obj_* is the
  // committed hit's object ray (read by a CHS via
  // VX_RT_OBJECT_RAY_*); cand_obj_* is the yield candidate's object ray
  // (read by an AHS/IS). For top-level / TriList (no-instance) hits this
  // equals the world ray.
  float  hit_obj_o[3]    = {0.f, 0.f, 0.f};
  float  hit_obj_d[3]    = {0.f, 0.f, 0.f};
  float  cand_obj_o[3]   = {0.f, 0.f, 0.f};
  float  cand_obj_d[3]   = {0.f, 0.f, 0.f};
  uint32_t hit_instance_id = 0;
  uint32_t hit_instance_custom = 0;
};

struct Slot {
  bool      in_use = false;
  SlotState state  = SlotState::RESERVED;
  RtuReq    req;
  std::array<LaneState, VX_CFG_NUM_THREADS> lanes = {};
  // The core that owns this slot: slots partition per core, so the pool cannot
  // hand one core's share to another.
  uint32_t  core_id = 0;
  // Promote order. The scheduler runs the oldest READY slot first and lets no
  // younger slot bypass it, or a wide trace would starve behind narrow ones.
  uint64_t  age = 0;
  // Rays of this slot still walking. The slot writes its record when the count
  // reaches zero, and each context releases itself the moment its own ray
  // retires rather than waiting for the slowest sibling.
  uint32_t  ctx_pending = 0;
  // Coherency gather: 3-bit octant signature.
  uint8_t   coh_signature = 0;
};

// The functional scene image a context walks through: the lines it has already
// pulled, keyed by address. A read of a line it does not hold is a miss — the
// walker reports it and stops, and the context fetches it. That is the whole
// demand-fetch model: the walk itself discovers the fetch stream.
struct SceneView {
  const std::unordered_map<uint64_t, LineBuf>* lines = nullptr;
  uint64_t base_line = 0;   // line holding the scene's first byte
  uint32_t byte_off  = 0;   // scene base within that line
  bool     miss      = false;
  uint64_t miss_line = 0;
};

struct Context {
  bool     valid = false;
  CtxState state = CtxState::IDLE;
  // Where the context goes once its PE work has drained: back to fetch the node
  // the walk stopped on, or out, its ray retired.
  CtxState next_state = CtxState::PROBE;
  uint32_t slot  = 0;
  uint32_t lane  = 0;
  // Lines this context has pulled. The hardware keeps one fetch buffer and re-reads
  // a node it revisits; the model keeps the bytes so the walk can be replayed
  // deterministically, and charges a fetch for every line the walk first
  // touches.
  std::unordered_map<uint64_t, LineBuf> lines;
  uint64_t base_line = 0;
  uint32_t byte_off  = 0;
  uint64_t req_addr  = 0;    // the line it wants (REQ / WAIT)
  // FSM states this step still owes the shared front end, then the PE pipeline
  // drain behind the last test it fed. img_states is the subset that decodes the
  // fetched image and so pays the extra staging cycle; it is retired first, which
  // does not change the total (the front end is serial and each state's cost is
  // independent of the order).
  uint32_t fsm_states = 0;
  uint32_t img_states = 0;
  uint32_t pe_lat     = 0;
  // Cumulative test counts already charged. A probe re-runs the walk from the
  // root, so the work of the newest step is the difference against these.
  uint64_t chg_box = 0, chg_tri = 0, chg_inst = 0, chg_restart = 0;
  uint64_t chg_nodes = 0, chg_leaves = 0;
};

// Shader queue entry. One per yielded (slot, lane). The
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
  uint32_t cand_instance;   // gl_InstanceID of the candidate
  uint32_t cand_custom;     // gl_InstanceCustomIndexEXT of the candidate
  // The hitAttribute already committed on this lane. An AHS/IS candidate has
  // none yet (its shader produces one), but a CHS fires on a committed hit and
  // must read the attribute the accepting shader wrote.
  uint32_t hit_attr;
  // Candidate object-space ray carried to the CB_YIELD so the AHS/IS
  // dispatcher can read VX_RT_OBJECT_RAY_*.
  float    cand_obj_o[3];
  float    cand_obj_d[3];
};

// ════════════════════════════════════════════════════════════════════
// 5. Performance counters (surfaced via RtuCore::perf_stats())
// ════════════════════════════════════════════════════════════════════

struct PerfStats {
  uint64_t rays_issued = 0;
  uint64_t rays_hit    = 0;
  uint64_t rays_miss   = 0;
  uint64_t mem_reads   = 0;
  // BVH4 walker observability.
  uint64_t bvh_nodes_fetched     = 0;
  uint64_t bvh_leaves_fetched    = 0;
  uint64_t bvh_instance_descents = 0;
  uint64_t bvh_box_tests         = 0;
  uint64_t bvh_tri_tests         = 0;
  // Short-stack overflow events: pushes past the VX_CFG_RTU_STACK_DEPTH HW
  // stack. SimX keeps an unbounded stack (never misses a hit); each overflow
  // entry is one the HW must re-descend for via trail-based restart, charged
  // in the cost model.
  uint64_t bvh_stack_restarts    = 0;
  // Callback-pipeline counters.
  uint64_t ahs_callbacks       = 0;
  uint64_t chs_callbacks       = 0;
  uint64_t miss_callbacks      = 0;
  uint64_t is_callbacks        = 0;
  uint64_t reformation_yields  = 0;
  // Coherency gather.
  uint64_t coherency_hits      = 0;
  uint64_t coherency_misses    = 0;
  // Ticks in which the RTU sent a memory request, and states the shared
  // select-align-execute front end retired. Three times the latter is the front
  // end's occupancy — the RTU's real throughput ceiling — and read against
  // bvh_box_tests it answers the question the merge stage hangs on: if the front
  // end is the bottleneck, collapsing duplicate fetches saves bandwidth, not
  // time.
  uint64_t mem_issue_ticks      = 0;
  uint64_t front_end_busy_ticks = 0;
  // Fetches the front end folded onto an already-in-flight line: the leader
  // compare catches contexts that go runnable together, the MSHR CAM catches
  // one that has drifted a cycle behind. mem_reads counts what actually went
  // out, so merged / (merged + mem_reads) is the duplicate rate.
  uint64_t fetches_merged      = 0;
  uint64_t mshr_full_stalls    = 0;
  // Context occupancy, summed per tick: bound counts contexts allocated to a
  // slot, active counts those still walking. bound - active is the tail — the
  // contexts sitting finished while their slot waits for its slowest ray.
  uint64_t ctx_bound_ticks     = 0;
  uint64_t ctx_active_ticks    = 0;
  uint64_t slot_busy_ticks     = 0;
  uint64_t rtu_busy_ticks      = 0;   // ticks with any slot in use
  // SIMD-PE cycle accounting: pipeline cycles charged across every context's
  // traversal, and ticks where at least one context was draining them.
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
    bvh_stack_restarts     += rhs.bvh_stack_restarts;
    bvh_tri_tests          += rhs.bvh_tri_tests;
    ahs_callbacks          += rhs.ahs_callbacks;
    chs_callbacks          += rhs.chs_callbacks;
    miss_callbacks         += rhs.miss_callbacks;
    is_callbacks           += rhs.is_callbacks;
    reformation_yields     += rhs.reformation_yields;
    coherency_hits         += rhs.coherency_hits;
    coherency_misses       += rhs.coherency_misses;
    front_end_busy_ticks   += rhs.front_end_busy_ticks;
    mem_issue_ticks        += rhs.mem_issue_ticks;
    fetches_merged         += rhs.fetches_merged;
    mshr_full_stalls       += rhs.mshr_full_stalls;
    ctx_bound_ticks        += rhs.ctx_bound_ticks;
    ctx_active_ticks       += rhs.ctx_active_ticks;
    slot_busy_ticks        += rhs.slot_busy_ticks;
    rtu_busy_ticks         += rhs.rtu_busy_ticks;
    walker_cycles_total    += rhs.walker_cycles_total;
    walker_busy_ticks      += rhs.walker_busy_ticks;
    return *this;
  }
};

}}  // namespace vortex::rtu

// ════════════════════════════════════════════════════════════════════
// Re-exports — code outside vortex::rtu (cluster.cpp, sfu_unit.cpp,
// scheduler.cpp) uses vortex::RtuReq etc. Keep those names alive in
// the parent vortex:: namespace.
// ════════════════════════════════════════════════════════════════════

namespace vortex {
  using RtuReqKind    = ::vortex::rtu::RtuReqKind;
  using RtuRspKind    = ::vortex::rtu::RtuRspKind;
  using RtuReq        = ::vortex::rtu::RtuReq;
  using RtuRsp        = ::vortex::rtu::RtuRsp;
  using RtuBusArbiter = ::vortex::rtu::RtuBusArbiter;
}

#endif  // _VX_RTU_TYPES_H_
