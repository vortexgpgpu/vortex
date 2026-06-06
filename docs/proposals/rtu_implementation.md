# PRISM RTU — Implementation & Refactor Document

**Status:** design review, awaiting approval before refactor lands.
**Target audience:** Vortex RTU contributors who will maintain SimX
today and translate to SystemC / SystemVerilog tomorrow.
**Scope:** professional OO decomposition of `sim/simx/rtu/` with file
naming, class responsibilities, header/cpp split, and SystemC mapping.
**Status note:** §8.6 async ray pool is *not yet implemented* — its
design is documented here (see §10) and will land in a follow-up commit.

---

## 1. Why refactor

Today's `sim/simx/rtu/` has grown to **2,275 lines across 5 files**,
of which **1,521 are in `rtu_core.cpp` alone**. That single file
contains:

- Scene-format constants (TRI_LIST, TLAS, BVH4 — three formats)
- Math primitives (`float3`, `dot`, `cross`, `ray_triangle`,
  `ray_aabb_intersect`, `reconstruct_child_aabb`, `affine_inverse_transform_ray`)
- The `RtuCore::Impl` god-class with **9 distinct concerns** mashed
  together: state-machine state (`State`, `LaneState`, `Slot`), slot
  pool, memory-fetch engine, flat-list walker, BVH4 walker, callback
  classifier, reformation queue, perf counters, debug printer.

Concretely, when adding a new feature today you touch this file in
3–5 different places per change, and the file no longer fits in a
single screen of context.

Future work makes this worse, not better:

- **§8.6** async ray pool — adds handle allocation, waiter tracking
- **§8.7** SIMD intersection coprocessors — pipelined box-PE / tri-PE
- **§8.10** private BVH cache — a new sub-unit
- **SystemC translation** — each "concern" should already be a class
  before it becomes a `SC_MODULE`
- **SystemVerilog translation** — each SystemC module is a candidate
  for one Verilog module; cross-module signals become RTL interfaces

This document proposes a refactor that:

1. Splits `rtu_core.cpp` into 8 focused files (≤300 LoC each)
2. Renames files to a consistent `rtu_*` prefix (drops `bvh_types.h`)
3. Establishes a stable layering: types ↓ math ↓ helpers ↓ engines ↓
   walkers ↓ scheduler ↓ unit ↓ top-level core
4. Maps cleanly onto SystemC modules (one class per `SC_MODULE`)
5. Preserves the existing test interface — **18/18 regression tests
   pass before and after** (the refactor is structural, not semantic)

---

## 2. Current layout (baseline)

```
sim/simx/rtu/
├── bvh_types.h     216 LoC — CW-BVH4 on-disk format
├── rtu_core.h      115 LoC — RtuCore SimObject + PerfStats
├── rtu_core.cpp   1521 LoC — everything else (see §1)
├── rtu_unit.h      208 LoC — RtuReq, RtuRsp, RtuUnit class
└── rtu_unit.cpp    215 LoC — RtuUnit methods
```

Problems summarized:

| Issue | Where |
|---|---|
| File naming inconsistent | `bvh_types.h` has no `rtu_` prefix |
| Scene-format constants split | rtu_core.cpp + bvh_types.h |
| Math helpers inline in core.cpp | float3, dot, cross, ray_triangle, ray_aabb_intersect, reconstruct_child_aabb, affine_inverse_transform_ray |
| Walkers conflated with scheduler | flat-list walker + BVH4 walker + slot scheduler all inside `RtuCore::Impl` |
| Mem-fetch state machine inlined | `issue_memory`, `drain_mem_rsp` |
| Callback classifier inlined | AHS/CHS/IS/MISS decision tree duplicated in flat-list and BVH4 paths |
| Reformation logic inlined | `ahs_queue_`, `warp_cb_inflight_`, `reformation_dispatch` |

---

## 3. Target layout

**13 files** organized in 5 layers. Each extraction defends against a
named, documented growth pressure from the §8 proposal roadmap.

```
sim/simx/rtu/
│
│   ─── Layer 1: pure types (header-only) ───────────────────────────
├── rtu_types.h        ~250  Common types: Req, Rsp, ReqKind, RspKind,
│                              SceneKind, PerfStats, Slot, LaneState,
│                              State enum, QueueEntry, math primitives
│                              (Vec3, Aabb, Ray + inline dot/cross/...)
├── rtu_bvh.h          ~200  CW-BVH4 on-disk format types + inline AABB
│                              reconstruction (renamed from bvh_types.h)
│
│   ─── Layer 2: ray-vs-primitive (§8.7 growth zone) ────────────────
├── rtu_isect.h        ~50   ray_triangle, ray_aabb_intersect,
│                              affine_inverse_transform_ray decls.
│                              §8.7 lands pipelined BoxPe/TriPe classes
│                              here.
├── rtu_isect.cpp      ~200  Scalar impls today; pipelined PE classes
│                              when §8.7 lands.
│
│   ─── Layer 3: hit policy (§8.8 growth zone) ──────────────────────
├── rtu_classifier.h   ~60   CallbackClassifier class decl.
├── rtu_classifier.cpp ~200  Per-tri opacity/cull/flag logic +
│                              finalise_lane CHS/MISS classifier.
│                              §8.8 + §11 ray-flag expansion lands here.
│
│   ─── Layer 4: scene traversal ────────────────────────────────────
├── rtu_walker.h       ~80   FlatWalker, Bvh4Walker class decls.
├── rtu_walker.cpp    ~400   Both walker classes. Uses rtu_isect
│                              + rtu_classifier; no policy logic here.
│
│   ─── Layer 5: per-cluster orchestration ──────────────────────────
├── rtu_memory.h       ~80   MemoryEngine class decl.
│                              §8.10 BvhCache class added here.
├── rtu_memory.cpp    ~250   issue_memory + drain_mem_rsp + line-fetch
│                              FSM. §8.10 BvhCache impl when it lands.
├── rtu_unit.h        ~220   RtuUnit (per-core SFU PE) + §8.6 wait_count
├── rtu_unit.cpp      ~280   RtuUnit methods.
├── rtu_core.h        ~140   RtuCore SimObject (public API).
└── rtu_core.cpp     ~250   RtuCore::Impl + nested SlotPool +
                              ReformationEngine. Tick() is 5 lines.

       Total: ~2500 LoC across 13 files (was 2275 across 5).
       Net growth ≈ 10% — mostly comment + API overhead.
       No new logic.
```

**Why 13 (vs 8 or 19):**

The 8-file shape was a sweet spot for *today*. The 13-file shape is a
sweet spot for *today + the next 6 months of §8 work*. Three named
growth vectors each get their own file:

| Future phase | New code volume | Without dedicated file | With dedicated file |
|---|---|---|---|
| §8.7 SIMD intersection PEs | +250 LoC | bloats rtu_walker.cpp 800→1050 | lands in rtu_isect.cpp |
| §8.8 ray-flag expansion + §11 OMM | +150 LoC | bloats walker classifier inline | lands in rtu_classifier.cpp |
| §8.10 private BVH cache | +200 LoC | bloats rtu_core.cpp | lands in rtu_memory.cpp |

Each extraction is *load-bearing*: it isolates a specific named change
the §8 proposal commits to. Speculative splits (walker-by-scene-kind,
SlotPool extraction, ReformationEngine extraction) were considered
and rejected as premature — they would extract things that *might*
grow without a concrete commit to point at. See §13 for the full
candidate analysis.

Today's bloated `rtu_core.cpp` shrinks from **1521 → ~250 LoC**.
Today's biggest single new file (`rtu_walker.cpp`) caps at **~400 LoC**
— well under the threshold. Every file in the 13-file layout fits on
~10 screens of code.

### 3.1 What goes where — concrete extraction map

| Today (rtu_core.cpp) | Tomorrow |
|---|---|
| `struct float3` + `dot_`/`cross_` | `rtu_types.h` (renamed to `Vec3`, `dot`, `cross`) |
| `inline tri_list_bytes / tlas_bytes / lines_for_bytes / lines_for_scene` | `rtu_types.h` (geometry helpers) |
| Scene-kind / tri-stride / instance-stride constants | `rtu_types.h` |
| `BvhInternalNode` / `BvhTri` / `BvhInstance` etc. | stay in `rtu_bvh.h` |
| `ray_triangle` / `ray_aabb_intersect` | **`rtu_isect.{h,cpp}`** |
| `affine_inverse_transform_ray` | **`rtu_isect.{h,cpp}`** |
| Per-tri opacity/cull/flag decision (currently duplicated in flat and BVH4 walkers) | **`rtu_classifier.{h,cpp}`** |
| AHS/CHS/MISS/IS queueing logic + ray-flag fast-out | **`rtu_classifier.{h,cpp}`** |
| `walk_bvh4_subtree` + `compute_intersections_bvh4_lane` | `Bvh4Walker` class in `rtu_walker.cpp` |
| Flat-list inner loop of `compute_intersections` | `FlatWalker` class in `rtu_walker.cpp` |
| `RtuCore::Impl::Slot` / `LaneState` / `State` enum | `rtu_types.h` |
| `RtuCore::Impl::QueueEntry` | `rtu_types.h` |
| `issue_memory` / `drain_mem_rsp` / line-fetch FSM | **`rtu_memory.{h,cpp}`** (MemoryEngine class) |
| `drain_requests` / slot allocator | `SlotPool` nested class in `rtu_core.cpp` |
| `ahs_queue_` / `warp_cb_inflight_` / `reformation_dispatch` | `ReformationEngine` nested class in `rtu_core.cpp` |
| Octant signature + 2-pass coherency gather | stays in `RtuCore::Impl` (top-level scheduling) |
| Perf counters | `rtu_types.h` (PerfStats) |
| RtuCore destructor's VX_RTU_STATS dump | stays in `rtu_core.cpp` |

**Bold = file extraction** (vs 8-file baseline). Bold rows are the
three load-bearing extractions defending against §8.7, §8.8/§11, and
§8.10 growth respectively.

---

## 4. Layering invariant

Each layer may depend only on **lower-numbered** layers:

```
       Layer 5: rtu_core         (RtuCore + SlotPool + ReformationEngine)
                              │
                              ├── rtu_unit  (per-core SFU PE)
                              │
                              └── rtu_memory (MemoryEngine + BvhCache)
                              │
                              ▼
       Layer 4: rtu_walker       (FlatWalker + Bvh4Walker)
                              │
                              ▼
       Layer 3: rtu_classifier   (CallbackClassifier: hit policy)
                              │
                              ▼
       Layer 2: rtu_isect        (ray_triangle, ray_aabb, transform)
                              │
                              ▼
       Layer 1: rtu_types + rtu_bvh   (POD types)
```

Key invariants:
- Layer 4 (walker) calls layer 3 (classifier) and layer 2 (isect) but
  knows nothing about layer 5 (memory, slot pool).
- Layer 5 modules (memory, unit, core) are *peers* — they cooperate
  through layer-1 types (Req, Rsp, Slot) but don't include each other
  except via the `RtuCore` top-level glue.
- Layer 2 (isect) and layer 3 (classifier) are stateless modules —
  pure compute. They become combinational logic in RTL.

This makes the SystemC port story trivial: each layer-3+ class is one
`SC_MODULE`. Layer 1-2 stays header-only or compiles into the
modules.

---

## 5. Layer 1 — types

### 5.1 `rtu_types.h`

Common, simulator-independent types. Header-only. Holds **everything**
that's plain data (no methods, or only trivial inlines).

```cpp
namespace vortex::rtu {

//
// Bus packets
//
enum class ReqKind : uint8_t { TRACE_NEW = 0, CB_ACTION = 1 };
enum class RspKind : uint8_t { TERMINAL  = 0, CB_YIELD  = 1 };
struct Req { /* per-lane ray descriptor + slot_idx + sim routing */ };
struct Rsp { /* per-lane status + hit attrs + slot_idx */ };

//
// Scene formats
//
enum class SceneKind : uint32_t { TriList = 0, Tlas = 1, Bvh4 = 2 };
constexpr uint32_t kSceneHeaderBytes = 16;
constexpr uint32_t kTriStride        = 40;
constexpr uint32_t kInstanceStride   = 64;
// ... other scene-format constants

//
// Math primitives — inline, header-only
//
struct Vec3 { float x, y, z; /* + ops */ };
struct Aabb { Vec3 mn, mx; };
struct Ray  { Vec3 o, d; float tmin, tmax; uint32_t flags; };
inline float dot(Vec3 a, Vec3 b);
inline Vec3  cross(Vec3 a, Vec3 b);

//
// Slot pool state machine (consumed by rtu_core's SlotPool + by
// walkers' LaneState reads)
//
enum class SlotState : uint8_t {
  RESERVED, ISSUE, AWAIT, COMPUTE, IN_QUEUE, RESP, EMITTED,
};
struct LaneState { /* line buffer + hit + candidate + scene-kind */ };
struct Slot      { Req req; std::array<LaneState, NUM_THREADS> lanes; ... };

//
// Reformation queue entry
//
struct QueueEntry { uint32_t slot_idx, warp_id, sbt_idx, cb_type; ... };

//
// Performance counters
//
struct PerfStats { /* same 16 fields as today */ };

}  // namespace vortex::rtu
```

Notes:
- POD/aggregate-only — no virtuals, no heap allocs, no constructors
  beyond trivial defaults.
- The `instr_trace_t*` / `block_id` fields on `Req`/`Rsp` are sim-only
  routing metadata; in RTL the response auto-flows via the bus
  arbiter's stored route. Group them under `#ifdef VX_SIM`.
- Math primitives live here (not in a separate `rtu_math.h`) because
  they're tiny and pervasive — splitting them adds a file without
  improving anything.

### 5.2 `rtu_bvh.h` (renamed from `bvh_types.h`)

CW-BVH4 on-disk format types + inline AABB reconstruction. Header-only.

```cpp
namespace vortex::rtu {

constexpr uint32_t kBvhWidth          = 4;
constexpr uint32_t kBvhLeafHeaderBytes = 16;
// ... format constants

enum class BvhKind : uint8_t { Internal=0, LeafTri=1, LeafInst=2, LeafProc=3 };

struct BvhInternalNode { /* 64 B */ };
struct BvhLeafHeader   { /* 16 B */ };
struct BvhTri          { /* 40 B */ };
struct BvhInstance     { /* 64 B */ };
struct BvhProcAabb     { /* 24 B */ };
struct BvhSceneHeader  { /* 16 B */ };

// Inline reconstruction (small enough to live in the header).
inline Aabb reconstruct_child_aabb(const float origin[3], const int8_t exp[3],
                                   const uint8_t qmin[3], const uint8_t qmax[3]);

}  // namespace vortex::rtu
```

**`rtu_bvh.cpp`?** Not yet — every helper here is small enough to
inline. Add `.cpp` later when a Phase-4-late host-side BVH builder or
a serializer lands and pulls in non-trivial code.

---

## 6. Layer 2 — `rtu_isect.{h,cpp}`

Ray-vs-primitive intersection and ray transform. Stateless pure
functions today; future home of pipelined-PE classes when §8.7 lands.

### 6.1 `rtu_isect.h`

```cpp
namespace vortex::rtu {

// Möller-Trumbore. out_back_facing is set true when the ray hit the
// triangle's back face (det < 0); used by §8.8 face culling.
bool ray_triangle(const Ray& r, Vec3 v0, Vec3 v1, Vec3 v2,
                  float& out_t, float& out_u, float& out_v,
                  bool& out_back_facing);

// Ray vs AABB slab test. out_t_near is the entry parameter (clamped
// to ray.tmin) — used by the BVH4 walker for nearest-child ordering.
bool ray_aabb(const Ray& r, Aabb box, float& out_t_near);

// Apply inverse of a 3x4 row-major affine to a ray.
void affine_inverse_transform_ray(const float xform[12],
                                  const Vec3& world_o, const Vec3& world_d,
                                  Vec3& obj_o, Vec3& obj_d);

// §8.7 future: pipelined PE classes land here.
// class BoxPe { void issue(...); bool ready(); ... };
// class TriPe { void issue(...); bool ready(); ... };

}  // namespace vortex::rtu
```

### 6.2 `rtu_isect.cpp`

Plain implementations of the three functions today (~200 LoC). When
§8.7 lands, the pipelined `BoxPe`/`TriPe` classes are added here —
the scalar functions stay as the underlying compute, with pipelined
latency wrappers on top.

**SystemC mapping:** the scalar functions are inline expressions. The
future pipelined PEs become `SC_MODULE(BoxPe)`/`SC_MODULE(TriPe)`
with `SC_METHOD` driven by clock.

---

## 7. Layer 3 — `rtu_classifier.{h,cpp}`

Hit policy. Given a hit was found, decide what to do with it. The
walker tells the classifier "ray hit triangle T at t=5.3, OPAQUE-
flagged, back-facing"; the classifier returns "commit" or "yield AHS"
or "ignore (culled by ray flag)" based on the full Vulkan ray-flag +
opacity ruleset.

### 7.1 `rtu_classifier.h`

```cpp
namespace vortex::rtu {

class CallbackClassifier {
public:
  enum class TriDecision : uint8_t { Ignore, Commit, Yield };

  // Apply ray-flag fast-out + per-tri flag overrides. Called per-tri
  // inside walker leaf-visit. Mutates LaneState (commit hit or stage
  // yield candidate). §8.8 ray-flag handling lives here.
  TriDecision classify_tri(LaneState& l, float t_hit, float u, float v,
                           uint32_t tri_flags, uint32_t ray_flags,
                           bool back_facing, uint32_t instance_id);

  // Called per-lane after walker finishes. Emits CHS / MISS callbacks
  // based on the lane's final hit state + ray flags. Returns true if
  // a callback was queued.
  bool finalise_lane(LaneState& l, const Slot& s, uint32_t lane,
                     class ReformationEngine& reform);
};

}  // namespace vortex::rtu
```

### 7.2 `rtu_classifier.cpp`

The per-tri opacity / cull / face-culling decision tree (~150 LoC
today; ~200 LoC after §8.8 + §11 OMM additions). The CHS/MISS
finalise logic (~50 LoC).

**SystemC mapping:** stateless → combinational logic block.
`SC_MODULE(CallbackClassifier)` with `SC_METHOD` driven by every hit.

---

## 8. Layer 4 — `rtu_walker.{h,cpp}`

Both walkers in one file. They call rtu_isect for primitive tests and
rtu_classifier for hit policy. The file contains ONLY walk-mechanics
code — no intersection math, no ray-flag logic.

### 8.1 `rtu_walker.h`

```cpp
namespace vortex::rtu {

class FlatWalker {
public:
  explicit FlatWalker(CallbackClassifier& c);
  bool walk_lane(Slot& s, uint32_t lane, class ReformationEngine& reform);
private:
  CallbackClassifier& classifier_;
};

class Bvh4Walker {
public:
  explicit Bvh4Walker(CallbackClassifier& c);
  bool walk_lane(Slot& s, uint32_t lane, class ReformationEngine& reform);
private:
  CallbackClassifier& classifier_;
};

}  // namespace vortex::rtu
```

### 8.2 `rtu_walker.cpp`

```cpp
namespace { // file-local helpers

// Shared walker context (scoreboard of best_t, yield candidate, etc.)
struct WalkCtx { /* moved from RtuCore::Impl::Bvh4WalkCtx today */ };

} // namespace

namespace vortex::rtu {

bool FlatWalker::walk_lane(Slot&, uint32_t, ReformationEngine&) { ... }
bool Bvh4Walker::walk_lane(Slot&, uint32_t, ReformationEngine&) { ... }

}  // namespace vortex::rtu
```

~400 LoC. No policy logic; no intersection math. Both walkers
sequence reads from the per-lane line buffer, call `rtu_isect` for
each primitive test, and feed results into `rtu_classifier`.

**SystemC mapping:** each walker becomes one `SC_MODULE` with a
traversal FSM. The shared `WalkCtx` becomes per-module state.

---

## 9. Layer 5 — `rtu_memory.{h,cpp}`, `rtu_unit.{h,cpp}`, `rtu_core.{h,cpp}`

Three peer modules that cooperate per-cluster via the SimX channels
and per-layer-1 types.

### 9.1 `rtu_memory.{h,cpp}`

The cluster-side memory-fetch engine. Owns the per-line fetch FSM and
the pending-mem tag table. Future home of the BvhCache class when
§8.10 lands.

```cpp
namespace vortex::rtu {

class MemoryEngine {
public:
  MemoryEngine(SlotPool& pool,
               std::vector<SimChannel<MemReq>>& dcache_req,
               std::vector<SimChannel<MemRsp>>& dcache_rsp);
  void tick();   // issue_memory + drain_mem_rsp in one tick
  uint64_t mem_reads() const;
private:
  void issue_memory();
  void drain_mem_rsp();
  struct PendingFill { uint32_t slot_idx; uint8_t lane; uint8_t line_idx; };
  std::unordered_map<uint32_t, PendingFill> pending_;
  uint32_t next_tag_;
  SlotPool& pool_;
  // §8.10 future: BvhCache cache_;
};

}  // namespace vortex::rtu
```

`SlotPool` is passed by reference at construction (defined as a
nested class in `rtu_core.cpp`). The memory engine reads slot lane
state to know what to fetch and where to put the response.

**SystemC mapping:** `SC_MODULE(MemoryEngine)` per cluster. The
future `BvhCache` becomes a nested `SC_MODULE` or its own peer.

### 9.2 `rtu_unit.{h,cpp}` (unchanged shape)

Existing files. Only changes from today:
1. New `inc_wait_count` / `dec_wait_count` / `wait_count` methods for
   §8.6 (barrier-style wait tracking).
2. `process_trace` calls `RtuCore::allocate_slot()` (§8.6).
3. `process_get` back-pressures when `wait_count > 0` (§8.6).

The class itself (per-core SFU PE, per-(warp,lane) regfile) is
already well-factored. No further split needed.

**SystemC mapping:** `SC_MODULE(RtuUnit)` per core.

### 9.3 `rtu_core.h` (unchanged public API)

```cpp
class RtuCore : public SimObject<RtuCore> {
public:
  using Ptr = std::shared_ptr<RtuCore>;
  // ... ports + lifecycle
  const PerfStats& perf_stats() const;
  int32_t allocate_slot();        // §8.6
  void    free_slot(uint32_t);    // §8.6
};
```

### 9.4 `rtu_core.cpp` — slim orchestrator + 2 nested classes

The big shrinkage. ~250 LoC structured as:

```cpp
namespace { // file-local

// SlotPool — manages the VX_CFG_RTU_CONTEXT_POOL Slot[] array,
// allocate()/free(), tick-level RESERVED→ISSUE drain.
class SlotPool {
public:
  SlotPool(uint32_t size);
  int32_t allocate();
  void    free(uint32_t idx);
  Slot&   at(uint32_t idx);
  template<typename Fn> void for_each(Fn);
private:
  std::vector<Slot> slots_;
};

// ReformationEngine — ahs_queue, warp_cb_inflight, batched CB_YIELD
// emit. Octant-signature gather lives here.
class ReformationEngine {
public:
  ReformationEngine(SimChannel<Rsp>& rsp_out);
  void push(const QueueEntry& e);
  void tick(PerfStats& perf);
  void warp_cb_clear(uint32_t wid);
private:
  std::deque<QueueEntry> queue_;
};

} // namespace

class RtuCore::Impl {
public:
  Impl(RtuCore* obj);
  void reset();
  void tick() {
    mem_.tick();
    drain_requests();
    compute_step();          // per slot, per lane → walker dispatch
    reform_.tick(perf_);
    emit_completions();
  }
  int32_t allocate_slot() { return pool_.allocate(); }
  void    free_slot(uint32_t i) { pool_.free(i); }
private:
  SlotPool             pool_;
  MemoryEngine         mem_;       // from rtu_memory.h
  CallbackClassifier   classifier_;// from rtu_classifier.h
  FlatWalker           flat_walker_;
  Bvh4Walker           bvh4_walker_;
  ReformationEngine    reform_;
  PerfStats            perf_;
  uint8_t              last_compute_signature_ = 0;  // §8.9 coherency
};
```

The whole `tick()` reads in 5 lines. The orchestration intent is
obvious; the heavy lifting is in the engines.

**SystemC mapping:** SlotPool, ReformationEngine each become an
`SC_MODULE` (or stay nested in `SC_MODULE(RtuCore)` as
sub-modules).

---

## 10. §8.6 async ray pool — Design F (shipped)

§8.6 landed as commit `b83e44b4` on top of the layer refactor.
The chosen design is **Design F (scoreboard-only)**: no scheduler
suspend/resume, only a parked WAIT trace + a new `vx_rt_get_after`
intrinsic that adds an explicit scoreboard dep onto WAIT's rd.
Design E (barrier-style suspend) is documented in §10.5 as the
alternative we evaluated and discarded.

### 10.1 The blocker the refactor removed

The pre-§8.6 `vx_rt_wait` / `vx_rt_get` race was masked by the
"`vx_rt_trace.rd` holds status until TERMINAL" trick: the kernel's
`uint32_t h = vx_rt_trace(scene)` waited on the writeback because
TRACE's writeback only fired at TERMINAL drain time. That collapsed
trace + wait into a single round-trip but precluded N rays in
flight at once.

The refactor isolated the WAIT/TERMINAL flow inside `RtuUnit` and
`SfuUnit`, so §8.6's new wait_parked_ + pending_terminals_ tables
landed in one file without cross-cutting touches. Same story for
the new `RtuCore::allocate_slot()`/`free_slot()` API — it lives
behind the `SlotPool` boundary established in refactor step 7.

### 10.2 Design F — scoreboard-only

The mechanism is two-sided:

- **TRACE** is now synchronous from the kernel's POV. `RtuUnit`
  calls `RtuCore::allocate_slot()` to reserve a slot (state
  `RESERVED`, invisible to the FSM), stamps `req.slot_idx`, and
  the SFU writes the slot index back as the handle via the
  standard 4-cycle writeback. The ray walks async in RtuCore.

- **WAIT** parks. `RtuUnit::process_wait` returns `nullptr` when
  the slot's TERMINAL hasn't landed yet — the trace goes into
  `wait_parked_[wid][slot]` and the SFU does NOT call
  `output.send`, so the scoreboard keeps WAIT's rd reserved
  indefinitely. When the TERMINAL rsp arrives, the SFU's drain
  calls `RtuUnit::on_terminal_rsp`, which finds the parked WAIT,
  applies the rsp to the regfile, `output.sends` the WAIT trace
  (this releases the scoreboard reservation 4 cycles later), and
  calls `RtuCore::free_slot()`.

- **Fast path**: if the TERMINAL beats the WAIT to the SFU (short
  ray), `on_terminal_rsp` latches the rsp into
  `pending_terminals_[wid][slot]` instead of dropping it. The
  next `process_wait(handle)` short-circuits: pop the latched
  rsp, apply, write status, free, return the trace for immediate
  writeback.

Both `wait_parked_` and `pending_terminals_` are keyed by
`(warp_id, slot)`; exactly one of them holds an entry for any
in-flight ray at any time. They're per-core because the SFU is
per-core, not because the slot itself is — slots themselves are
cluster-scope in the `SlotPool`.

### 10.3 `vx_rt_get_after` — the post-mret race fix

Parking WAIT keeps WAIT's rd reserved, so `vx_rt_get_after(slot,
wait_status)` (with `rs1 = wait_status`) stalls at scoreboard until
TERMINAL drains. Ordinary `vx_rt_get` (rs1 = x0) does NOT stall —
fine inside trap-context dispatchers where the regfile is already
populated via `apply_callback_payload`, but wrong in post-WAIT
kernel code where the regfile hasn't been touched by `apply_response`
yet.

```c
// New macro in vx_raytrace.h:
#define vx_rt_get_after(slot, wait_status) ({                            \
  uint32_t __v;                                                          \
  __asm__ volatile (".insn r %1, 5, %2, %0, %3, x0"                      \
      : "=r"(__v)                                                        \
      : "i"(RISCV_CUSTOM1), "i"(((slot) << 2) | 1), "r"(wait_status));   \
  __v;                                                                   \
})
```

The decoder change is one line: GET now always registers `rs1` as a
src reg (was unconditionally `x0` before). `vx_rt_get` users are
unaffected because reads of `x0` never see a reservation (`x0` is
never a writeback target).

The critical post-mret case: CB_YIELD raises an async trap on the
WAIT-parked warp. The dispatcher runs at `mtvec`, calls
`vx_rt_get` (regular, no `_after`) to read candidate hit attrs
(populated by `apply_callback_payload`), decides ACCEPT/IGNORE/
TERMINATE, calls `vx_rt_cb_ret`, then `mret`. The kernel resumes
at the post-WAIT PC. If the next op is `vx_rt_get_after`, it stalls
on the still-reserved WAIT rd → no race against the eventual
TERMINAL.

### 10.4 Why we did NOT need scheduler suspend

The original sketch (Design E) wanted `vx_rt_wait` to call
`scheduler_->suspend(wid)`, mirroring `BarrierUnit::wait`. That
introduces a structural constraint — the warp can't fetch past
WAIT — that the prior attempt had to immediately violate to make
CB_YIELD dispatchers runnable: resume the warp on CB_YIELD,
re-suspend after `mret`. That round-trip is what deadlocked in
the previous session (an `mret` detection hook is non-trivial).

Design F sidesteps the entire suspend dance. The scoreboard alone
enforces "no post-WAIT consumer of rd dispatches until WAIT
writes back," which is exactly what the kernel idiom requires.
Independent post-WAIT ops (those not depending on WAIT's rd) are
allowed to make progress in the pipeline shadow of WAIT — a small
parallelism win that the suspend design forbids.

Trade-off accepted: the kernel writer MUST use `vx_rt_get_after`
(not `vx_rt_get`) for post-WAIT reads. This is documented in the
header and applied uniformly across all 18 existing smoke tests.

### 10.5 Design E (rejected) — scheduler suspend/resume

For the record:

- `vx_rt_wait` → `scheduler_->suspend(wid)`.
- TERMINAL drain → `scheduler_->resume(wid)`.
- CB_YIELD drain → `scheduler_->resume(wid)` (so dispatcher runs)
  BEFORE `raise_async_trap`.
- mret detection (per-tick check in `SfuUnit::on_tick`) →
  re-suspend if the warp still has a parked WAIT.

Discarded for: (a) the mret-detection hook is fragile, (b) the
resume/re-suspend window is exactly where the prior session's
deadlock lived, (c) Design F is strictly simpler and ships
equivalent semantics. Kept in the doc as the path to take ONLY
if `vx_rt_get_after` migration proves infeasible for some
downstream callee (e.g. compiler-generated code that can't be
made to thread `wait_status`).

### 10.6 Shipped scope

- `vx_raytrace.h`: +1 macro (`vx_rt_get_after`) + float helper
  (`vx_rt_get_f_imm_after`).
- 18 existing test kernels: 1-line edit each (`vx_rt_get` →
  `vx_rt_get_after`, threading the WAIT's rd as `wait_status`).
- New test `rtu_smoke_async_batch`: 4 traces in flight, sequential
  waits, validates per-handle hit_t (catches any handle ↔ slot
  crosstalk).
- `RtuCore::allocate_slot()` / `free_slot()` public API.
- `RtuCore::Impl::drain_requests` consumes `req.slot_idx` instead
  of calling `pool_.allocate()` inline.
- `RtuCore::Impl::emit_completions` sets state to `EMITTED` (not
  `in_use = false`); free happens on WAIT.
- `RtuUnit::{process_trace, process_wait, on_terminal_rsp,
  wait_handle, wait_would_short_circuit, terminal_would_writeback}`.
- `SfuUnit`: TRACE branch = synchronous writeback; WAIT branch =
  conditional writeback; TERMINAL drain = `on_terminal_rsp`;
  `set_rtu_core()` hookup called from `Cluster`.
- `decode.cpp`: GET registers `rs1` unconditionally.

Total: ~670 LoC + 17 kernel-line edits + 1 new test = ~770 LoC
across 32 files. 19/19 RTU smoke tests pass on simx.

---

## 11. SystemC / SystemVerilog mapping

Each layer-2+ **class** (not file) maps 1:1 to a `SC_MODULE` and
(eventually) to a Verilog `module`. When the SystemC port happens,
each class becomes its own module file; for SimX today, related
classes can group per file.

| C++ class             | SimX file          | SystemC module          | Verilog module       |
|---|---|---|---|
| `ray_triangle` / `ray_aabb` | rtu_isect.cpp | (function calls today)  | (inlined in PEs)     |
| `BoxPe` / `TriPe` (§8.7) | rtu_isect.cpp | `SC_MODULE(BoxPe)` / `SC_MODULE(TriPe)` | `rtu_box_pe` / `rtu_tri_pe` |
| `CallbackClassifier`  | rtu_classifier.cpp | `SC_MODULE(CbClassifier)` (combinational) | `rtu_cb_classify` |
| `FlatWalker`          | rtu_walker.cpp     | `SC_MODULE(FlatWalker)` | `rtu_walker_flat`    |
| `Bvh4Walker`          | rtu_walker.cpp     | `SC_MODULE(BvhWalker)`  | `rtu_walker_bvh4`    |
| `MemoryEngine`        | rtu_memory.cpp     | `SC_MODULE(MemEngine)`  | `rtu_mem_engine`     |
| `BvhCache` (§8.10)    | rtu_memory.cpp     | `SC_MODULE(BvhCache)`   | `rtu_bvh_cache`      |
| `RtuUnit`             | rtu_unit.cpp       | `SC_MODULE(RtuUnit)`    | `rtu_unit`           |
| `SlotPool`            | rtu_core.cpp       | `SC_MODULE(SlotPool)`   | `rtu_slot_pool`      |
| `ReformationEngine`   | rtu_core.cpp       | `SC_MODULE(Reform)`     | `rtu_reform`         |
| `RtuCore`             | rtu_core.cpp       | `SC_MODULE(RtuCore)`    | `rtu_core_top`       |

Communication patterns:
- Today's `SimChannel<T>` → SystemC `sc_fifo<T>` → Verilog
  valid/ready interface.
- Plain method calls (e.g. `pool.allocate()`) → SystemC
  `sc_export`/`sc_port` → Verilog single-cycle handshake.

Strict layering (§4) means a single module can be translated to
SystemVerilog (e.g. `rtu_walker_bvh4`) while the rest stays in SimX
C++ via SystemC co-simulation — incremental migration.

---

## 12. Migration plan

The refactor is structural; the algorithm doesn't change. Acceptance
is **18/18 RTU regression tests pass at every step**. Each step is a
single commit, independently reversible. Each step keeps the rest of
the codebase buildable.

| Step | Work | Output | Validation |
|---|---|---|---|
| 1 | Rename `bvh_types.h` → `rtu_bvh.h`; namespace as `vortex::rtu::`; update includes | rename + ~5 #include edits | 18/18 |
| 2 | Create `rtu_types.h`; move `Req`/`Rsp`/`ReqKind`/`RspKind` from `rtu_unit.h`; move math (`float3`→`Vec3`/`dot`/`cross`) + scene constants + `Slot`/`LaneState`/`State` enum + `QueueEntry` + `PerfStats` from `rtu_core.cpp` | new file + ~10 edits | 18/18 |
| 3 | Create `rtu_isect.{h,cpp}`; move `ray_triangle`/`ray_aabb`/`affine_inverse_transform_ray` from `rtu_core.cpp` | new files + ~5 edits | 18/18 |
| 4 | Create `rtu_classifier.{h,cpp}`; extract per-tri opacity/cull/flag logic + CHS/MISS finalise logic from the two walkers in `rtu_core.cpp`; deduplicate the inlined classifier code | new files + walker dedup | 18/18 |
| 5 | Create `rtu_walker.{h,cpp}`; move `FlatWalker`/`Bvh4Walker` classes from `rtu_core.cpp`; they now call rtu_isect + rtu_classifier | new files + ~500 LoC moved | 18/18 |
| 6 | Create `rtu_memory.{h,cpp}`; extract `MemoryEngine` (`issue_memory`/`drain_mem_rsp`/line-fetch FSM) from `rtu_core.cpp` | new files + ~250 LoC moved | 18/18 |
| 7 | Slim `rtu_core.cpp`: extract `SlotPool` + `ReformationEngine` as anonymous-namespace classes; `RtuCore::Impl::tick()` becomes the 5-line orchestrator | edit core (1521 → ~250 LoC) | 18/18 |

**7 commits total.** The refactor lands *before* §8.6, so §8.6's new
code (handle ABI, wait-suspend, `vx_rt_get_after`) goes into the
right files from day 1.

---

## 13. Resolved design questions

1. **Namespace policy** — adopt `vortex::rtu::` sub-namespace to
   scope `Req`/`Rsp`/`Vec3`/etc. without collision with the broader
   `vortex::` namespace.

2. **`rtu_bvh.cpp`?** No for now. Every BVH helper (including
   `reconstruct_child_aabb`) is small enough to stay inline in
   `rtu_bvh.h`. Promote to `.cpp` when an on-device BVH builder
   (§8.11) or `vk_bvh.h` bit-compatible decoder (Phase 4-late) lands.

3. **`LaneState` granularity** — stays as one big struct in
   `rtu_types.h`. The line buffer + hit state + candidate state +
   BVH state are all logically per-lane and read in close proximity
   during walker iteration. Splitting for RTL (separate BRAMs) is a
   wrapper problem at translation time, not a SimX problem.

4. **`RtuUnit` placement** — stays per-core (SfuUnit owns it).
   Doesn't move into `SlotPool` (which is per-cluster). They
   communicate only via the bus channels.

### 13.1 Why we stopped at 13 files

The review considered 4 more candidates beyond the C layout
(13 → 14 files). All four were rejected. Diagnostic: "Name a future
commit whose diff would shrink by ≥40% if this file existed."

| Candidate 14th file | Justification | Verdict |
|---|---|---|
| Split walkers into `rtu_walker_flat.cpp` + `rtu_walker_bvh4.cpp` | Flat walker will eventually retire when CW-BVH4 covers all workloads | **Rejected** — flat is currently required by 13 of 18 tests; not legacy yet |
| Extract `rtu_reform.{h,cpp}` (ReformationEngine) | Phase 3-B async drain + NVIDIA SER would land here | **Rejected** — Phase 3-B is explicitly data-driven-deferred; no concrete commit to point at |
| Extract `rtu_pool.{h,cpp}` (SlotPool) | Defines `Slot` lifecycle; would enable independent unit tests | **Rejected** — strongest candidate, but no test plan exists; SlotPool size won't grow with §8 work |
| Extract `rtu_stats.{h,cpp}` (PerfStats + VX_RTU_STATS dump) | Counters will grow with each phase | **Rejected** — PerfStats is pure POD struct; new counters are 1-line additions; no behavioral growth |

The thirteen extractions in C are *load-bearing*: each defends
against a named, documented growth vector in the §8 proposal
roadmap. Speculative extractions are the failure mode the original
19-file proposal exhibited — fixed by following the rule "extract in
response to load, not in anticipation of speculative load."

If a new growth vector emerges in the next ~6 months that doesn't fit
the existing 13 files, that's the right moment to add the 14th.
Candidate triggers:
- **Vortex needs a 2nd SBT format** → `rtu_sbt.{h,cpp}`
- **§11 multi-RTU shared BVH cache** → split BvhCache out of `rtu_memory.cpp` into `rtu_bvh_cache.{h,cpp}`
- **Hardware BVH builder** → `rtu_bvh_builder.{h,cpp}`

---

## 14. Acceptance criteria for the refactor commit(s)

- All 18 RTU regression tests pass at each migration step.
- File count: exactly **13 files** under `sim/simx/rtu/`.
- No single file > **500 LoC** (target ~250-400 each; rtu_walker.cpp
  may approach 500 as the largest).
- `rtu_core.cpp::Impl::tick()` reads as a 5-line list of engine calls.
- Layering invariant (§4): no file includes a higher-numbered layer.
- Doxygen-style header comments on each public class describe its
  one-sentence responsibility, its dependencies, and its RTL analog.
- `docs/proposals/rtu_simx_proposal.md` §8.5 onwards updated to
  point to file names in the new layout.

## 15. Sign-off

When the review is approved, the refactor commits land first
(steps 1–12 in §12), then the §8.6 follow-up (with `vx_rt_get_after`,
new `rtu_smoke_async_batch` test, and the kernel-edit sweep) lands
on top.

Ready for review.
