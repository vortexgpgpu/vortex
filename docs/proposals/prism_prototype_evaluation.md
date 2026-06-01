# PRISM SimX Prototype — Evaluation Report

Status: evaluation, no proposed work
Scope: read-only audit of the ray-tracing prototype committed to
`~/dev/vortex-raytracing` (vortex_2.x branch), described publicly in:

- Chen & Tine, *PRISM: Accelerating Ray Tracing on RISC-V GPU*
  (UCLA, source repo `LazyLatte/vortex-raytracing`).
- The associated baseline paper, Saed et al., *Vulkan-Sim*, MICRO 2022.

The prototype is an attempt to port the Vulkan-Sim ray-tracing pipeline
onto Vortex by adding (i) a RISC-V "RTU" ISA extension, (ii) a SimX
functional model of the Ray-Tracing Unit (`RTUnit` + `RTCore`), and
(iii) a decoupled cycle-level timing model (`RTSim`). RTL was never
written; this is a pure SimX prototype.

This document does **not** propose a redesign. It is a punch list and a
foundational-flaws section intended to inform any future RT work on the
v3 tree (`~/dev/vortex_v3/rtx`).

Code paths referenced below use the prototype's source root
`~/dev/vortex-raytracing/…`. Equivalent paths in v3 do not yet exist —
the RT extension has not been ported.

## 1. What the prototype is

### 1.1 Bird's-eye view

The prototype adds a fifth functional unit class to Vortex, gated by
`EXT_RTU_ENABLE`:

- A **macro instruction** `trace_ray(...)` is expanded in the decoder
  into four micro-ops that move ray properties into a per-(warp,thread)
  shader state slot and then trigger traversal.
- A C++ **functional ray engine** (`RTCore`) walks a 6-wide compressed
  wide-BVH (CW-BVH) entirely at execute time, recording every memory
  load it issues into a per-thread transaction queue.
- A separate **cycle-level timing engine** (`RTSim`) drains that
  transaction queue against a configurable RT-cache/DCache port, with
  fixed-latency BVH-internal/leaf/intersection delays.
- Non-terminal shader invocations (any-hit / intersection) and terminal
  shader invocations (closest-hit / miss) are pushed into four hardware
  queues; a separate `get_work` instruction pops up to `SIMD_WIDTH` ray
  IDs of one queue type into a warp.
- The kernel programmer manually partitions warps into "producer"
  (issues `trace_ray`) and "consumer" (polls `get_work` and dispatches a
  Shader Binding Table entry). Synchronization is via a local-memory
  `producer_done` flag plus per-payload `done` flags polled with
  `vx_vote_all`.

### 1.2 The ISA extension

Defined in [tests/regression/raytracing/kernel.cpp:1](../../sim/simx/decode.cpp)
and the macros in [kernel/include/vx_raytrace.h](../../kernel/include/vx_raytrace.h).
The encoding reuses `RISCV_CUSTOM0` with funct7=3 and funct3 selecting
one of five operations. SimX decode lives in
[sim/simx/decode.cpp:1248-1332](../../sim/simx/decode.cpp).

| funct3 | Op       | Behaviour                                                                                 |
|--------|----------|-------------------------------------------------------------------------------------------|
| 0      | RTX      | Decoder splits this into 4 micro-ops: SET_ATTR(ORIGIN), SET_ATTR(DIRECTION), SET_ATTR(T+PAYLOAD), TRACE. Uses 10 source registers (8 FP + 2 INT). |
| 1      | GET_WORK | Returns a per-lane bitmask `(1 << shader_type)` of the popped warp's queue type, 0 if no work. |
| 2      | GET_ATTR | Reads one of 27 immediate-indexed ray/hit attributes from the per-(warp,thread) state.    |
| 3      | SET_ATTR | Writes ORIGIN / DIRECTION / T+PAYLOAD / HIT_ATTR_N.                                       |
| 4      | COMMIT   | One of four actions: ANYHIT_ACCEPT / ANYHIT_IGNORE / INTERSECTION_ACCEPT / INTERSECTION_IGNORE. |

The attribute index space is defined in
[hw/VX_types.toml:271-310](../../hw/VX_types.toml):

```
VX_RT_WORLD_RAY_RO_X..VX_RT_OBJECT_RAY_RD_Z   // 12 ray-property reads
VX_RT_T_MIN, VX_RT_T_MAX
VX_RT_HIT_T, VX_RT_HIT_ATTR_0..7              // u,v + 6 user attrs
VX_RT_HIT_INSTANCE_ID, VX_RT_HIT_PRIMITIVE_ID, VX_RT_HIT_GEOMETRY_INDEX
VX_RT_PAYLOAD_ADDR
VX_RT_RAY_ORIGIN, VX_RT_RAY_DIRECTION, VX_RT_RAY_T_PAYLOAD  // setter aliases
VX_RT_INTERSECTION_ACCEPT/IGNORE, VX_RT_ANYHIT_ACCEPT/IGNORE
```

Five DCRs at 0x006-0x00A hold global TLAS / BLAS / BVH / TRI / AABB base
pointers — one scene at a time, per-device, no per-kernel scoping.

### 1.3 The SimX functional model

Implemented in
[sim/simx/rt_core.{h,cpp}](../../sim/simx/rt_core.cpp) and
[sim/simx/rt_unit.{h,cpp}](../../sim/simx/rt_unit.cpp).

State held per in-flight ray:

```
std::unordered_map<uint32_t, Ray>            rays_;             // world ray
std::unordered_map<uint32_t, TraversalState> traversal_states_; // ~1 KB each
std::array<ShaderQueue<32, SIMD_WIDTH>, 4>   shader_queues_;    // 4 queues
```

`TraversalState` carries: the (currently object-space) `Ray`, the
best-hit so far, `RT_BOX_INTERSECTION_WIDTH` (=8) candidate prim_hits, a
32-entry `TraversalTrail`, a 99-entry `ShortStack` (when `RT_STACK_SIZE
> 0`), a status enum, `node_ptr`, `root_ptr`, `root_level`, `level`,
`instanceID`, `geometryIndex`, leaf flags, prim batch state.

Each `TRACE` micro-op calls
[RTUnit::traverse → RTCore::allocate → RTCore::traverse](../../sim/simx/rt_core.cpp#L90).
`RTCore::traverse(rayID, thread_info)` is a single C++ function that
runs the *entire* BVH walk in a `while(1)` switch, until either:

- a non-opaque primitive batch is hit and an ANYHIT / INTERSECTION
  shader needs to fire (status becomes paused, queue entries pushed,
  function returns); or
- the TLAS root walk is finished and a CLOSEST / MISS shader is queued
  and the function returns.

Memory reads issued during this walk are *real*: `dcache_read` in
[sim/simx/rt_unit.cpp:42-44](../../sim/simx/rt_unit.cpp#L42) calls
through to `Core::dcache_read → Emulator::dcache_read → mmu_.read`. The
walk has the actual data and produces the actual best-hit.

In parallel, each load is appended to
`thread_info.RT_mem_accesses` (a `std::deque<RTMemoryTransactionRecord>`
keyed by address, size, and a `TransactionType` enum). That deque is the
only handoff to the timing model.

`commit()` ([rt_core.cpp:255](../../sim/simx/rt_core.cpp#L255)) resumes
the same ray's `TraversalState`. For opaque primitives the user's
hit-t replaces best_hit unconditionally; for non-opaque the prim_hit
slot is cleared. When all prim_hit slots of the current batch are
drained, `traverse()` is recursively re-entered for the same ray. There
is no upper bound on this recursion depth.

### 1.4 The SimX timing model

Implemented in
[sim/simx/rt_sim.{h,cpp}](../../sim/simx/rt_sim.cpp) and the per-warp
trace data in [sim/simx/rt_trace.{h,cpp}](../../sim/simx/rt_trace.cpp).

RTSim runs once per cycle and admits up to `RT_WARP_BUFFER_SIZE` (=4)
warps simultaneously. For each admitted warp it:

1. `add_warp()` — drains one input warp from any of the `ISSUE_WIDTH`
   `Inputs` channels into the `warp_buffers_` set. Warps with no
   recorded mem accesses (non-TRACE/COMMIT ops, e.g. GET_WORK) are
   pass-through.
2. `schedule_warp()` — picks the first non-stalled warp in the buffer
   (insertion-order via `std::unordered_set` — non-deterministic in
   theory; observed deterministic per-run because libstdc++).
3. `process_memory_request()` — emits one read (or one queued store) per
   cycle from the chosen warp to `rtu_mem_req[0]` (single block, single
   port — `NUM_RTU_BLOCKS=1`).
4. `process_memory_response()` — consumes one response from
   `rtu_mem_rsp[0]`, looks up the originating thread by **address match**
   on the head of that thread's RT_mem_accesses deque, advances it, and
   stamps the per-transaction-type latency (e.g.
   `RT_BVH_INTERNAL_NODE_LATENCY=8`) into the thread's
   `intersection_delay` counter.
5. `process_intersection_delay()` — decrements each thread's
   `intersection_delay`. When it hits 0, queued stores (if any) are
   pushed to `mem_store_q`.
6. `check_completion()` — when a warp has empty mem_accesses, zero
   intersection_delays, and zero pending writes, it is forwarded to the
   `Outputs` channel with its accumulated wall-cycle count as the
   delivery delay.

Per-thread cycle distributions across `{warp_stalled, warp_waiting,
warp_executing} × {awaiting_processing, awaiting_scheduling, awaiting_mf,
executing_op, trace_complete}` are accumulated in
`RtuTraceData::track_rt_cycles` and aggregated at warp completion.

### 1.5 The producer/consumer kernel pattern

The pattern is hard-coded in the test kernel at
[tests/regression/raytracing/kernel.cpp:43-163](../../tests/regression/raytracing/kernel.cpp#L43):

```
if (vx_warp_id() % 2 == 0) {
    // PRODUCER: emit BATCH_SIZE rays/thread, poll payloads[i].done,
    //          re-emit bounces up to arg->max_depth, write back pixel
    *producer_done = 1;
} else {
    // CONSUMER: while (*producer_done == 0)
    //   ret = vortex::rt::get_work();
    //   if (ret) sbt[ctz(ret)](arg);
}
```

The pairing is purely a kernel convention; nothing in the RTU enforces
it. The SBT lives in a kernel-loaded buffer of four function pointers,
one per shader stage, written by the host via DCR-less buffer alloc and
indexed in the consumer via `__builtin_ctz(ret)`.

## 2. Does it work?

### 2.1 What appears correct

- The 6-wide CW-BVH layout (`cwbvh_node_t` in
  [tests/regression/raytracing/common.h:69](../../tests/regression/raytracing/common.h#L69))
  matches the layout `RTCore::traverse` expects.
- Ray-box and ray-triangle intersection math
  ([rt_core.cpp:298-418](../../sim/simx/rt_core.cpp#L298)) are
  textbook slab / Möller-Trumbore.
- BLAS object-space transformation
  ([rt_core.cpp:281-296](../../sim/simx/rt_core.cpp#L281)) correctly
  applies the inverse 3×4 transform on instance entry.
- The TLAS / BLAS handoff state machine — `INSTANCE_HIT` → reset
  `root_ptr`, transform `state.ray`, increase `root_level` — at least
  for *single-level* instancing returns to the TLAS root correctly when
  `FINISHED` is hit inside the BLAS.
- The short-stack mechanism implements the Vaidyanathan/Woop/Benthin
  "wide BVH traversal with a short stack" approach
  (`ShortStack` in [sim/simx/types.h:1803](../../sim/simx/types.h#L1803)),
  with the encoded "isFarthest" bit packed into the low bit of pushed
  node addresses (relies on 2-byte alignment).
- Shader-queue dispatch returns the right shader type, and the OR-mask
  return value lets the consumer warp dispatch all 32 lanes coherently
  in a single iteration (lanes pulled from the same queue are always
  the same shader type).
- Cornell Box and Sponza scenes render to correct-looking PPM images at
  the scales the prototype tests (128×128).

### 2.2 Bugs and ambiguities visible in the code

These are findings I am confident about from reading the source. They
are *not* speculative.

#### B1. Short-stack `bottom_` is never read

`ShortStack::pop()` ([sim/simx/types.h:1818-1824](../../sim/simx/types.h#L1818))
uses only `count_` and `head_`. The `bottom_` field is bumped on
overflow in `push()` but never consulted. The wrap-around silently
discards the oldest entry without ever flagging the consumer. This is
formally consistent with the wide-stack restart protocol *only* because
`TraversalStatus::RESTART` re-enters from `root_ptr` and the trail
prevents re-visiting subtrees — but the design accepts an unbounded
re-fetch cost without exposing it as an event.

#### B2. `TraversalState::pop()` uses an int32 cast of `level - 1`

[rt_core.h:205-212](../../sim/simx/rt_core.h#L205). If `level == 0`
(possible after a BLAS finish if `root_level` was 0), the loop body
runs once with `i == -1` after the cast, and returns the parent level
of the TLAS root as `-1`, triggering `FINISHED`. This is the intended
behaviour. But the loop bound `i >= (int32_t)root_level` will also
mismatch when `root_level` is non-zero and `level` happens to equal
`root_level` (no parent in this AS) — it would return the level above
the BLAS root, i.e. a TLAS level whose trail is no longer trustworthy
(see B6).

#### B3. ANYHIT path overwrites `commit()`'s commentless return-path bug

In [rt_core.cpp:263-269](../../sim/simx/rt_core.cpp#L263):

```cpp
else if(type == ShaderType::INTERSECTION){
    if(state.leaf_flags == OPAQUE){
        state.best_hit = hit;
    }else{
        state.prim_hit[hitID] = hit;
        //return ShaderType::ANYHIT;  // should return when !state.has_prim_hit()
    }
}
```

The commented-out `return ShaderType::ANYHIT` suggests an intended
chained-shader flow (intersection-accept → enqueue any-hit). The code
as committed does *not* chain — `prim_hit[hitID]` is set in-place
without re-pushing the ray into the ANYHIT queue. The author flagged
this and left it for later. Result: non-opaque procedural primitives
with any-hit shaders are silently incorrect.

#### B4. `state.prim_hit[hitID].valid = false` after a non-opaque INTERSECTION

[rt_core.cpp:273](../../sim/simx/rt_core.cpp#L273) clears the valid bit
*after* the user accepted the hit and stashed it in `prim_hit[hitID]`.
The next iteration of the same prim-batch will then see `has_prim_hit
== false` and pop the BVH — meaning the accepted hit is never re-read
through the comparator tree, and only the BEST_HIT path (which OPAQUE
takes) records the final hit. Combined with B3 this means the
non-opaque INTERSECTION accept never updates `best_hit`. The user's hit
is functionally lost; visually procedural geometry is missed.

The sphere test renders correctly only because `IS` immediately calls
`set_attr<VX_RT_HIT_T>` and then commits `INTERSECTION_ACCEPT`, and the
CHS-only test (no any-hit) happens to read the result via
`prim_hit[hitID]` before the clear — but the in-the-loop semantics of
the commit path do not propagate it. *I have not run the prototype to
confirm at which scenes the bug becomes visually visible*; it is
visible by inspection.

#### B5. `dcache_read` size mismatch on BLAS instance leaf

[rt_core.cpp:110](../../sim/simx/rt_core.cpp#L110) reads exactly **52
bytes** for a `BLASNode` whose `sizeof` is 104 bytes (8B header + 48B
invTransform + 48B transform = 104B if `mat3x4_t` is 12 floats). The
host-side struct
([common.h:117](../../tests/regression/raytracing/common.h#L117)) has
`bvh_offset (4) + invTransform (48) + transform (48) + mat_offset (4) +
padding (24) = 128 B`. So the SimX-side `BLASNode` (different layout, no
transform/mat_offset/padding) is *not* the same as the host-side
`blas_node_t`. The functional read of 52 bytes covers the device-side
SimX struct's `{bvh_offset (4), invTransform (48)}` = 52 B, which
matches what the SimX traversal needs — but only because the device
host-side layout puts `invTransform` immediately after `bvh_offset`.
Any change to either side (e.g. natural alignment, a transform write)
silently corrupts the BLAS read. This is fragile; there is no static
assertion.

#### B6. TLAS↔BLAS trail mixing

`TraversalTrail` is a single 32-entry array shared between TLAS and
BLAS levels. When entering a BLAS at TLAS level `L`, `state.root_level
= L` and `state.level = L+1`. The BLAS uses trail entries `[L+1 .. L+1
+ blas_depth - 1]`. When the BLAS finishes
([rt_core.cpp:201-208](../../sim/simx/rt_core.cpp#L201)) the code
restores `root_level = 0` and calls `pop()`, which walks `level - 1`
down to `root_level` looking for a still-unexplored sibling. But the
trail at the TLAS levels still holds the values it had when the BLAS
was entered — they were not modified during the BLAS walk, so this is
*usually* correct. The fragility: if the BLAS itself contains a nested
INSTANCE_LEAF (multi-level instancing), the inner BLAS's `root_level`
overwrites the outer's, and `findNextParentLevel` cannot tell which
level belongs to which AS. The code assumes single-level instancing.

#### B7. `MAX_TRAIL_LEVEL = 32` is a hard cap on total depth

[rt_core.h:11](../../sim/simx/rt_core.h#L11). For Sponza the prototype
reports BVH depth 12; TLAS depth is ~3-4. There is no overflow handling
— `trail[parentLevel]++` on `parentLevel == MAX_TRAIL_LEVEL-1` runs off
the array on the next BLAS push. No assert in the SimX build, no
trace-debug helper.

#### B8. `ray_id_` overflow

[rt_unit.cpp:55](../../sim/simx/rt_unit.cpp#L55) `(ray_id_++) &
0x0FFFFFFF` — 28 bits, 268 M rays. The high 4 bits are reserved for
`hitID`. For a 128×128 cornellbox render with 1 sample/pixel this is
fine; for production rendering (1080p × 32 spp × multi-bounce) you
overflow inside one frame and collide live entries in
`traversal_states_`.

#### B9. `shader_states_[wid][tid]` overwrite race

The SimX RTU keeps exactly one `ShaderState` per (warp, lane). The
producer warp's `TRACE` writes `state.world_ray / tmin / tmax /
payload_addr` to `shader_states_[producer_wid][tid]`; the consumer
warp's `GET_WORK` writes `world_ray / object_ray / tmin / tmax / hit /
payload_addr` to `shader_states_[consumer_wid][tid]`. Because producer
and consumer warps have different IDs there is no overwrite *between*
them. But two back-to-back `GET_WORK`s on the same consumer warp
overwrite each other before the first shader runs. The consumer
serializes via the kernel C++ loop, so functionally OK; but the
hardware does not enforce it.

#### B10. Memory store path is plumbed but inert

`MemoryStoreTransactionRecord` and `mem_store_q` are wired through
[rt_sim.cpp:131-151](../../sim/simx/rt_sim.cpp#L131) but `RT_store_transactions`
is never populated by `RTCore` — the only intended writer was the
commented-out hit-point store in the QUAD_LEAF_HIT path. The pending
writes set is therefore always empty, and the `has_pending_writes()`
check in completion is always trivially true.

#### B11. `process_returned_mem_access` is address-matched per thread

[rt_trace.cpp:97-115](../../sim/simx/rt_trace.cpp#L97). A returning
response is matched to a thread by comparing its address against
`m_per_scalar_thread[tid].RT_mem_accesses.front().addr`. If multiple
in-flight threads have the same address at head (BVH root is the
classic case), all of them advance on a single response — which is the
*intended* coalescing behaviour. But if two threads' deques have the
same address at non-head positions (a deeper sibling re-visit pattern)
the response will only match the heads, leaving the later occurrence
stuck. The deduplication in `update_next_rt_accesses` makes this rare
but not impossible.

#### B12. `is_stalled()` checks only address-not-in-queue; not waiting-on-op

A warp is reported stalled only if no thread has an unmarked memory
access *and* none has an op delay. This is correct for back-pressure
into the dispatcher. But the `track_rt_cycles` bucketing treats
"in-op-delay" as `warp_executing` for the chosen warp and as
`warp_waiting` for the others — which conflates "this warp is being
scheduled" with "the op is making progress". The published latency
distribution histograms therefore over-count the chosen warp's
executing cycles by `op_latency * (num_other_warps)`.

#### B13. The "warp partitioning is hardware" claim is software-only

The PRISM paper's contribution list claims a "RISC-V RT Extension
providing custom instructions for low-latency communication between
GPGPU warps and the RT Core" and "Symmetric Warp Partitioning"
producer-consumer pairing. The prototype implements neither in
hardware:

- The ISA extension has *no* notion of producer or consumer; both
  TRACE and GET_WORK are runnable from any warp.
- The 1:1 even/odd pairing lives entirely in `kernel.cpp` source code,
  reinforced only by a software `producer_done` flag in shared memory.
- The PRISM-claimed `FETCH` opcode is actually just `GET_WORK`. The
  paper's `ATTR` family is `GET_ATTR / SET_ATTR`. The 1:1 mapping in
  the paper's Table II is at best an aspirational view of what the
  prototype does.

Status: the "low-latency communication" is just memory-mapped
`shader_states_[wid][tid]` updates plus an OR-mask return; no register
forwarding, no hardware queue arbitration, no pairing enforcement.

#### B14. Producer can deadlock

The producer waits on `vx_vote_all(payloads[i].done)`
([kernel.cpp:101](../../tests/regression/raytracing/kernel.cpp#L101)).
`done` is set only by the terminal CLOSEST/MISS shaders. If the
consumer warp is starved (e.g. dispatcher always picks producer first,
or the queue is empty because shader callbacks never executed because
the producer's TRACE has not yet drained from RTSim into RTCore's
queue), the producer spins forever.

In practice the prototype works because (a) RTSim drains warps in
limited batches via `RT_WARP_BUFFER_SIZE=4`, (b) `add_warp` round-robins
across the `ISSUE_WIDTH` Inputs, and (c) the dispatcher's round-robin
batch indexing happens to interleave producer/consumer TRACE/GET_WORK
roughly fairly. None of this is a guaranteed schedule.

#### B15. `EPSILON = 1e-6f` and `LARGE_FLOAT = 1e30f` are arbitrary

[rt_core.h:9-10](../../sim/simx/rt_core.h#L9). Both are used in
intersection self-occlusion guards and the t-far init. The
ray-triangle intersection's `if (tf <= EPSILON) return LARGE_FLOAT`
returns a sentinel that is then compared against `state.best_hit.t` in
`ray_nTri_intersect`. Self-intersection avoidance is the user's
problem (the kernel passes `T_MIN = 0.0001f`); the constants don't
match.

## 3. Efficiency

### 3.1 Functional cost

Every `TRACE` micro-op runs the *entire* traversal as a single C++ call
at execute time, in zero simulated cycles. For a Sponza primary ray
visiting ~150 BVH nodes (paper's reported average × 2 for two-level
walks) the work per `TRACE` is:

- 150× node read from MMU memory (real read, real data)
- 150× quantized AABB decode + ray-box test on up to 6 children
- 8× short-stack push / pop pairs (typical)
- 4-12 ray-triangle tests per triangle leaf

Order of magnitude: hundreds of nanoseconds of host CPU time per ray on
the SimX side, *per* simulated cycle of `TRACE` issue. For a 128×128
Sponza render that's ~16 K rays × 1 sample × max_depth=1, i.e. ~16 K
`TRACE`s, plus shader recursion. SimX's wall clock is dominated by this
traversal cost; the RTSim cycle pump (point 5 below) is trivial in
comparison.

### 3.2 Memory transaction volume

The functional model records *every* node read in `RT_mem_accesses`,
even when the same address is read by multiple lanes of the same warp.
Deduplication happens in `update_next_rt_accesses` at scheduling time,
not at recording time. For a 32-thread warp tracing primary rays
through the same root, the recording is 32× the unique address count;
the deduplicator collapses it to 1× on the wire — but only because the
deduplicator stores a `set<pair<addr,size>>` and walks it linearly each
cycle. For deep traversals this is O(num_warps × num_threads × depth)
work per cycle.

### 3.3 Memory port utilization

`NUM_RTU_BLOCKS = 1` and `RT_WARP_BUFFER_SIZE = 4`. The RT pipeline
emits **one** memory request per cycle across all 4 admitted warps. The
warp picker picks the first non-stalled warp; given typical primary-ray
coherency, almost all 4 warps hit the same node at the same time, so 3
of 4 are stalled waiting on the (single) response, and the chosen one
issues another address one cycle later. In steady state this gives
throughput of 1 request / cycle / RT-unit, regardless of the 4-warp
parallelism. The "8 warps" sweet-spot Vulkan-Sim claims is not
reproduced; the prototype is bottlenecked by the 1-port memory
adapter.

### 3.4 Fixed latencies

All four BVH transaction types are 8 cycles
([VX_config.toml:252-255](../../hw/VX_config.toml#L252)). No
parameterization per cache-line size, per-BVH-width, or per intersection
unit instance count. The intersection-test cost is folded into the
single "BVH_QUAD_LEAF_HIT" 8-cycle stamp. With
`RT_TRI_INTERSECTION_WIDTH = 4`, the simulator says it can intersect
four triangles in 8 cycles regardless of whether the SIMD unit is busy
on another batch.

### 3.5 Shader callback cost

`commit()` triggers a recursive `traverse()` call which records *more*
mem accesses into the same `thread_info.RT_mem_accesses` deque
([rt_core.cpp:277](../../sim/simx/rt_core.cpp#L277)). The producer warp
is back-pressured by the dispatcher only when it tries to issue another
TRACE — but each commit happens via a separate `COMMIT` instruction
running on the consumer warp. That commit instruction creates a *new*
`RtuTraceData` ([execute.cpp:1628](../../sim/simx/execute.cpp#L1628))
and the commit's recursive traverse writes recorded accesses into that
new trace_data. So shader-callback mem accesses are billed to the
consumer warp's COMMIT issue, not to the producer warp's original
TRACE. The PRISM paper's per-warp-latency comparison against Vulkan-Sim
therefore measures only the *primary* traversal cost on producer
warps; the consumer-side commit-driven re-traversal is hidden in a
separate latency bucket the paper does not show.

## 4. Complexity

| Layer | Lines | Files |
|---|---|---|
| RTCore (functional engine) | 433+292 | rt_core.{h,cpp} |
| RTUnit (per-core wrapper)  | ~365 | rt_unit.{h,cpp} |
| RTSim (timing engine)      | 253+48 | rt_sim.{h,cpp} |
| RtuTraceData (per-warp telemetry) | 222 | rt_trace.{h,cpp} |
| SimX integration (decode/execute/types/func_unit/core/dcrs) | ~250 | spread across simx/ |
| RT runtime header           | 79 | kernel/include/vx_raytrace.h |
| RT test app (host + 5 vxbins) | ~3000 | tests/regression/raytracing/ |
| Total RTU-specific SimX     | ~1900 (excl. test app) | |

The SimX additions are well-localized and gated by `EXT_RTU_ENABLE`.
The integration touchpoints are minimal:

- One new `FUType::RTU` and one new dispatcher slot in
  [core.cpp:222](../../sim/simx/core.cpp#L222).
- One new memory-adapter block in
  [core.cpp:165-197](../../sim/simx/core.cpp#L165).
- One new `RtuType` switch in `decode` and `execute`.
- Five new DCRs at `VX_DCR_BASE_RTX_*`.
- One new dispatcher buffer depth of `2`.

By comparison the test application is larger than the simulator code,
because the host side has to build the CW-BVH, encode quantized child
AABBs, pack triangle leaves, and convert between two-level
host-vs-device struct layouts (see B5).

## 5. Limitations

L1. **Single-scene per device.** The TLAS / BLAS / BVH / TRI / AABB
base pointers are global DCRs. Multi-scene workloads cannot coexist;
neither can multi-pipeline workloads (different shader binding tables).

L2. **No vkCmdTraceRays-equivalent dispatch shape.** The host hard-codes
`block_size = (8,8)` and a 2D grid. There is no notion of (width,
height, depth) like Vulkan or DXR. Re-targeting to a compute kernel is
the user's problem.

L3. **SBT is four function pointers in a kernel-loaded buffer.** No
groupings, no hit-groups, no shader-record stride/offset semantics
beyond the placeholders in the payload struct
([shader.h:13-15](../../tests/regression/raytracing/shaders/shader.h#L13)).
The example kernel embeds the stride/offset fields in the payload but
the shaders ignore them
([closet-hit.cpp:19-21](../../tests/regression/raytracing/shaders/closet-hit.cpp#L19)).

L4. **No callable shaders, no ray flags, no skip-AABB.** Vulkan's
`RayFlagsBits` (cullFront, cullBack, opaque, terminate-on-first-hit,
skip-closest, skip-triangle, skip-AABB) are absent. Opacity is a
*per-leaf* flag in the BVH node, not a per-ray override.

L5. **No `traceRayEXT` from within a non-raygen shader.** Recursion via
`traceRayEXT` calls in CHS/AHS is the Vulkan default mode. The
prototype's kernel does it by *the producer warp* re-issuing a
`vortex::rt::trace_ray` in its own polling loop after the previous
bounce's CHS has set `payload->done`. So the CHS itself never calls
trace_ray; only the producer kernel does. This rules out
trace-from-shader and ray reordering across bounces.

L6. **No payload register file.** Payload is shared via the kernel's
own struct at `payload_addr` and accessed by shaders via a dependent
memory load. Vulkan's `vkLocations`-style payload register layout is not
modelled.

L7. **No NIR/SPIR-V/DXIL frontend.** Shaders are hand-written C++
compiled to RISC-V .vxbin blobs at known load addresses
(0x80100000-0x80400000). Shader binding is by elf offset in a flat
SBT.

L8. **No RTL.** The PRISM paper Table I claims "Synthesizability: Yes",
but no SystemVerilog exists in the prototype tree under
`hw/rtl/`. The "synthesizable" claim rests entirely on the *intent* of
the data structures, not on an RTL implementation. The author's own
README at the source repo confirms this.

L9. **Performance evaluation scope.** Only two scenes (Cornell Box,
Sponza) at 128×128, primary rays + first-bounce, no denoiser. SIMT
efficiency in the RT unit, DRAM efficiency, cache-miss breakdown, and
roofline analysis from Vulkan-Sim are not reproduced — the paper
reports only `avg warp latency` and `avg thread latency` for these two
points.

L10. **No simx↔rtl differential validation.** Since no RTL exists,
there's no oracle. The Vulkan-Sim baseline being compared against is
itself a simulator; the prototype's claim that it "tightens
warp/thread latency convergence" is two unvalidated models compared
against each other.

L11. **No multi-RT-unit cross-warp sharing.** `NUM_RTU_BLOCKS=1` and
`RT_WARP_BUFFER_SIZE=4` are global. Scaling to more cores creates more
RT units, each independent — no shared BVH cache, no inter-RT shader
queue.

L12. **Cap on `RT_BOX_INTERSECTION_WIDTH=8` and `RT_TRI_INTERSECTION_WIDTH=4`**
are baked into the `Hit prim_hit[8]` array. A leaf with more than 8
procedural primitives is iterated in `RT_BOX_INTERSECTION_WIDTH`-sized
batches via `state.prim_batch_finished_count`. Functional but slow.

L13. **No exception path.** A malformed `node.type` triggers
`std::abort()`. A host that uploads an invalid BVH crashes SimX.

## 6. Foundational flaws

This is the section the rest of the document was leading to. These are
not bugs; they are *design* problems that would block any attempt to
take this prototype to RTL or to use it as a credible architecture
study.

### F1. Functional traversal and timing model are entirely decoupled

This is the single most important flaw, and it cascades into most of
the others.

`RTCore::traverse(rayID, thread_info)` runs at execute time as a
zero-cycle C++ function. It walks the BVH, computes intersections,
fires shader-queue pushes, and returns. The memory accesses it
performed along the way are recorded into a `std::deque` in *issue
order*, but with no causal annotation: there is no "this load depends
on the result of that load". The timing model then drains the deque in
issue order, with constant per-type latencies, against a single memory
port.

Consequences:

1. The simulator cannot model the actual serial dependency chain of
   BVH traversal — every BVH node access depends on the *result* of
   the parent's child-pointer field. An RTL implementation would have
   to wait for the parent's data to arrive before issuing the child's
   address. The prototype issues both in the same execute cycle and
   replays them with no ordering constraint between them. The reported
   cycles are an under-estimate by at least the depth × per-node
   latency for any non-coherent ray.

2. Cache state has no effect on the functional traversal. The
   prototype's MMU reads always succeed; the cache hierarchy sees the
   replayed transaction stream after the fact. A real BVH miss should
   spike both the per-load latency *and* the dependent operations
   downstream; the prototype model has no causal path for that.

3. Shader callbacks execute via `commit()`, which calls `traverse()`
   recursively. The recursion happens at the consumer warp's COMMIT
   execute time — not at the producer's TRACE time. So the producer's
   TRACE latency in the reported stats includes only the *primary*
   traversal cost; secondary traversal cost is billed to a different
   warp. The PRISM paper's warp-latency comparison versus Vulkan-Sim
   is therefore comparing different quantities.

4. There is no way to validate the timing model against any RTL
   ground-truth, because the functional model bypasses every structural
   bottleneck the timing model is supposed to expose.

### F2. The Ray Context is unbounded software state

The PRISM paper (Section II, "PRISM RT Core Architecture") shows a
`Ray Context Allocator` that "assigns a slot" from a hardware-resident
pool. The prototype's `RTCore::allocate` does:

```cpp
traversal_states_[rayID] = TraversalState(world_ray, tlas_addr, tmin, tmax);
```

That is a `std::unordered_map<uint32_t, TraversalState>` insertion.
There is no pool, no capacity check, no back-pressure when full. The
key is the auto-incrementing `ray_id_` (B8) so collisions are deferred
to 2^28 rays.

Each `TraversalState` is ~1 KB of mixed POD: 24 B ray + 8 hits ×
~40 B + 32 trail entries × 4 B + 99 stack entries × 4 B + ~50 B of
flags / pointers. With even a single core, a busy 32-thread warp will
hold 32 × 1 KB = 32 KB of live ray context inside RTCore. Multiply by
the dispatcher's in-flight depth and the prototype routinely holds
hundreds of KB.

For RTL this is unacceptable. A real RT core must:

- Have a fixed-capacity ray context BRAM (Vulkan-Sim sized this at "up
  to eight warps" per RT unit, with a published cell count).
- Apply back-pressure to TRACE issue when the context pool is full.
- Spill long-stack state to per-thread DRAM (Vulkan-Sim does this via
  the short-stack + spill scheme).

The prototype models *none* of this. Any RTL port would have to throw
away `traversal_states_` and replace it with a bounded pool, which
breaks the functional model's "allocate-on-issue" assumption (F1).

### F3. The C++ recursion in shader callbacks has no analog in hardware

`commit()` re-enters `traverse()` for the same ray. The Vulkan-Sim
"delayed shader execution" model defers all any-hit / intersection
shader work to *after* the BVH walk completes, by storing intersection
records in a buffer. The PRISM paper distances itself from this and
calls its own approach "interleaved shader execution" with
"suspend-and-resume via stateful hardware contexts". In the prototype,
"interleaving" is implemented as:

- The C++ function `traverse()` returns when it hits a non-opaque leaf
  with valid prim_hits.
- The TraversalState struct stays in `traversal_states_[rayID]`.
- A separate `COMMIT` instruction, possibly cycles later, re-invokes
  `traverse(rayID)` from the C++ side, which picks up where it left
  off.

This has zero modelled cost. There is no:

- Pipeline drain to extract live registers.
- Spill of in-flight box-test / triangle-test state.
- Resume cost paid by re-fetching cached state.
- Bandwidth/IO accounting for the suspend.

A real RTL implementation of "suspend-and-resume mid-traversal" is
expensive: the in-flight intersection result has to be checkpointed,
the next-node pointer queue has to be paused, the inputs to the
intersection unit have to be saved. The PRISM contribution claim —
"hardware-managed suspend-and-resume mechanism" — has no hardware
mechanism in the prototype, only a C++ struct lifetime.

### F4. Producer/consumer is a software convention, not an ISA contract

(See B13.) The PRISM paper's Section III ("Hardware-Software
Interface") describes a *symmetric warp partitioning* scheme where
producer warps issue `TRACE` and poll a payload `done` signal, while
consumer warps execute a `FETCH` service loop. This is presented as
the *hardware-software contract*. In the prototype:

- The producer/consumer split is a single `if (vx_warp_id() % 2 == 0)`
  branch in user kernel C++.
- The `done` signal is a `volatile bool` in the user's payload struct
  in DRAM.
- The "fetch service loop" is a plain `while` loop polling
  `vortex::rt::get_work`.
- The "synchronized exit" is a shared-memory byte
  (`producer_done`) plus a `payload->stop = true; payload->done =
  true` pattern.

Nothing in the ISA or the RTU enforces the 1:1 pairing. Anything in
the kernel can issue both TRACE and GET_WORK. A user who gets the
pairing wrong — e.g. all warps producer, or producer count not equal
to consumer count — will not see a hardware error; the symptom will be
deadlock or zero pixels.

This is not the "RISC-V RT Extension for low-latency communication
between GPGPU warps and the RT Core" the paper claims. It is
*kernel-side message-passing through main memory*, with no
participation from the RTU.

### F5. The shader queue is a bag, not an ordered structure

Vulkan permits any traversal order for procedural and any-hit shaders.
The prototype takes this license to its extreme: `shader_queue_pop`
picks the *largest* of four queues to drain, and the queue itself is a
simple ring with no per-ray identity. The consequence is that the
ordering of any-hit and intersection invocations across rays is a
function of dispatcher scheduling, dispatcher round-robin, and the
warp-buffer admission order in RTSim.

This is correct under Vulkan semantics for any-hit and
intersection. It is *not* correct for CHS or MISS: there should be at
most one terminal shader per ray, and Vulkan does not permit a CHS to
race with another shader on the same ray. In the prototype, two
back-to-back `get_work` invocations on the same consumer warp can
return a CHS+MISS interleaving across lanes — fine if both lanes carry
*different* rays, which is the case here, but the per-ray ordering is
not enforced by the RTU.

Worse, the rays that fire a CHS lose their `TraversalState` only when
the consumer side calls back into the runtime. There is no explicit
`endTraceRay()` — `traversal_states_[rayID]` lives forever once
allocated. Memory leak.

### F6. Latency model is constants × address count

Per-transaction latency is a single constant per type, summed into
`intersection_delay`. There is no:

- L1D / L2 / DRAM hierarchy specific to RT loads (the dedicated RT
  cache is just a plumbed-but-unsized `MemArbiter` slot).
- Bank parallelism, row-buffer locality, or memory controller
  scheduling.
- Per-warp coalescing accounting (deduplication exists but not in the
  latency calc).

Vulkan-Sim's roofline plot, DRAM efficiency study (avg 46% across
benchmarks), and cache breakdown chart all rely on a real memory
hierarchy. The PRISM prototype's timing model cannot produce any of
those, because the LSU-side cache it shares with regular loads is
bypassed by the functional reads (F1), and the RT-side adapter
emits one request / cycle regardless of cache state.

### F7. No path from prototype to RTL

The prototype has no RTL counterpart in `hw/rtl/`. The CW-BVH layout,
the short-stack, the ray-context pool — none of them have a
SystemVerilog file. The claim of synthesizability is unverified.

If RTL were attempted from this prototype, the following would have
to be redone from scratch:

- Ray context pool with a real allocator and back-pressure (F2).
- Memory request issue serialized by data dependency (F1).
- Suspend / resume state machine with explicit register save/restore
  (F3).
- An actual shader-queue arbitration unit with bounded capacity and a
  defined priority policy (F5).
- A cache interface that is *the same one* the functional model uses,
  not a parallel one (F6).
- A producer / consumer warp arbiter, or a workgroup-level scheduler
  that enforces the pairing (F4).

The shape of the existing SimX code does not constrain any of these
decisions. The prototype is a *functional reference* that happens to
be wrapped in a timing harness, not an *architectural specification*.

### F8. The ISA extension does not generalize

Reusing `CUSTOM0/funct7=3` is fine. But the encoding choices bind the
ISA tightly to one specific implementation:

- `VX_RT_HIT_ATTR_0..7` is fixed-width at 8 × 32 bits = 32 bytes.
  This is the Vulkan `hitAttributeEXT` analog (intersection → CHS/AHS
  handoff), and 32 bytes is in fact within Vulkan's hit-attribute
  budget — so the slot count is defensible. It is, however, baked into
  the attribute-ID immediate space and the `Hit` POD struct, so
  widening it later is an ISA break, not a config knob. (The Vulkan
  `rayPayloadEXT` analog is separate: the prototype handles the user
  payload by passing its address through `VX_RT_PAYLOAD_ADDR` and
  letting shaders dereference normally; hardware never touches the
  payload contents.)
- `traceRayEXT` is encoded as 8 floating-point inputs + 2 integer
  inputs in *fixed register positions* (f11-f18, x19-x20). This burns
  10 ABI registers per call site and forces the decoder to expand to
  four micro-ops. Compare with Vulkan-Sim's PTX extension, which adds
  *one* CISC-like `traverseAS` opcode that wraps the entire pipeline.
- `GET_WORK` returns only a per-lane shader-type bitmask; it carries
  no ray or hit state. The consumer warp must re-issue one `GET_ATTR`
  per attribute it needs (origin, direction, hit-t, hit-attrs, IDs,
  payload pointer), and each is R-A-W dependent on the previous —
  there is no burst path. This is the opposite of low-latency
  communication; for a typical CHS that touches ~6-10 attributes it is
  6-10 serial scalar reads from the RTU's attribute file before the
  shader can do useful work.
- No instruction for ending a ray's traversal explicitly. Once
  `traversal_states_[rayID]` is allocated, it is alive until the
  RTCore object is destroyed.

The PRISM paper's Table II lists a fifth opcode `COMMIT` whose role is
"resumes a suspended traversal state". In the prototype, `COMMIT`
*does* trigger the resume — but it does so by synchronously running
the entire next slice of the traversal in C++. There is no
"resume cost" to model, and there is no way for the user to know how
much work the COMMIT did. The opcode is a black-box: it could cost 1
cycle or 10 000 cycles depending on what the BVH looks like next.

### F9. The pose against Vulkan-Sim is favourable by construction

The PRISM paper reports lower average warp latency and lower
average-warp/average-thread gap than Vulkan-Sim on the same two
scenes. The improvements are claimed to come from:

- Eliminating delayed shader execution.
- Tighter SIMT efficiency via interleaved shader execution.
- Stateful Ray Contexts that avoid memory writes during traversal.

Each of these claims dissolves under inspection:

- "Eliminating delayed shader execution": the prototype does not in
  fact run any-hit/intersection inline with traversal. It returns from
  `traverse()` (i.e., yields) and waits for the consumer warp to be
  scheduled, run the shader, and call COMMIT, which re-invokes
  `traverse()`. The latency between yield and resume is *not* charged
  to the producer warp's TRACE bill; it's amortized into the consumer
  warp's GET_WORK + COMMIT bill. So the producer's warp latency is
  artificially low.

- "Tighter SIMT convergence via interleaving": the per-warp-tail
  problem Vulkan-Sim describes is that retiring warps wait for the
  longest thread. The prototype's TRACE issues *all 32 lanes' rays in
  parallel* but the traversal is serialized on the single dispatcher
  port (Sec. 3.3), so the "warp" finishes when the host-side C++ loop
  finishes — which is when the *first* call to `traverse()` returns,
  not when the last lane's ray finishes. The reported "warp latency"
  is therefore primary-traversal-only; long-tail effects are absorbed
  into the consumer side.

- "Stateful Ray Contexts that avoid memory writes during traversal":
  true in the prototype because the BVH traversal never writes back.
  But Vulkan-Sim's writes are for hit-record persistence across the
  delayed shader phase. The prototype does not need them only because
  it lifts the hit state into `traversal_states_[rayID]`, which is a
  C++ `unordered_map` not a hardware-bounded buffer (F2).

The conclusion is that the PRISM-vs-Vulkan-Sim comparison is not
apples-to-apples: the two simulators bill different cycle accounts to
different stages of the same pipeline. Both are sub-RTL models; the
prototype's lower numbers come at least in part from billing less
work to the metric being plotted.

### F10. No test of correctness beyond rendered images

The prototype's only correctness check is a PPM dump compared
visually. There is no:

- Cycle count regression vs a fixed cone of inputs.
- Cross-driver check (simx vs opae) — only simx ever ran.
- Cross-AS check (multiple BLAS instances, transforms with
  determinant ≠ 1).
- Adversarial BVH (degenerate triangles, empty leaves, deeply
  unbalanced trees).

This is consistent with prototype scope, but it means the bug list in
Section 2 cannot be discharged without a real test suite. Several of
the bugs there (B3, B4, B11) are not visible in the two test scenes.

## 7. Recommendations (not proposals)

This document is not a redesign proposal. The recommendations below
are framing for whoever picks up RT on v3.

R1. **Treat the prototype as a functional oracle, not a starting
point.** The BVH layout, the CW-BVH compressed format, the
ray-box / ray-triangle math, and the producer/consumer kernel pattern
are correct under Vulkan semantics and worth keeping. Everything in
`RTSim` and the per-trace memory deque should be replaced. See memory
[Use SimX as goal-reference oracle when RTL debugging stalls](feedback_simx_as_rtl_oracle.md)
for the canonical Vortex stance on this.

R2. **Re-do the ISA from scratch.** The 10-fixed-register macro-op
TRACE and the GET_ATTR / SET_ATTR attribute file should be replaced
with one of:

- A CISC traceRayEXT-style opcode that reads inputs from a *struct in
  memory*, like vulkan-sim's PTX. Throughput is bounded by the
  RT-core context pool, not by ABI register pressure.
- A vortex2-style minimal runtime API (cf.
  [feedback_runtime_minimalism.md](feedback_runtime_minimalism.md))
  with helpers in `vortex2_rt.h` and the hardware exposing just
  `dispatch`, `wait`, and `commit_attr`.

R3. **Build the cycle model on top of the existing functional model,
not beside it.** Replace `RTCore::traverse` with a state machine that
issues *one* memory read per `tick()` and waits for its response
before computing the next address. This dissolves F1, and lets the
shared L1D model both the functional and timing sides.

R4. **Use SimX as oracle for RTL by emitting a node-by-node trace
log.** The Vulkan-Sim paper's correlation study (Sec. VI.G) is the
right model — port a known external RTU's cycle count against the
SimX model, parameterizing the RT-cache and stack depth until the
slopes match. The prototype provides none of this.

R5. **Drop the producer/consumer kernel pattern in favour of
hardware-managed shader scheduling.** Vulkan-Sim's "one thread per
raygen" with on-demand shader callbacks reads (no kernel-side warp
partitioning) is closer to NVIDIA's model and aligns with the v3
memory [feedback_design_aligns_with_nvidia_gpu.md](feedback_design_aligns_with_nvidia_gpu.md).
The "symmetric warp partitioning" idea is interesting but the
prototype's pure-software realization is not it.

R6. **Make the ray-context pool a first-class architectural parameter.**
TraversalState should be defined in `VX_config` as a hardware-bounded
BRAM with a known cell count. Back-pressure into the TRACE issue
becomes the natural throttle for ray throughput.

R7. **Add at least one procedural-primitive any-hit test.** B3 / B4
are blocking for any procedural geometry. The current `IS` test only
covers opaque spheres without any-hit, so the bug is dormant.

R8. **Land RTL as a milestone before scaling the simulator.** The
prototype's value as an architecture study is bounded by the absence
of an RTL counterpart. The Vortex v3 convention (see
[feedback_no_prs_direct_commits.md](feedback_no_prs_direct_commits.md))
is to land substantial, testable features. A SimX-only RT extension
without RTL does not meet that bar.

## 8. Appendix — file map

```
sim/simx/
  rt_core.{h,cpp}    # functional engine (BVH walk + intersection)
  rt_unit.{h,cpp}    # per-core wrapper, shader_states_, DCR plumbing
  rt_sim.{h,cpp}     # cycle-level timing engine (warp buffer + mem ports)
  rt_trace.{h,cpp}   # per-warp telemetry, mem-access deques, latency dist
  types.h            # ShortStack, ShaderQueue, RtuReq, RtuRsp, RtuMemAdapter
  execute.cpp:1594   # EXT_RTU_ENABLE branch in executor
  decode.cpp:1248    # macro-op expansion of TRACE → 4 micro-ops
  core.cpp:163-223   # RTU dispatcher + dcache arbiter wiring
  func_unit.{h,cpp}  # thin RtuUnit forwarder

hw/
  VX_config.toml:244 # NUM_RTU_*, RT_BVH_*, RT_*_LATENCY knobs
  VX_types.toml:16   # VX_DCR_BASE_RTX_* DCR ids
  VX_types.toml:271  # VX_RT_* attribute ids + commit actions

kernel/include/
  vx_raytrace.h      # trace_ray, get_work, get_attr, set_attr, commit asm

tests/regression/raytracing/
  main.cpp tracer.cpp scene.cpp scene_list.cpp bvh.cpp mesh.cpp surface.cpp
  common.h           # device-host shared structs (cwbvh_node_t, blas_node_t)
  kernel.cpp         # producer/consumer kernel
  shaders/
    miss.cpp closet-hit.cpp intersection.cpp any-hit.cpp
    sphere.h cornellbox.h bunny.h sponza.h spring.h rtiow.h
```
