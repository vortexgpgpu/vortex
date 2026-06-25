# v2 Review: prism_v3 RTU SimX implementation
**Date:** 2026-06-17
**Scope:** `sim/simx/rtu/` — rtu_core.cpp, rtu_unit.cpp, rtu_walker.cpp, rtu_isect.{h,cpp}, rtu_memory.cpp, rtu_classifier.{h,cpp}, rtu_bvh.h, rtu_types.h
**Method:** read-only; cross-referenced against `hw/rtl/rtu/*.sv`, `rtu_simx_proposal.md`, `rtu_isa_v2_proposal.md`. `rtu_smoke` rebuilt + `make run-simx` → PASSED. No code modified.

---

## 1. Overall assessment (maturity grade: **B**)

The SimX RTU is a genuinely strong *functional* oracle and a structurally honest TLM. The async SlotPool FSM (ISSUE/RESERVED/AWAIT/COMPUTE/IN_QUEUE/RESP/EMITTED) is clean, the macro-op uop generator faithfully implements the ISA v2 window ABI (TRACE2 4-uop split, GETW/GETWF per-slot windowed reads), and the refactor into layered files (types → isect → classifier → walker → memory → unit → core) is well-factored, single-responsibility, and SystemC-translatable as the comments claim. Memory really flows through the cluster dcache with 1-deep drain/issue; backpressure is real; the recent `VX_CFG_RTU_NUM_CTX` pool-sizing fix and the `kRtuSetupLatency=17` charge are correct and bring the pool/back-pressure and per-ray setup span into parity.

What holds it back from an A is **timing parity of the PE cost model**. The single largest issue is not the two gaps named in the brief (transform latency, stack-stall) — it is that the BoxPe/TriPe `cycles_for` latencies the model charges (`NODE_LATENCY=4`, `TRI_LATENCY=6`) are roughly an order of magnitude below the *actual* RTL pipeline depths (box PE = 31 cycles, tri PE = 91 cycles), and the model assumes a 4-wide *parallel* PE array while the RTL instantiates exactly **one** shared box PE and **one** shared tri PE streaming one primitive per cycle across all contexts. So the cost model is internally consistent and tunable, but its defaults are fiction relative to the HW it is meant to predict. Functionally an A-; as a cycle-approximate timing oracle, a C+ — net **B**.

Correctness of the traversal/intersection math is good but not watertight (ray-AABB divides by `rd[i]` with no inf-handling; Möller-Trumbore is the non-watertight single-sided variant), which matters for the stated "true GPU" goal where watertightness is table-stakes.

---

## 2. Correctness findings

### C1 — `ray_aabb_intersect` divides by zero on axis-aligned rays — **Medium**
`rtu_isect.cpp:58` computes `inv = 1.0f / rd[i]` with no guard. The header (`rtu_isect.h:57-59`) explicitly documents "Assumes well-conditioned rays (no axis-aligned ray with zero direction component)." Axis-aligned rays (shadow rays straight down a coord axis, AABB face grazing) are common in real scenes and produce `±inf`/`NaN`. The standard robust slab test (precompute `1/d`, swap by sign, use `±FLT_MAX` clamps) is what NVIDIA/Intel HW does and what the RTL's reciprocal pipeline already produces. The walker passes `ctx.best_t` as `tmax` so a NaN comparison silently mis-prunes. Not a test failure today (fixtures are well-conditioned) but a correctness landmine for any production scene.

### C2 — Möller-Trumbore is not watertight — **Medium (design-goal gap)**
`ray_triangle` (`rtu_isect.cpp:19-50`) is the classic single-sided MT with a fixed absolute `EPS=1e-6` on `det` and exact-zero edge tests (`u<0 || u>1`, `u+v>1`). This leaks/double-counts at shared triangle edges — the exact artifact watertight intersection (Woop et al., used by NVIDIA RT core and embree) was designed to eliminate. The brief lists watertightness as an explicit eval criterion and "true GPU" alignment; the RTL `VX_rtu_tri_pe` uses the same MT formulation, so SimX and RTL *agree*, but both diverge from shipping HW. Acceptable for an oracle whose RTL twin matches it, but should be called out as a known fidelity ceiling, not silently inherited.

### C3 — `affine_inverse_transform_ray` recomputes a full 3×3 inverse per instance per ray — **Low (correctness OK, see E1 for cost)**
`rtu_isect.cpp:69-108` is mathematically correct (cofactor adjugate / det, singular→identity passthrough). Two concerns: (a) the singular-matrix fallback to *identity* silently produces wrong object-space rays rather than skipping the instance — a degenerate instance transform will report bogus hits instead of a miss; (b) the header (`rtu_isect.h:75-77`) notes non-uniform scale would require renormalising `hit_t` and declares it out of scope — but the walker treats the BLAS `hit_t` as the world `hit_t` unconditionally (`rtu_walker.cpp:491` comment "the t parameter is preserved"), so any scaled instance returns a wrong distance. Vulkan permits arbitrary affine instance transforms incl. scale; this is a real correctness gap for scaled instances, currently masked because fixtures use rotation+translation only.

### C4 — TLAS `hit_instance_id` reports the loop index, not the instance's assigned ID (flat walker) — **Low**
In `FlatWalker::walk_lane` (`rtu_walker.cpp:544`, `565`) `best_instance`/`yield_instance` are set to `inst_idx` (the loop counter), whereas the BVH4 walker correctly uses `inst->instance_id` (`rtu_walker.cpp:235`). The flat-list TLAS path never reads the instance record's `instance_id` field (offset is defined but unused). For scenes where instances are not laid out in ID order, `gl_InstanceID` is wrong on the flat path. BVH4 path is correct.

### C5 — `wait_handle` assumes one handle for the whole warp — **Low (documented Phase-1 limitation)**
`rtu_unit.cpp:79-90` reads only the first active lane's `rs1` as the canonical WAIT handle, and `free_slot` frees that single slot. This is correct *only* because TRACE2 allocates one pool slot per warp (`process_trace2_uop` uop 0 allocates one slot covering all lanes). It is internally consistent with the current allocator but means the "per-(warp,lane)" concurrency described in proposal §4.1 is really per-warp in practice — divergent per-lane handles would break. Worth an assert that all active lanes carry the same handle.

### C6 — `ray_triangle` tmax cull uses stale `tmax` in flat walker, allowing redundant work but not wrong results — **Informational**
`rtu_walker.cpp:531-533` deliberately tests against `s.req.tmax[t]` (not `best_t`) so an opaque hit doesn't pre-cull a non-opaque candidate. Correct for AHS semantics, but combined with `TERMINATE_ON_FIRST_HIT` shrinking `s.req.tmax[t]` in place (`rtu_walker.cpp:555`) it mutates the request packet's tmax — benign here (single walk) but a foot-gun if a slot is ever re-walked (callback resume re-enters COMPUTE; the `walk_done` latch prevents re-walk today, so safe — but fragile).

### C7 — Octant signature sampled from first active lane only — **Informational**
`rtu_core.cpp:463-470` derives the slot's 3-bit coherency signature from `first_active` lane's direction. For a divergent warp (lanes spanning multiple octants) this mislabels the slot. The proposal (§5.3) intends per-context signatures; with one slot per warp the per-lane octant divergence is lost. Coherency-gather is advisory (perf only), so no correctness impact, but it weakens the modelled benefit (see P3).

---

## 3. Efficiency findings (model complexity)

### E1 — Per-instance affine inverse recomputed per ray — **Low**
`affine_inverse_transform_ray` does a full adjugate+det every call (`rtu_isect.cpp:69-108`), invoked once per (ray, instance) in both walkers. Real HW (and the RTL `VX_rtu_xform`) consumes a *pre-inverted* or directly-applied object→world matrix; the inverse is a build-time or instance-load-time computation, not per-ray. For the oracle this is only host wall-clock, not modelled cycles, but it both (a) misrepresents where the cost lives and (b) is the thing the unmodeled 36-cycle transform latency (P1) should be charging for. Fold the two: when you add the transform-latency charge, also note the inverse is amortizable.

### E2 — `compute_intersections` two-pass octant scan is O(2·NUM_CTX) every tick — **Low**
`rtu_core.cpp:492-583` scans all slots twice per tick (matching-signature pass then non-matching). With `NUM_CTX = NUM_THREADS` (e.g. 32) that is 64 slot-visits/tick, most skipped via the `state != COMPUTE` guard. Cheap in absolute terms, but the two-pass structure exists only to reorder *which* COMPUTE slot updates `last_compute_signature_` first — and since every COMPUTE slot is advanced in the same tick regardless of pass, the coherency "preference" only affects the signature-history counter, not actual scheduling or memory issue order. The pass split buys a perf-counter nicety at 2× scan cost; a single pass that tracks the best-match slot would be equivalent and half the work.

### E3 — `read_scene_bytes` byte-at-a-time copy across lines — **Informational**
`rtu_walker.cpp:40-49` copies byte-by-byte with a div/mod per byte. Called for every node/leaf/tri field read during the (functional, one-tick) walk. Host-only cost, but for the BVH path on a few-hundred-tri mesh this is the hot loop. A line-spanning `memcpy` fast path (common case: read fits in one line) would cut it dramatically. Not parity-affecting.

### E4 — `kRtuMaxLinesPerLane` × `MEM_BLOCK_SIZE` per-lane line buffer is large — **Informational**
`LaneState` (`rtu_types.h:391`) embeds `line_data[kRtuMaxLinesPerLane][64]`. With `kRtuMaxBvhSceneBytes=16384` that is ~258 lines × 64 B × NUM_THREADS lanes × NUM_CTX slots of POD per RtuCore — a multi-MB `Slot` array. This is the pre-fetch-whole-BVH shortcut the comments flag (§8.5.1 demand-fetch is the fix). It inflates the SimX memory footprint and, more importantly, models a pre-fetch that the RTL does NOT do (RTL demand-fetches nodes mid-walk), so memory-traffic counts diverge (see P5).

---

## 4. Performance / RTL-parity findings

### P1 — **BoxPe/TriPe cost-model latencies are ~10× too low and assume parallelism the RTL doesn't have — Critical**
`rtu_isect.cpp:113-127`:
```
BoxPe: cycles = ceil(n/VX_CFG_RTU_BOX_PE=4) + VX_CFG_RTU_NODE_LATENCY=4 - 1
TriPe: cycles = ceil(n/VX_CFG_RTU_TRI_PE=4) + VX_CFG_RTU_TRI_LATENCY=6 - 1
```
The actual RTL pipeline depths (computed, not from these knobs):
- `VX_rtu_box_pe.sv:76`: `LATENCY = LAT_ORIGIN+LAT_DEQUANT+LAT_SLAB+LAT_MINMAX+LAT_REDUCE+LAT_CMP = 9+9+9+1+2+1 = ` **31 cycles**
- `VX_rtu_tri_pe.sv:68`: `LATENCY = 8*F + V + 2 = 8*9 + 17 + 2 = ` **91 cycles**

Two compounding errors:
1. **Latency magnitude.** SimX charges 4/6; RTL is 31/91. A ray that does, say, 20 box tests + 4 tri tests gets charged `~23` cycles in SimX vs a pipeline whose per-primitive *fill* dominates very differently in RTL. This is the dominant traversal cost and it is off by ~10×.
2. **Width / sharing.** SimX divides by `VX_CFG_RTU_BOX_PE=4` (a 4-wide parallel array) AND charges every slot independently with no cross-slot contention (`rtu_isect.h:106` "Cross-slot PE contention is NOT modelled"). The RTL scheduler instantiates **exactly one** `box_pe` (`VX_rtu_scheduler.sv:390`) and **one** `tri_pe` (`:407`), shared across **all** NUM_CTX contexts, fed **one child per cycle** via `CS_FEED`/`feed_idx` (`:737-741`). So real throughput is 1 box/cycle *cluster-wide*, pipelined; SimX models 4/cycle *per slot* with unlimited slots. Under a full 32-context warp this over-states box/tri throughput by up to `4 × NUM_CTX`.

Worse, the `VX_CFG_RTU_BOX_PE` / `NODE_LATENCY` / `TRI_LATENCY` knobs are **dead in the RTL** — `grep` confirms no `hw/rtl/rtu/*.sv` references them (corroborated by `rtu_rtl_efficiency.md` Opt 6 retiring them). So the SimX cost model is parameterized entirely by knobs the hardware ignores. **This is the headline parity defect.** Recommendation in §6 (P0).

### P2 — **Per-instance transform latency (36) unmodeled — High (confirmed)**
Confirmed as stated in the brief. `VX_rtu_xform.sv:62`: `LATENCY = 4*F = 4*9 = ` **36 cycles**, charged per instance descent in the RTL (`CS_XFORM`→`VX_rtu_xform`, `VX_rtu_scheduler.sv:418-422`). SimX `compute_intersections` charges only `BoxPe::cycles_for + TriPe::cycles_for + setup` (`rtu_core.cpp:530-538`) — no transform term. The instance-descent counter already exists (`perf.bvh_instance_descents`, incremented at `rtu_walker.cpp:232`), so the fix is mechanical: `cycles += 36 * (instance_descents_delta)`, mirroring the box/tri delta pattern already in `compute_intersections`. **Confirmed P1-class recommendation; the counter to drive it is already present** — this is lower-effort than the brief implies. `VX_CFG_RTU_XFORM_LATENCY` is indeed dead RTL config; hard-code `4*RTU_LATENCY_FMA=36` as `rtu_types.h:242` already documents.

### P3 — **Short-stack spill/restart behavior unmodeled — Medium (confirmed)**
Confirmed. The SimX walker uses a fixed `kBvhStackCap=16` array (`rtu_walker.cpp:240-242`) and **silently truncates** deeper sub-trees (`:319` `if (stack_top < kBvhStackCap)` drops the push). The RTL (`VX_rtu_scheduler.sv:288, 752`) gates pushes on `sp_q != RTU_STACK_DEPTH` — but a real short-stack design (`VX_CFG_RTU_STACK_DEPTH`, the proposal's §5.2 "short stack" + §8.5.1 trail-based RESTART) handles overflow by *restarting* traversal from a trail marker, which costs re-descent cycles, not by dropping nodes. So SimX has two divergences: (a) **functional** — it drops nodes the RTL would re-visit, potentially *missing* a closest hit in a deep BVH (correctness, not just timing); (b) **timing** — no spill-stall / re-descent cycles charged. The brief flags only the stall; the silent functional truncation is the more serious half. For the current shallow fixtures neither bites, but `kBvhStackCap=16` hard-coded (ignoring `VX_CFG_RTU_STACK_DEPTH`) means SimX and RTL can disagree on *results* for a tall BVH. Recommendation: drive the cap from `VX_CFG_RTU_STACK_DEPTH` and model trail-RESTART (P1 in §6).

### P4 — Coherency-gather benefit is modeled as a counter only, not as memory reuse — **Low**
`compute_intersections` updates `coherency_hits/misses` (`rtu_core.cpp:505-506`) based on octant-signature match, but the match does **not** change memory issue order, dcache hit rate, or cycle cost — it only reorders which COMPUTE slot touches `last_compute_signature_` first (see E2). The proposal (§5.3) intends same-octant rays to share L1 fetches; SimX charges identical memory cost regardless of coherence. So the headline NVIDIA-CGU-style win (BVH-fetch L1 reuse) is observable in a counter but absent from the timing model. Parity-neutral vs the RTL *if* the RTL also doesn't reorder (the RTL scheduler picks runnable contexts by fixed priority, `VX_rtu_scheduler.sv:237`, not octant) — so SimX and RTL actually agree that coherency-gather is a no-op today. Flag: the *proposal's* claimed benefit is unmodeled in both, which is fine for parity but means the architectural value is unvalidated.

### P5 — Whole-BVH pre-fetch vs RTL demand-fetch inflates `mem_reads` — **Medium**
`rtu_memory.cpp:139-152` pre-fetches `scene_bytes` (whole serialized structure, up to 16 KB) for every lane on the BVH path. The RTL demand-fetches nodes as the walk reaches them (`VX_rtu_scheduler.sv` `CS_FETCH`/line RAM per visited node). For a ray that prunes 90% of the tree, SimX issues ~10× the memory requests and ~10× the line-fill cycles the RTL would. `perf.mem_reads` and the memory-pipeline cycle accounting therefore over-count badly on any non-trivial BVH. The comments (`rtu_types.h:272-277`, `rtu_memory.cpp:146`) honestly flag demand-fetch (§8.5.1) as the HW-faithful fix. This is the second-biggest timing-parity gap after P1 and is *functional-adjacent* (it changes the modeled dcache traffic the dispatcher sees).

### P6 — Setup latency charged once per ray is correct; verify it isn't also re-charged on callback resume — **Informational (looks correct)**
`kRtuSetupLatency=17` is charged under the `!s.setup_charged` latch (`rtu_core.cpp:535-538`), and `setup_charged` is reset only on slot free/alloc (`rtu_types.h:427`, `rtu_core.cpp:90`). Callback resume re-enters via CB_ACTION → RESP (not back through COMPUTE re-walk, since `walk_done` stays set), so the 17 is charged exactly once per ray. Matches RTL `SETUP_LAT=RTU_FDIV_LAT=17` (`VX_rtu_scheduler.sv:91`) which is the one-time 1/dir reciprocal span. **This recent fix is correct.** Good.

### P7 — Single mem port; no NUM_RTU_BLOCKS=2 outstanding modeling — **Low**
`RtuCore` ctor wires `dcache_req_out(1)` / `dcache_rsp_in(1)` (`rtu_core.cpp:695-696`) and `MemoryEngine` issues/drains one per tick. Proposal §5.6 specifies `NUM_RTU_BLOCKS=2` for two outstanding requests to keep the box-PE array fed. With a single port the modeled memory pipeline is narrower than the proposal's target; whether the RTL `VX_rtu_mem` is 1- or 2-wide should be checked to set the port count to match (not verified here).

---

## 5. "True GPU" alignment vs NVIDIA / AMD / Intel

**Where PRISM-SimX aligns well:**
- **Register-resident ray state + windowed ISA (TRACE2/GETW/GETWF)** is a faithful analog of NVIDIA's compiler-allocated ray descriptor registers and Intel's Ray Bank — ray inputs/hit attrs never touch the dcache (`rtu_unit.cpp` regfile), only BVH/tri data does. This is the correct shape and a genuine strength.
- **Async trace + reformation queues** (`ReformationEngine`, `rtu_core.cpp:154-263`) model HW warp reformation in the spirit of Intel BTD / NVIDIA SER — grouping yielded lanes by `(warp_id, sbt_idx)` so the callback executes coherently. The same-warp serialization gate (`warp_cb_inflight_`) correctly models the trap-CSR single-occupancy constraint.
- **CW-BVH4/6 + procedural-AABB + TLAS-instance leaf decode** matches the Intel Xe-HPG / Mesa `vk_bvh.h` public-format direction.

**Where it diverges from shipping HW (and the gaps matter for the perf-competitive goal):**
- **Box/tri test rates.** NVIDIA RT core does ~4 box tests/clk and 1 tri/clk *with deep pipelines*; AMD RDNA ray accelerator does 4 box or 1 tri/clk per CU; Intel does similar. The PRISM **RTL** is a single shared box PE + single tri PE at 1 prim/cycle (no 4-wide box array) — already a throughput design point below the references. The **SimX model** then over-states it further by assuming a 4-wide parallel array (P1). So neither the RTL nor (especially) the SimX model currently reflects a competitive box-test rate, and the SimX model can't be used to *evaluate* widening the box PE because its cost knobs are dead in RTL.
- **Watertight traversal.** Shipping HW uses watertight ray-tri (Woop); PRISM uses single-sided MT (C2). For path-traced production scenes this produces visible cracks. Both SimX and RTL share the flaw, so it's a *design* gap, not a parity gap.
- **Short-stack / restart.** Modern RT cores use a short stack + restart trail to bound SRAM (NVIDIA, Intel both do this). PRISM RTL has the short stack; SimX **silently truncates** instead of restarting (P3) — so SimX does not model the re-descent cost that is the whole reason short-stack designs exist, and can miss hits a real short-stack would find via restart.
- **Coherency gather / SER.** Modeled as a counter but with zero timing effect (P4); the actual L1-reuse / reorder benefit that NVIDIA CGU and SER deliver is unvalidated in the model. This is the headline "long-tail incoherent secondary ray" win the proposal (§1.1) promises to close, and the model currently can't show it.

Net: the **architecture** is well-aligned with Intel Xe-HPG; the **SimX timing model** is not yet faithful enough to use as the perf-competitiveness oracle the proposal's objective #1 demands, primarily because of P1/P2/P5.

---

## 6. v2.1 recommendations

### P0 — must fix before SimX can be trusted as a timing oracle
1. **Re-base BoxPe/TriPe `cycles_for` on the real RTL pipeline depths (P1).** Replace the dead `VX_CFG_RTU_NODE_LATENCY=4` / `TRI_LATENCY=6` knobs with the computed RTL values: box PE = 31 (`9*3 + 1 + 2 + 1`), tri PE = 91 (`8*9 + 17 + 2`), expressed symbolically from `RTU_LATENCY_FMA=9` / `RTU_FDIV_LAT=17` so they track. Add a comment cross-referencing `VX_rtu_box_pe.sv:76` / `VX_rtu_tri_pe.sv:68`.
2. **Model the single shared box/tri PE, fed 1 prim/cycle, with cross-context contention (P1).** Drop the `÷ BOX_PE=4` parallelism. The RTL has ONE box PE + ONE tri PE per RtuCore streaming one primitive per cycle across all NUM_CTX contexts. Either (a) charge `n_tests` issue cycles (1/cycle) + one drain of LATENCY, and serialize the PE across slots via a shared per-tick PE-busy budget, or (b) at minimum stop dividing by a width the HW doesn't have. This is the difference between the model being directionally right vs ~`4×NUM_CTX` optimistic on traversal throughput.
3. **Charge the 36-cycle per-instance transform latency (P2).** In `compute_intersections`, capture `instance_descents` delta exactly like the box/tri deltas and add `cycles += 36 * delta` (36 = `4*RTU_LATENCY_FMA`, per `VX_rtu_xform.sv:62`). The counter already exists; this is a ~5-line change.

### P1 — parity-significant, do next
4. **Stack: honor `VX_CFG_RTU_STACK_DEPTH` and model restart, not silent truncation (P3).** Replace `kBvhStackCap=16` with the config knob, and on overflow model trail-based RESTART (re-descend cost) instead of dropping pushes — this fixes both a timing gap *and* a potential missed-closest-hit correctness gap on deep BVHs.
5. **Move toward demand-fetch on the BVH path (P5).** The whole-BVH pre-fetch over-counts `mem_reads` and line-fill cycles by the tree's prune ratio. Even an interim "fetch only nodes the walk visits" (recording visited offsets during the functional walk, then issuing exactly those lines) would close most of the gap and shrink the multi-MB `Slot` footprint (E4).
6. **Robust ray-AABB slab test (C1).** Precompute `1/d`, sign-swap, clamp with `±FLT_MAX`; eliminates the divide-by-zero on axis-aligned rays. Matches what the RTL reciprocal pipeline already feeds.

### P2 — correctness hardening / cleanups (lower urgency)
7. **Scaled-instance `hit_t` renormalization (C3).** Either renormalize `hit_t` by the transform's directional scale at instance exit, or document+assert rotation+translation-only. Today scaled instances silently report wrong distances.
8. **Flat-walker `hit_instance_id` should read `instance_id` from the record, not the loop index (C4).** Align with the BVH4 path.
9. **Singular-transform should skip the instance, not pass through as identity (C3).** A degenerate transform currently fabricates hits.
10. **Single-pass coherency scan (E2) + line-spanning `read_scene_bytes` fast path (E3).** Host-perf only, but removes the 2× slot scan and the byte-at-a-time copy in the BVH hot loop.
11. **Assert one-handle-per-warp in `wait_handle` (C5)** so the per-warp allocator assumption is enforced rather than implicit.
12. **Document watertightness ceiling (C2)** as a known SimX↔RTL-agreed deviation from shipping HW, so it isn't mistaken for a parity bug.

---
*All file:line references verified against the tree at review time. `rtu_smoke` rebuilt and `make run-simx` → PASSED; no regressions introduced (read-only review).*
