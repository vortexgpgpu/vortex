# v2 Review: prism_v3 graphics SimX implementation
**Date:** 2026-06-17
**Reviewer scope:** sim/simx/raster/, sim/simx/tex/, sim/simx/om/, gfx dispatch in sim/simx/sfu_unit.cpp, graphics decode in sim/simx/decode.cpp, host SW mirror sw/common/gfx_render.*

## 1. Overall assessment (+ maturity grade)

**Grade: B / B+.** The three fixed-function SimX models (RASTER coarse-bin walker, vx_tex4 integer-mip TMU, vx_om4 windowed ROP) are well-structured, readable, and functionally correct for the single-draw oracle case — the test `gfx_pipeline_tex` confirms `image device vs host-Binning: PASS` and `image batched-draw vs host-Binning: PASS`. The producer/consumer FSMs are clean, the rcache/ocache/tcache traffic is modeled as real MemReq/MemRsp through the cache hierarchy (good parity intent), and the RASTER TE/BE walker faithfully reproduces the RTL's priority-FIFO + bypass quad-emission order (raster_core.cpp:586-666) rather than the simpler recursive Morton DFS in `gfx_render.cpp` — a deliberate parity choice that is correctly documented.

The grade is held below A by: (1) the open §8 multi-draw determinism failure, which I reproduced, spatially localized, and root-caused to an **unscoreboarded SFU→OM shared-window handoff** (a following fragment's `SETW` can overwrite the per-(warp,lane) window mid-way through the prior fragment's async `vx_om4` sub-pixel reads) — confirmed *not* a cache, reset, or OM-concurrency bug by experiment; (2) a genuine timing-parity gap — the FF cores are fully reset on every launch, so cross-draw FF pipeline occupancy is not modeled; (3) several minor correctness/robustness issues listed below. The SimX↔SW-oracle agreement is strong; the SimX↔RTL **timing** parity is weaker than the charter implies.

I reproduced §8 deterministically (146 bytes, identical across 2 runs):
```
*** one-batch frame vs two-batch frame differs in 146 bytes
depth multi-draw frame one-batch vs two-batch: FAIL (depth gated later draw: yes)
```

## 2. Correctness findings

### §8 root cause (this is the headline finding — empirically localized and confirmed)

**`sim/simx/sfu_unit.cpp:318-358` (vx_om4 sub-pixel sequencer reading the shared window) + `sim/simx/decode.cpp:1003-1010` (vx_om4 has no rd) + `sim/simx/decode.cpp:1043-1050` (SETW has no rd) — the vx_om4 quad submit reads the shared per-(warp,lane) graphics window across 4 cycles with no scoreboard/double-buffer protection against the next fragment's SETW overwriting it — SEVERITY: HIGH. This is the §8 bug.**

What §8 actually is (I instrumented the test to localize it, see Appendix):
- The diff is **73 scattered pixels across the whole frame** (bbox [4,12]..[121,119]), **not** an edge at the near/far seam. CBUF-diff pixels ≡ ZBUF-diff pixels (same count, same bbox) → the colour change is purely a depth-gating cascade; the **primary divergence is in the depth buffer**.
- At the diff pixels the depth flips between `0xffffffff` (clear value — far quad's depth write *did not* land) and `0xff800000` (far quad's depth *did* land). So the far full-screen quad's OM depth writes reach a **different set of pixels** in one-batch vs two-batch. It is a fragment-payload / fragment-loss pattern, fully deterministic (identical 73 pixels across repeated runs).

Why the SFU window race is the cause:
- The fragment kernel stages each fragment's colour/depth into the shared graphics window via `SETW` (window slots `OM_WIN+0..7`), then issues `vx_om4`, which reads those slots back. The window is a single per-`(warp_id, lane, slot)` register file (`RtuUnit::regfile_`, rtu_unit.h:174-186) — **no double buffering, no per-fragment versioning.**
- `vx_om4` carries **no destination register** (decode.cpp:1003-1010), and `SETW` carries **no destination register** (decode.cpp:1043-1050). Neither participates in the scoreboard. So the *next* fragment's `SETW` writes are **not** ordered against the *previous* fragment's still-in-flight `vx_om4`.
- The SFU sub-pixel sequencer issues one of the 4 sub-pixels per cycle and reads the window each cycle inside the loop (`rtu_unit_->window_get(...)` at sfu_unit.cpp:343-344, with `continue` re-entering once per cycle at line 349). The 4 reads are spread across ≥4 SFU cycles. If a following fragment's `SETW` (same warp/lane, same `OM_WIN` slots) lands between sub-pixel reads — which is *not* prevented, since nothing scoreboards either op — the OM reads a **mix of two fragments' depth/colour**, corrupting (or dropping, via the depth test) the write.
- The exact interleaving of "next SETW" vs "prior vx_om4 sub-pixel reads" depends on upstream pipeline timing (warp issue cadence, TEX async-completion timing, RASTER feed rate), which differs between the one-batch and two-batch submission shapes. Hence the **deterministic-but-path-dependent** 73-pixel scatter.

Ruled out by code trace + experiment:
- **FF-core / cache / reset / DCR layer is fully symmetric.** Every launch goes `vortex_start → processor_.run()` (vortex.cpp:154-156) → `reset() → SimPlatform::reset()` (processor.cpp:222,248-249), so OM/TEX/RASTER **and the window regfile** are reset every launch; a `CMD_CACHE_FLUSH` retires synchronously after every launch in both paths (device.cpp:494-508; cmd_processor.cpp:408-418; vortex.cpp:147-149). None of these differ between the two paths.
- **OM-internal same-pixel concurrency is NOT the cause.** I rebuilt with `VX_CFG_OM_MEM_QUEUE_SIZE=1` (forcing OM slots to serialize) and the diff was **unchanged at exactly 146 bytes** — proving the divergence enters the OM already different, i.e. it is upstream in the SFU window handoff, not in OmCore's R-M-W. (An OmCore same-pixel R-M-W interlock is still a *separate* fidelity gap worth having — see §2 secondary — but it does not explain §8.)

**Fix direction:** the window handoff between `SETW` and the async `vx_om4` must be ordered. Options, in rough order of fidelity: (a) give `vx_om4` a scoreboard handle (a dummy rd or an explicit window-slot dependency) so a following `SETW` to the same slots stalls until the OM submit has captured the window — this mirrors how `vx_tex4` uses its rd as a "scoreboard sync handle" (tex_core.cpp:199); (b) snapshot all 4 sub-pixels' window payload into the trace at sub-pixel F==0 instead of reading the window once per cycle (sfu_unit.cpp:331-345), so a mid-sequence SETW cannot perturb sub-pixels 1..3; (c) double-buffer / version the OM window. Option (b) is the smallest correct SimX change and removes the per-cycle window dependency entirely. **Confirm with the instrumented build (Appendix): the 73-pixel diff should go to 0.** Note the RTL must enforce the same ordering or it will mismatch — this is a shared parity requirement.

### Other correctness findings

- **`sim/simx/om/om_core.cpp:93-99, 197-223, 524` — OmCore has no same-pixel R-M-W interlock — SEVERITY: MEDIUM (fidelity gap, NOT the §8 cause).** `OmCore` runs up to `kInflight = VX_CFG_OM_MEM_QUEUE_SIZE = 4` slots concurrently; `drain_req_in` admits any free slot with no check that an in-flight slot targets the same pixel. Two same-pixel fragments can both READ stale depth before either WRITES, and the last write wins by slot scheduling — a real ROP serializes same-(x,y) fragments. The `VX_CFG_OM_MEM_QUEUE_SIZE=1` experiment proves this is **not** what breaks §8 (diff unchanged), but it remains a latent ordering inaccuracy that should be fixed for correctness on workloads with same-pixel overdraw. Mirror the RTL ROP's per-pixel ordering.

- **`sim/simx/om/om_core.cpp:482-483` — duplicated assignment `mreq.op = MemOp::ST;` (written twice) — SEVERITY: LOW (cosmetic/dead).** Harmless but signals a copy/paste slip; clean up.

- **`sim/simx/om/om_core.cpp:434-453, 464-469 / WRITE_ISSUE` — `any_pending`/`all_done` logic is redundant and slightly fragile — SEVERITY: LOW.** The first loop sets `any_pending=true`, the re-scan recomputes `all_done`, and the final branch `if (!any_pending && all_done) DONE; else if (all_done) DONE;` collapses to "if all_done DONE". The `any_pending` variable is dead. Same redundant shape in `advance_read_issue` (om_core.cpp:289-302) — the `all_issued` re-scan duplicates the in-loop bookkeeping. Works, but invites future bugs.

- **`sim/simx/raster/raster_core.cpp:200-219 / issue_pending_loads` — `pending_count_` underflow guard is load-bearing but the FSM trusts `pending_count_==0` to advance — SEVERITY: LOW.** A dropped/duplicated rcache tag (drain_mem_rsp silently `continue`s on unknown tag, raster_core.cpp:229-231) would desync `pending_count_` from `pending_reads_`. Since `advance_producer` gates state transitions on `pending_count_==0` (raster_core.cpp:259/264/269), a desync stalls the producer forever. Prefer gating on `pending_reads_.empty()` (the authoritative set) rather than a parallel counter.

- **`sim/simx/tex/tex_core.cpp:191-194 / advance_addr` — trilinear `lod1` clamp uses `VX_TEX_LOD_MAX` as both the comparison bound and the clamped value — SEVERITY: LOW (verify).** `lod1 = (lod0+1 < VX_TEX_LOD_MAX) ? lod0+1 : VX_TEX_LOD_MAX`. The host mirror (gfx_render.cpp:330) uses the identical formula, so SimX↔oracle is consistent; just confirm `VX_TEX_LOD_MAX` is an inclusive max LOD index and not a count (off-by-one risk at the smallest mip). Consistency with the oracle means any error is shared, not a parity gap.

- **`sim/simx/om/om_core.cpp:409-415 / color write when `color_read_==false`** — comment acknowledges dst_color is unread and the merge is a no-op (writemask 0xf). Correct, but `l.dst_color` is then an uninitialized-by-read `0` and `c_write_value = (0 & ~0xffffffff)|(blended & 0xffffffff)` = blended. Fine. Just flagging that the "merge is a no-op" only holds because `cbuf_writemask_==0xffffffff` in that branch; the invariant is implicit.

- **`sim/simx/sfu_unit.cpp:336-345 / vx_om4 sub-pixel decode** — `qy` is masked with `((1u << (VX_RASTER_DIM_BITS-2)) - 1)` (sfu_unit.cpp:338) while `qx` uses `VX_RASTER_DIM_BITS-1` (sfu_unit.cpp:337). The descriptor packing in raster_core.cpp `encode_pos_mask` uses `kPosBits = VX_RASTER_DIM_BITS-1` for **both** x and y (raster_core.cpp:48-53). The vx_om4 unpack masking y to one fewer bit is asymmetric with how the raster side packs it — for 128px bins this does not overflow, but it is an inconsistency that will bite at larger `VX_RASTER_DIM_BITS`. SEVERITY: LOW (latent).

- **`sw/common/gfx_render.cpp:43, 60, 79, 280, 307 / `assert(false)` in `default:` of format/wrap/filter switches** — these abort the simulator on an unsupported enum rather than returning a defined value. Acceptable for a model, but a malformed DCR would crash simx instead of producing a diagnosable mismatch.

## 3. Efficiency findings

- **RASTER runs the full rasterizer synchronously in one tick** (raster_core.cpp:272-278, `run_rasterizer` walks every tile×prim and fills `quad_queue_` in a single `advance_producer` call). This is simple and fast but means the RASTERIZE "stage" has zero modeled latency regardless of triangle count — fine for functional correctness, poor for timing fidelity (see §4).

- **`prim_data_` is an `unordered_map<uint16_t, rast_prim_t>`** (raster_core.cpp:741) re-built per frame with per-pid `std::unordered_map<uint16_t,bool> seen` dedup (raster_core.cpp:360-373). For the small triangle counts in tests this is negligible; for real workloads the hash-map churn and the `primary_pids_` linear dedup are O(pids²)-ish. A sorted vector + `std::unique` would be both faster and more deterministic.

- **OM/TEX per-lane `LaneState`/`Slot` arrays are zero-reinitialized (`l = LaneState{}`) on every accept** (om_core.cpp:216, tex_core.cpp:166-169). Cheap, but the `filled = {}` and full-struct reset run for inactive lanes too. Minor.

- **`enqueue_byte_range` coalesces within a cache line but issues one MemReq per line with no cross-request dedup** (raster_core.cpp:171-187). If two pids share a prim cache line (adjacent pids, 32-byte `rast_prim_t`), the loader fetches the line twice. The rcache absorbs it (hit), so it is a modeled-traffic inflation, not a correctness issue — but it overcounts `perf_stats_.mem_reads`.

## 4. Performance / RTL-parity findings

- **No cross-launch FF pipeline state — the biggest parity gap.** Because `processor_.run()` resets `SimPlatform` (and thus every FF SimObject) at the start of every launch (vortex.cpp:154-156 → processor.cpp:248-249), the OM/TEX/RASTER pipelines start **empty** for each draw. Real hardware keeps these units' queues warm across draws in a frame. The model therefore cannot represent inter-draw FF overlap or back-pressure carryover. For a single-launch kernel this is invisible; for the multi-draw frames the charter targets it is a structural timing inaccuracy. (It is *also* what makes §8's cache analysis a red herring — the reset is total.)

- **RASTER quad production has no rate model.** All quads for a frame are enqueued in one tick (raster_core.cpp:272-278); the consumer then drains at "one request → one response per cycle, one stamp per active lane" (raster_core.cpp:708-729). So the *consumer* side models a quad/cycle rate, but the *producer* (the TE/BE walker that the comments go to great lengths to make order-accurate) contributes **zero cycles**. The carefully-modeled priority-FIFO/bypass ordering affects only the *order* of quads, never the *timing*. If RTL parity for RASTER throughput matters, the walker should advance one pipe-stage per tick (it is already written as a 2-stage pipeline with FIFOs — it just runs to completion in a loop at raster_core.cpp:619).

- **OM/TEX "one request per ocache/tcache port per cycle" is correctly rate-limited** (om_core.cpp:262 `budget=1`, tex_core.cpp:213 `budget=kTcacheNumReqs`), and the comments explicitly warn against looping >1 into one channel (good, this is the right parity discipline). This part is solid.

- **Perf counters are approximations:** `mem_latency += pending_*.size()` each tick (raster_core.cpp:146, om_core.cpp:173, tex_core.cpp:147) integrates outstanding-request-cycles, a reasonable Little's-law proxy, but `stall_cycles` is a coarse "any slot waiting" boolean. Fine for trend tracking, not for cycle-accurate parity.

## 5. "True GPU" alignment vs NVIDIA/AMD/Intel

- **RASTER (bin → tile → block → quad descent with priority FIFOs and a TL-bypass):** This is genuinely representative of a tiled/hierarchical rasterizer (NVIDIA's hierarchical-Z tile walk, AMD's scan-converter coarse/fine raster). The coarse-128px-bin front end matched to a TE/BE walker is a credible model. **Good alignment.**

- **TEX (vx_tex4, integer per-lane LOD, bilinear 4-tap + trilinear 2-LOD blend, per-corner cache-line fetch):** Matches a real TMU's tap-gather + filter pipeline and HW-computed LOD from quad derivatives (sfu_unit.cpp:277-288 `vx_tex_quad_lod`). The quad-LOD-from-window-derivatives path is exactly how real TMUs derive mip level. **Good alignment.**

- **OM/ROP (depth/stencil test + blend R-M-W against a cache):** Architecturally correct shape, **but** real ROPs are defined by their **per-pixel ordering guarantee** (raster order / fragment-order preservation). Two ordering hazards exist here: (1) the SFU→OM window handoff is unscoreboarded (§8 root cause), and (2) OmCore itself does not interlock same-pixel R-M-W (§2 secondary). A "true GPU" ROP enforces fragment order end-to-end; this model enforces it at neither point. **Partial alignment — both need ordering fixes; (1) is the one that breaks §8.**

- **vx_om4/vx_tex4 windowed-quad submit (SFU sequences 4 sub-pixels reading a shared graphics window in the RTU regfile):** Clever reuse of the RTU window as the OM/TEX payload staging area, and the SFU sub-pixel sequencer (sfu_unit.cpp:318-358) is a reasonable model of a quad-granular ROP/TMU front end. The dependence on `VX_CFG_EXT_RTU_ENABLE` for the window (sfu_unit.cpp:359-368 has no non-RTU path) is a coupling smell but is documented.

## 6. v2.1 recommendations (P0/P1/P2)

**P0 — Order the SFU→OM window handoff (resolves §8).**
- Root cause confirmed: `vx_om4` reads the shared per-(warp,lane) graphics window once per cycle across its 4 sub-pixels (sfu_unit.cpp:331-345) with no scoreboard or buffering, and neither `vx_om4` nor `SETW` carries an rd (decode.cpp:1003-1010, 1043-1050), so the next fragment's `SETW` can overwrite the window mid-sequence. The `VX_CFG_OM_MEM_QUEUE_SIZE=1` experiment already proved the OmCore R-M-W is innocent.
- Smallest correct fix: snapshot all four sub-pixels' window payload (colour[0..3], depth[0..3]) into the trace at sub-pixel `F==0` in `SfuUnit::on_tick` (sfu_unit.cpp:322-330), then issue from the snapshot — remove the per-cycle `window_get` at lines 343-344. This makes a mid-sequence SETW harmless. File: `sim/simx/sfu_unit.cpp`.
- Equivalently/additionally give `vx_om4` a scoreboard handle so a following same-slot `SETW` stalls (mirrors `vx_tex4`'s rd-as-sync-handle, tex_core.cpp:199); decide whether the RTL orders this via scoreboard or via window double-buffering and match it. **Parity requirement, not a SimX-local patch.**
- Verify with the instrumented build (Appendix): the 73-pixel CBUF/ZBUF diff → 0, `frame_diff==0`, `depth_gated==yes` all hold.

**P0/P1 — Add an OmCore same-pixel R-M-W interlock (separate from §8, for overdraw correctness).**
- Stall `OmCore::drain_req_in` (om_core.cpp:197) from admitting a request whose lane pixel addresses collide with any in-flight slot until that slot reaches DONE. Mirror the RTL ROP per-pixel ordering. File: `sim/simx/om/om_core.cpp`.

**P1 — Model RASTER walker latency / cross-launch FF warmth (parity).**
- Make `te_walk_tile` advance one pipe-stage per `tick()` instead of running to completion in `run_rasterizer` (raster_core.cpp:619-665 is already a 2-stage FIFO pipeline — drive it from the FSM, one step per cycle). This gives RASTER a real quad-emission rate for timing parity.
- Document (or, if feasible, relax) the per-launch `SimPlatform::reset()` so multi-draw frames can model warm FF pipelines; at minimum, annotate the known limitation in the charter's parity section so timing comparisons are not mis-read. Files: `sim/simx/raster/raster_core.cpp`, `sim/simx/processor.cpp`.

**P1 — Gate RASTER producer on the authoritative pending set, not the shadow counter.**
- Replace `pending_count_==0` transition gates with `pending_reads_.empty()` (raster_core.cpp:259/264/269) and drop `pending_count_`, removing the desync-deadlock risk on a dropped tag. File: `sim/simx/raster/raster_core.cpp`.

**P2 — Cleanups.**
- Remove the duplicate `mreq.op = MemOp::ST;` (om_core.cpp:482-483).
- Delete dead `any_pending`/`all_issued` bookkeeping in OM WRITE_ISSUE/READ_ISSUE (om_core.cpp:289-302, 422-469).
- Make the vx_om4 `qy` unpack mask symmetric with the raster `encode_pos_mask` packing (sfu_unit.cpp:337-338 vs raster_core.cpp:48-53) to remove the latent large-`VX_RASTER_DIM_BITS` bug.
- Replace the O(n²) pid dedup with a sorted-vector `std::unique` (raster_core.cpp:358-373) for determinism and speed.

---
### Appendix: §8 reproduction, localization, and elimination
- Built `sim/simx` with `CONFIGS="-DVX_CFG_EXT_RTU_ENABLE -DVX_CFG_EXT_TEX_ENABLE -DVX_CFG_EXT_OM_ENABLE -DVX_CFG_EXT_RASTER_ENABLE"`, ran `tests/graphics/gfx_pipeline_tex` (build32_amo) via the simx driver, `-w128 -h128`, twice.
- Both runs: `one-batch frame vs two-batch frame differs in 146 bytes`, `depth gated later draw: yes`, `RESULT: FAIL`. All other cross-checks PASS. Deterministic.
- **Localization:** instrumented a copy of `main.cpp` to read back both cbuf images *and* both depth buffers and dump diff coordinates. Result (deterministic across runs):
  `CBUF diff pixels=73 bbox=[4,12]..[121,119]` and `ZBUF diff pixels=73 bbox=[4,12]..[121,119]` — identical pixel sets. Sampled depths flip between `0xffffffff` (clear / no write) and `0xff800000` (far quad's depth written). So it is the far full-screen quad's depth writes landing at a different scattered pixel set — **not** an edge at the centered near/far seam.
- **Elimination of OM concurrency:** rebuilt libsimx with `VX_CFG_OM_MEM_QUEUE_SIZE=1`; the diff stayed at exactly 146 bytes → OmCore R-M-W concurrency is **not** the cause; the divergence enters the OM already different, i.e. upstream in the SFU window handoff.
- **Ruled out by code trace:** FF-core residual (full reset per launch, vortex.cpp:154 → processor.cpp:248), DCR/window-regfile persistence (reset symmetric), and cache-flush asymmetry (per-launch CMD_CACHE_FLUSH retires synchronously in both paths, device.cpp:504 / cmd_processor.cpp:408 / vortex.cpp:149).

---
### §8 follow-up (2026-06-20): the SFU-window diagnosis above is WRONG — three hypotheses refuted

Re-investigated §8 with direct experiments rather than code trace. The §6 P0 root cause (unscoreboarded SFU→OM window handoff) is **disproven**, and so are the two backup theories. Current state of knowledge:

- **Refuted — SFU→OM window race.** Implemented the §6 recommended fix exactly: snapshot all four sub-pixels' colour/depth window payload at sub-pixel `F==0` (added `om_color_`/`om_depth_` to SfuUnit, read from the snapshot instead of live `gfx_window_.get` per cycle). Rebuilt, re-ran `gfx_pipeline_tex -w128 -h128`: **still FAIL, unchanged.** Reasoning confirms why: the SFU processes one OM trace across all 4 sub-pixels **without popping its input**, so a following fragment's `SETW` is stuck behind it in the SFU input queue and cannot mutate the window mid-sequence. The window handoff is not the bug. (Snapshot change reverted — it was an ineffective experiment.)
- **Refuted — OM ocache write-loss across the per-launch reset.** ocache is **write-through** (cluster.cpp:254,270) and `run()` spins until `SimChannelBase::inflight_count()==0` (processor.cpp:241), so every launch's depth writes reach memsim before it returns. `Cache::on_reset` → `set.reset` → `line.reset` clears `valid` (cache.cpp:118-119), so each launch starts cold and reads coherent depth from memory. No dirty-line loss.
- **NEW localization — the bug is B=1-specific and in the FRONT END, not OM.** `gfx_pipeline_tex` **FAILs at `-w128 -h128` (B=1, a single 128px bin) but PASSes at `-w256 -h256` (B=2×2=4 bins).** So it is not the OM/window path at all; it is the binning front end in the single-bin case, and it only manifests one-batch vs two-batch.
- **The sharp contradiction to chase next.** Each `OP_LAUNCH` is one `processor_.run()` that resets the platform and runs to full drain — so `far-frag` is deterministic *from reset* and its only input is device memory, which is identical before it in both submission shapes (same launch sequence, write-through coherent). Yet the result differs. One of these infra assumptions is therefore false. **Next step:** instrument `main.cpp` to read back the zbuf *after the near draw* in both the one-batch and two-batch paths and diff — if near's zbuf already differs, a launch is not as reset-deterministic as assumed; if it matches but far's differs, far-frag is getting non-memory state carried across the in-batch launch boundary (points at an inter-launch barrier the one-batch path is missing, matching the kernel review's P0-2). Then bisect the nine front-end stages at B=1.
- **Scope note:** this is a pre-existing P0 separate from Sprint B (items 5/6/7a/7b, all shipped). Sprint B item 6 (RASTER walker latency) was confirmed orthogonal — §8 fails identically with and without it, and all single-draw functional cross-checks PASS.

### §8 follow-up #2 (2026-06-20): timing/global-state sensitive — not a single-unit config bug

- **OM config persistence is NOT the cause.** `OmCore::reset()` (om_core.cpp:124-135) preserves `dcrs_` and the cached depth state (zbuf base/pitch, depth func/writemask, depth_stencil/blender config), exactly like `RasterCore::reset()`. So OM depth config persists across the per-launch reset *and* across the batch boundary — d1 (two-batch far) already inherits d0's config.
- **Experiment (perturbation, not fix):** adding an explicit `depth_cfg(d1)` to the two-batch far draw flips the frame check FAIL→PASS — **but also corrupts the earlier buffer cross-check / image-device-vs-host checks (13596-byte diffs)**, which compare vectors captured in Paths B/C *before* Path D runs. A Path-D edit perturbing pre-captured-vector comparisons means the change shifts **global device/timing state**, i.e. the four extra DCR_WRITE commands reshape command-stream timing rather than restoring lost config. So the depth_cfg "fix" is a coincidental timing mask, **reverted**.
- **Refined conclusion:** §8 is a **B=1-only, timing-sensitive interaction** (deterministic for a fixed command stream, shifts when the stream is perturbed) somewhere in the single-bin front-end → FF consumer path, with non-obvious cross-path state coupling in the test harness. It is NOT a normal-operation rendering bug: all single-draw checks and all B≥4 (≥256×256) multi-draw checks are bit-exact.
- **Decisive next step (deferred to a dedicated effort):** capture the *far draw's* `tilebuf` (bin headers + sorted pids) in both the one-batch and two-batch contexts and diff. If the far tilebuf differs → front-end binning nondeterminism at B=1 (a real device bug); if identical → the divergence is a RASTER/OM consume-side timing artifact. Then bisect the nine front-end stages at B=1. Estimated: a focused half-day of instrumentation; tracked as an open P0, not blocking the remaining P1 schedule.
