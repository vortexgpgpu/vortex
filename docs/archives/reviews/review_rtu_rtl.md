# v2 Review: prism_v3 RTU RTL implementation

**Date:** 2026-06-17
**Reviewer scope:** `hw/rtl/rtu/` (SystemVerilog ray-traversal unit) on branch `prism`, tree `~/dev/vortex_v3/prism_v3`. Read/grep only; no synthesis run.
**Target constraint:** Alveo U55C @ 300 MHz; maximize BRAM/DSP hard-block reuse; reuse `VX_fma_unit`/box+tri PEs for proc-AABB & TLAS transform (no new datapaths).

---

## 1. Overall assessment (maturity grade: **B / B+**)

This is a clean, well-structured, genuinely *minimal* RTU. The microarchitecture is sound and matches the stated charter: a single shared FP geometry datapath (one box PE, one tri PE, one reciprocal, one node decoder) time-multiplexed across a small per-warp context pool by a short-stack closest-hit FSM. The block-reuse mandate is honored faithfully — the box PE doubles for procedural-leaf raw AABBs (`raw`/`raw_min`/`raw_max`, `VX_rtu_box_pe.sv:51-53,151-167`), `VX_rtu_fdot3` is reused for the TLAS R^T matrix-vector products (`VX_rtu_xform.sv:106-115`), and a single `VX_fma_unit` family backs every FP op. The FP-efficiency refactor (fused `VX_rtu_fmac3` normalize-once dot/cross, `EN_EXCEPT=0` on provably-finite FMAs, `ACC_SPLIT`, shared single reciprocal, per-context state → BRAM) is well-reasoned and the as-built area numbers (LUT -18%, SRL -70%, DSP held flat, BRAM rebalanced) are credible and on-charter.

The grade is held below A by three things, in priority order:

1. **A structural throughput/scalability ceiling** (not a logic bug): the RTU core processes **exactly one warp at a time** with **no request queue**, and `VX_CFG_RTU_NUM_CTX` is bound to `NUM_THREADS` (lanes), not warps. This is the *direct* cause of the deferred "sustained multi-warp `rt_raycast` scoreboard timeout" — it is head-of-line serialization, not a deadlock (see §2.1). It also means the design's named scaling axis (latency hiding over RTCache miss latency) is mostly unrealized.
2. **Timing is not yet closed.** Per the efficiency doc's own as-built table, the best signed-off P0–P4 build is **WNS −0.028 ns (still violated)**, and P5 (BRAM) regressed to −0.388 before the P6 restage whose post-fix WNS was still "in flight." So 300 MHz is *close but unconfirmed* — the headline hard constraint is unmet at time of review.
3. **Dead/misleading config surface** (`RTU_BOX_PE`/`RTU_TRI_PE`/`CONTEXT_POOL` knobs that don't shape the RTL), and a few latency-knob mismatches vs `VX_config.toml` (§3).

Within its declared opaque-only / minimal scope the logic is correct (18/18 rtlsim, bit-identical across the refactor), the reset/backpressure discipline is consistent, and the tag-routed async PE/mem responses are handled correctly. This is a solid v1-of-hardware; the gap to A is the multi-warp architecture and confirmed timing closure, both of which the team already has on the open-items list.

---

## 2. Correctness findings

### 2.1 [`VX_rtu_core.sv:211` + `VX_rtu_scheduler.sv:46-47`] — single-warp serialization is the multi-warp "scoreboard timeout" root cause — **HIGH (functional ceiling), not a deadlock**

`VX_rtu_core` is a single-context FSM: `req_ready` is asserted **only in `C_IDLE`** (`VX_rtu_core.sv:211`), and stays low for the *entire* traversal of the admitted warp (`C_BUSY` spans thousands of cycles of BVH/RTCache walking). The SFU side (`VX_gfx_window` `bstate`) is likewise single-context (`arm_go` accepted only in `B_IDLE`). The only TRACE2 warp-unlock is `async_trap_if.unlock`, driven from `armwb_fire` after the RTU core completes its round-trip for *that* warp.

Consequence under sustained multi-warp load: warps queue strictly serially behind one RTU core. A warp parked on WAIT2 accumulates the scoreboard watchdog counter (`VX_scoreboard.sv` `timeout_ctr`, fixed `STALL_TIMEOUT = 100000`, `VX_gpu_pkg.sv`) for roughly `Σ(traversal latency of all warps ahead)`. Once that sum crosses 100k cycles, the simulation `RUNTIME_ASSERT` fires — reported as a "hang." **This is forward-progress-preserving serialization exceeding a fixed sim watchdog, not a circular-wait deadlock.** The core always returns to `C_IDLE` and every warp is eventually serviced.

Note `NUM_CTX` parallelizes *lanes within one warp*, not warps — so adding warps adds pure serialization, exactly matching the failure signature. The SimX oracle sizes its pool at `CONTEXT_POOL=32` and pools across warps (`VX_config.toml:57`); the RTL has no such cross-warp pool. This is the single most important architectural gap (see P0, §6).

### 2.2 [`VX_rtu_scheduler.sv:234-245`] — round-robin selector is a serial priority chain — **LOW (correctness OK; timing/throughput concern, see §4)**

The selector loops `off = NUM_CTX-1 downto 0` and assigns on every `runnable[cand]`, so the *last* iteration (`off=0`, `cand=cc`) wins — correctly giving priority to `cc` (the just-run context), which is what enforces the implicit single-context reservation of the shared datapath during CS_SETUP / CS_FEED / CS_WAIT spans. Correctness is fine and the reservation is what makes the shared reciprocal and box-collection safe (§2.3). But it is a combinational `% NUM_CTX` priority cascade; see §4.3.

### 2.3 [`VX_rtu_scheduler.sv:368-379, 598-610`] — shared reciprocal correctness depends on the implicit reservation — **LOW (correct as built; fragile)**

The single `VX_rtu_recip` runs continuously and is fed `recip_din = ray_q.dir[setupaxis_q]` from the EXEC snapshot. It is only correct because a CS_SETUP context monopolizes selection (§2.2) so the divider sees a stable input for its whole span; the counter `setup_ctr` (counting EXEC visits to SETUP_LAT=17) over-waits relative to the recip's true depth, which is safe. This is correct but **undocumented-as-load-bearing**: if the selection priority (§2.2) is ever changed to interleave contexts during setup (e.g. to hide the 3×17-EXEC setup latency), the reciprocal result silently corrupts. Recommend an explicit "context owns the recip for its setup span" guard or assertion. Same coupling applies to the box-result collection (`coll_idx[sel_q]` advanced async in `VX_rtu_scheduler.sv:544-548` while the same context stays selected across CS_FEED/CS_WAIT).

### 2.4 [`VX_rtu_scheduler.sv:289-296` stack RAM] — push/pop ordering across phases is correct — **OK (verified)**

`stack_ram` is LUTRAM, `OUT_REG=0`, `RDW_MODE="W"` (async combinational read, `VX_dp_ram.sv:378-379`). Read addr uses live `sp[sel]-1` in SELECT (latched to `stacktop_q`); write happens in EXEC CS_PUSH at `sp_q`. Push (CS_PUSH) and the consuming pop (CS_POP) are separated by full micro-steps, and `sp` is updated in EXEC, so the SELECT-phase read always reflects the committed `sp`. No same-cycle RDW hazard. Correct.

### 2.5 [`VX_rtu_box_pe.sv:24-30, 121-126`] — origin-subtract-before-multiply for axis-aligned rays — **OK (good)**

The slab test computes `(mn-ro)*inv_d` rather than `mn*inv_d - ro*inv_d`, so axis-aligned rays where `inv_d = ±inf` yield a signed inf (non-constraining slab) instead of `inf-inf = NaN`. This is the correct watertight-traversal choice and matches real-HW practice. The `EN_EXCEPT=0` is correctly applied only to the finite origin/dequant FMAs while the slab FMAs (`fma_t0`/`fma_t1`, `LAT_SLAB`) keep default `EN_EXCEPT=1` (they see ±inf). Verified consistent.

### 2.6 [`VX_rtu_tri_pe.sv:75-76, 264-265`] — Möller–Trumbore EPS and two-sided det test — **OK, minor watertightness note — LOW**

Hit gate uses `|det| >= 1e-6` via two compares (`cdp|cdn`) and back-face from `det<0`. Fixed absolute `EPS=1e-6` is scale-dependent: for very small or very large world-space triangles the absolute determinant threshold can mis-classify near-degenerate or large triangles (false miss / false hit on grazing rays). The minimal scope (orthographic test scene) doesn't exercise this, and the SimX oracle presumably matches, but it is a known watertightness corner for production scenes. Flag for the precision gate (§6 of the efficiency doc, still un-run).

### 2.7 [`VX_rtu_fmac3.sv:42-47`] — exponent-difference shift has no clamp — **LOW (benign under finite-input assumption)**

`sh = max_pe - pe[i]` then `field >> sh`. With the FW=80-bit field, a term whose exponent is >80 below the max shifts to zero (correct). `pe=0` denotes an unused/zero term and `max_pe` is taken over actual terms, so `sh` is bounded by the real exponent spread. Under the documented finite-input/FTZ assumption this is safe; there is no UB. Worth an assertion that `pe[i] <= max_pe` always (it is, by construction) for defensiveness.

### 2.8 [`VX_rtu_xform.sv:18-26`] — R^T inverse assumes orthonormal rotation — **LOW (documented, scope-correct)**

The transform applies `R^T` (no determinant/division), valid only for orthonormal R. The header documents this as bit-equivalent to SimX's cofactor inverse *for orthonormal R*, which is all a valid Vulkan TLAS produces. Non-uniform-scale / shear instances would be silently wrong, but those are out of the minimal scope and the driver cap-guard routes them to SW. Acceptable as a documented limitation; should be an explicit `STATIC`/runtime note so it isn't relied on for general affine TLAS later.

### 2.9 [`VX_rtu_core.sv:140-207`] reset / backpressure — **OK**

Reset initializes `cstate`, `sch_start`. Response handshake (`C_RSP`/`C_CBYIELD` gated on `rsp_ready`) and request gate are consistent valid/ready. Idle (non-lane) contexts when `NUM_CTX>NUM_LANES` are explicitly tied and UNUSED-tagged (`VX_rtu_core.sv:236-243`). No combinational valid↔ready loop observed. `mem_rsp_ready=1'b1` (always-accept) in `VX_rtu_scheduler.sv:445` is safe because the f_buf RAMs capture every response by tag; fine given one outstanding line per context.

---

## 3. Efficiency / area findings

### 3.1 Block reuse — **strong, on-charter**
- Box PE reused for procedural-leaf raw AABBs via `raw` path (`VX_rtu_box_pe.sv:51-53,137-167`) — no second box datapath. Good.
- `VX_rtu_fdot3` reused for the TLAS world→object transform (`VX_rtu_xform.sv:106-115`) — no new matrix unit. Good.
- Single `VX_rtu_fmac3` shared by both `fdot3` (3 terms) and `fcross3` (2 terms/axis), normalize+round once instead of per-FMA (`VX_rtu_fmac3.sv:14-21`). This is the headline LUT win (-18% LUT, -70% SRL as-built) and is the correct FPGA move.
- One reciprocal time-multiplexed over 3 axes (`VX_rtu_scheduler.sv:362-379`); collapsed from 3 setup dividers. Correct trade (2 dividers → 2 extra setup passes).

### 3.2 [`VX_config.toml:55-56` vs RTL] — `RTU_BOX_PE`/`RTU_TRI_PE` knobs are dead in RTL — **MEDIUM (misleading)**
The config advertises `RTU_BOX_PE=4`, `RTU_TRI_PE=4` ("parallel intersection PEs, matches BVH_WIDTH"). The RTL instantiates **exactly one** `VX_rtu_box_pe` and **one** `VX_rtu_tri_pe` (`VX_rtu_scheduler.sv:390,407`), streaming one child/triangle per cycle. The efficiency doc P2 claims the `RTU_BOX_PE`/`RTU_TRI_PE` *pkg localparams* were retired, but the `VX_CFG_*` config knobs survive and still imply 4-wide parallel PEs that don't exist. This is the gap between the proposal's "`VX_rtu_box_pe[W]` … run the W slab tests in parallel" (minimal proposal §5) and the actual single-streaming datapath. Either wire real W-wide parallelism (a genuine throughput win for internal nodes — W box tests/cycle) or delete/relabel the knobs so they don't mislead integrators. The single PE is a deliberate area choice, but the config surface lies about it.

### 3.3 [`VX_config.toml:58-62` vs `VX_rtu_pkg.sv:49-70`] — latency knobs diverge from config — **LOW (intentional but confusing)**
`VX_CFG_RTU_NODE_LATENCY=4` / `TRI_LATENCY=6` / `XFORM_LATENCY=3` in the toml are SimX cycle knobs; the RTL uses `RTU_LATENCY_FMA=9` (and derived `8*F+V+2` tri, `4*F` xform). These are deliberately decoupled (RTL pipe depth ≠ SimX abstract latency), but the names collide and a reader will assume the RTL honors the toml. The pkg comment explains the `=9` floor well; recommend renaming the SimX knobs (`*_SIMX_LATENCY`) or documenting the decoupling in `VX_config.toml`.

### 3.4 Per-context state placement — **good, with one caveat**
- `stack` → LUTRAM (`VX_rtu_scheduler.sv:291`, `LUTRAM(1)`), narrow & shallow — correct choice.
- `f_buf` node image (`g_fbuf_ram`, one RAM per line slot, `OUT_REG=1`) and `inst_xform` (`xform_ram`, 12×32b) → BRAM via `FORCE_BRAM` heuristic (`VX_rtu_scheduler.sv:331-355`). On-charter (BRAM was at 0%). Caveat: at `NUM_CTX=4` these are very shallow (depth 4) wide RAMs — they spend whole RAMB tiles at low occupancy (the efficiency doc §7 flags this). Fine in *fabric* terms (the goal) but confirm RAMB tile count is acceptable; the per-context cost only amortizes at production `NUM_CTX`, which the RTL does not yet reach (§3.5).
- ray/hit/obj records remain FF — narrow, appropriate.

### 3.5 [`VX_rtu_core.sv:43-48`, `VX_config.toml:63`] — `NUM_CTX` decoupling knob exists but is bound to `NUM_THREADS` — **MEDIUM**
The `VX_CFG_RTU_NUM_CTX` plumbing and `STATIC_ASSERT(NUM_CTX >= NUM_LANES)` are in place, and per-context state is on BRAM ready to scale. But the default is `= NUM_THREADS`, and crucially the ray→context *mapping* for `NUM_CTX ≠ NUM_LANES` (queuing rays into a deeper pool, pooling across warps) is **not implemented** — the extra contexts simply idle (`VX_rtu_core.sv:73-75`). So the BRAM investment buys nothing yet, and the multi-warp ceiling (§2.1) stands. The knob is half-built.

### 3.6 DSP usage — **on-charter, flat**
DSP held at 146 (1.6%) across the refactor; the optional `DSP_SEED` reciprocal (`VX_rtu_recip.sv:41-133`) is a clean, correct seed-ROM(BRAM)+2×NR(DSP) implementation that is opt-in (`VX_CFG_RTU_RECIP_DSP_SEED=0` default), respecting the DSP-scarcity priority. The seed ROM, Q-format NR steps, and normalize/pack look numerically right (~9e-8 rel err, well inside 1e-4). Good.

---

## 4. Performance / 300 MHz timing-closure findings

### 4.1 Timing is NOT yet closed — **HIGH (the hard constraint is unmet at review time)**
Per the efficiency doc as-built table: best P0–P4 signoff build = **WNS −0.028 ns (VIOLATED)**; P5 BRAM move regressed to **−0.388 ns**; P6 restage rtlsim-green but post-fix WNS "in flight." The binding path migrated from scheduler addr-gen → `VX_fma_unit` ALIGN→ACC carry chain → (post-P5) the `f_buf` BRAM-read → 1024-bit `f_aligned` barrel shift. The P6 ALIGN flop (`fbuf_q`, `VX_rtu_scheduler.sv:181,588-593`) is the right structural fix (register BRAM output before the shift cone), but **the review cannot confirm 300 MHz** — this is the top open risk against the stated constraint.

### 4.2 [`VX_rtu_scheduler.sv:250-276`] — the byte-align barrel shift + node decode is the structural critical cone — **MEDIUM**
`f_aligned = fbuf_q >> f_shift` is a `BUF_BITS`-wide (≥1024-bit for 2-line nodes) variable barrel shifter, feeding the combinational `VX_rtu_node_decode` (all qmin/qmax/child_off slices) AND the leaf-vertex slices AND `node_lines`/`leaf_lines`/`inst_lines` arithmetic, all in EXEC. Even with the ALIGN flop in front, the shift→decode→FSM-next-state path is long. The 3-phase (SELECT/ALIGN/EXEC) split is good, but consider: (a) precomputing the byte-shift amount in SELECT (it depends only on `struct_addr[5:0]`, already available pre-EXEC), and (b) registering `node_img` (the low `RTU_NODE_IMG_BITS` after shift) before the decoder. The decoder itself is pure slicing (cheap); the barrel shifter is the cost.

### 4.3 [`VX_rtu_scheduler.sv:234-245`] — selector `% NUM_CTX` priority cascade — **LOW at NUM_CTX=4, MEDIUM at production**
The runnable-select loop is a combinational chain with a modulo per iteration. At `NUM_CTX=4` it is trivial. If `NUM_CTX` is raised to production width (the design's stated intent), this becomes an O(NUM_CTX) priority encoder with `%` — a likely timing and area problem. Replace with a rotate + leading-one (`VX_find_first`-style) priority encoder before scaling `NUM_CTX`.

### 4.4 Throughput: no latency hiding during setup / node-feed — **MEDIUM**
Because a context monopolizes selection across CS_SETUP (3×17 EXEC visits = ~153 cycles incl. phases) and CS_FEED/CS_WAIT (box stream), the shared datapath does **not** interleave other contexts during these spans — defeating the context pool's purpose for everything except cache-miss and tri-test parking. Real RT cores keep the box/tri units busy every cycle across many rays; here the box PE is idle during any selected context's setup/pop/dispatch. Combined with §2.1 (one warp at a time), sustained occupancy of the FP datapath is low. This is the performance counterpart to the area-efficient single-PE choice.

### 4.5 Recip / tri-PE depth — **OK**
`RTU_FDIV_LAT=17` (NR div) and tri-PE `8*F+V+2 = 8*9+17+2 = 91` cycles are long but fully pipelined and parked (the context yields during CS_TRI_WAIT), so they cost latency, not throughput — appropriate given the design parks correctly. `RTU_LATENCY_FMA=9` floor (keeping the mantissa multiply on DSP) is well-justified in `VX_rtu_pkg.sv:49-54`.

---

## 5. "True GPU" alignment vs NVIDIA / AMD / Intel

**What aligns well:**
- **CW-BVH4 / 64 B node = 1 cache line** is the right call and explicitly mirrors **AMD RDNA2/3** (BVH4, 64 B box nodes, `image_bvh_intersect_ray`). The minimal proposal's rationale (§1) is correct and HW-grounded.
- **Box + tri as the two intersection primitives**, short-stack closest-hit, front-to-back ordering with `t_hit` pruning — all standard and correct, matching NVIDIA RT-core and AMD ray-accelerator semantics.
- **Procedural-leaf reuses the box PE** for AABB-entry, yielding IS to a shader — conceptually aligned with how all three vendors handle custom-intersection AABالبات.
- **Cluster-shared fixed-function unit behind the SFU**, opaque fast-path commits in-HW — matches the "traversal in HW, shading in shader" split.

**Where it falls short of competitive RT HW:**
- **No box/tri MIMD parallelism.** NVIDIA RT cores run multiple box test units (and a tri unit) concurrently per ray; AMD does 4 box / 1 tri per clock. This RTL does **one box test per cycle, one tri per ~91 cycles, one warp at a time** (§3.2, §2.1). The `RTU_BOX_PE=4` knob promises this and the RTL doesn't deliver it. This is the single biggest architectural gap vs real HW — internal-node traversal is bottlenecked at 1 child/cycle when the natural design is W/cycle.
- **No ray reordering / coherency sorting.** The SimX model has §8.9 octant-signature 2-pass coherency gather (`rtu_implementation.md:172,592` — lives in `RtuCore::Impl`), but it is **entirely absent from the RTL**. Modern RT HW treats this as essential: **Intel Xe has a dedicated Thread Sorting Unit (TSU)**; NVIDIA Ada added Shader Execution Reordering (SER); AMD bins rays for coherence. The RTL ships zero reordering — every ray walks independently in lane order. For the minimal opaque test this is fine; for "true GPU" / production raytracing it is the second major gap. Not a v1 defect, but the headline item for the roadmap.
- **No cross-warp context pool.** Real RT cores keep dozens–hundreds of rays in flight to hide BVH memory latency. The RTL's "pool" is `NUM_THREADS` (lanes of one warp); `NUM_CTX` decoupling is stubbed (§3.5). This is the same root as §2.1.
- **Memory: single outstanding line per context, always-ready response.** Competitive units have a wide MSHR/ray-queue feeding the box units continuously. `VX_CFG_NUM_RTU_BLOCKS=2` is advertised but the scheduler issues one line at a time per context. Adequate for minimal scope; a bandwidth ceiling otherwise.

**Verdict:** The *microarchitectural skeleton* (short-stack + context-pool + shared PE + CW-BVH4) is sound and vendor-aligned. The *scaling dimensions that make RT HW fast* — MIMD box/tri units, deep cross-warp ray pools, and ray-coherency reordering — are either stubbed or absent. That is consistent with "minimal opaque-only RTL by design," but the config surface (`BOX_PE/TRI_PE/CONTEXT_POOL`) overstates what's built.

---

## 6. v2.1 recommendations

### P0 — must do (unblocks the deferred multi-warp failure + the hard timing constraint)
- **P0a — Break the single-warp serialization (§2.1).** Add a small request queue / multiple in-flight *warp* contexts to `VX_rtu_core` so `req_ready` need not gate on `C_IDLE`, OR at minimum implement the stubbed `NUM_CTX > NUM_LANES` ray→context mapping (`VX_rtu_core.sv:73-75`, `VX_rtu_scheduler.sv` launch) so a second warp's rays queue into idle contexts instead of blocking the SFU. The BRAM per-context state is already in place for this. This is the structural fix for "sustained multi-warp `rt_raycast` timeout." (If a true fix is out of scope for v2.1, at least decouple the scoreboard watchdog from RTU latency so the sim assertion reflects real deadlock, not serialization — but that is a patch, not the fix.)
- **P0b — Confirm 300 MHz closure (§4.1).** Land the P6 `build_w4_p6` WNS measurement and the production-`NUM_CTX` synth the efficiency doc lists as open. Without a confirmed non-negative WNS the project's hard constraint is unmet. If the `f_aligned` barrel shift (§4.2) is the residual binding path, precompute the shift amount in SELECT and register `node_img`.

### P1 — should do
- **P1a — Wire real W-wide box parallelism or delete the knob (§3.2).** Instantiate `RTU_BOX_PE` box PEs and test W children/cycle for internal nodes (the largest single throughput win, and the thing that makes it look like real RT HW). If not, remove `VX_CFG_RTU_BOX_PE`/`RTU_TRI_PE`/`CONTEXT_POOL` or relabel them so they stop advertising parallelism the RTL lacks.
- **P1b — Make the shared-datapath reservation explicit (§2.3).** Add an assertion/comment that a context owns the reciprocal (and box-collection) for its full setup/feed span; the current correctness silently depends on the selector priority quirk (§2.2).
- **P1c — Replace the `%NUM_CTX` selector with a rotate+priority-encoder (§4.3)** before raising `NUM_CTX`, else it re-breaks timing at production width.
- **P1d — Run the §6 precision gate** (efficiency doc, still un-run) — especially tri-PE scale-dependent `EPS` watertightness (§2.6) and the FTZ/no-except changes — against the SimX oracle before trusting the FP refactor for production scenes.

### P2 — nice to have / roadmap
- **P2a — Ray-coherency reordering (§5).** Port the SimX §8.9 octant-signature gather into RTL (or a TSU-style pre-sort) — the defining feature gap vs Intel Xe TSU / NVIDIA SER / AMD ray binning. Roadmap item, not a v1 defect.
- **P2b — Reconcile `VX_config.toml` latency knobs vs RTL (§3.3)** — rename SimX-only knobs `*_SIMX_LATENCY` and document the RTL decoupling.
- **P2c — Document the orthonormal-only TLAS inverse (§2.8)** as an explicit constraint so general-affine instancing isn't built on it later.
- **P2d — Defensive assertions** in `VX_rtu_fmac3` (`pe[i] <= max_pe`) and on `node.n_children` clamp paths.
