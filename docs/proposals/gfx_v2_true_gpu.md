# gfx_v2 — True GPU Graphics on Vortex (master plan)

**Status:** CURRENT — the single source of truth for the Vortex graphics program.
Self-contained; consolidates and supersedes the gfx_v2 proposal/review set now in
[../archives/](../archives/) (kept for history only). RTU/ray-tracing is **out of
scope** here (tracked separately under `proposals/rtu_*`).
**Tree:** `~/dev/vortex_v3/prism_v3` (+ Mesa driver `~/dev/mesa_vortex`).
**Date:** 2026-06-20.

---

## 0. Definition of done (the north star)

> **Pass the Vulkan Conformance Test Suite on the Alveo U55C, running on 4 Vortex
> cores, with rendering performed on-device using the fixed-function RASTER + OM +
> TEX acceleration — no host-side rendering, no per-draw host round-trips.**

Everything in this plan is justified by, and scheduled against, that single
criterion. Three sub-claims it decomposes into:

1. **On-device** — the whole draw (vertex → setup → bin → raster → fragment →
   ROP) executes on Vortex over device-resident memory; the host is only the
   driver (compile + submit + present). llvmpipe is an **offline oracle**, never a
   runtime path.
2. **Fixed-function accelerated** — RASTER, OM, TEX do their jobs in hardware on
   the U55C bitstream at 4 cores; the on-device SIMT software path is the
   always-correct fallback for what the FF units cannot represent.
3. **Conformant** — the result passes Khronos CTS, not just our internal golden
   traces.

---

## 1. Architecture we are building

### 1.1 End-to-end draw flow (target)

```
 host: compile shaders + build command/state block ──submit──► CP command ring
                                                               │ (host idle until present)
  ┌──────────────────── CP sequences, fully device-resident ────────────────────┐
  │ ① VS / (later GS/tess/mesh)  ─► transformed vertex records      (resident)   │
  │ ② Triangle setup (SIMT): cull, near/guardband clip (0–7 subtris),           │
  │    snap to fixed-point, plane eqs (z/w,u/w,v/w,1/w), min-z       (resident)  │
  │ ③ Parallel bin-sort (SIMT) ─► dense primbuf + bin headers + PID array        │
  │ ④ CP programs RASTER / OM / TEX config device-side (DCR/descriptor)          │
  │ ⑤ Fine raster (RASTER FF) ─► FS (SIMT) ─► vx_tex4 (TEX FF) ─► vx_om4 (OM FF)  │
  │     └─ any FF-unrepresentable state → on-device SIMT software path instead    │
  └──────────────────────────────────────────────────────────────────────────────┘
                                                               │
                                       color attachment resident ──► present (DMA)
```

Everything between submit and present is device-resident and host-untouched.

### 1.2 Dual-path principle

- **FF units = fast path.** RASTER (coverage/quad-gen), OM (depth/stencil/blend/
  ROP), TEX (sampling). Fixed-point, mobile-class, composable primitives — no FP
  datapath inside any FF unit.
- **On-device SIMT software = complete path.** A device-side software rasterizer/
  sampler/ROP (CUDARaster-lineage) catches everything the FF units cannot
  represent (exotic formats, blend/logic-op modes, MSAA resolve, …).
- **The CP chooses per-draw.** Full residency makes the SW path *mandatory*, not
  optional: once intermediates never touch host memory, there is nothing for the
  CPU to fall back to.

### 1.3 The binding interface law — FF ↔ SIMT doctrine (C1–C5)

The graphics program kept re-incurring one class of bug — *un-ordered shared
FF↔SIMT side-band state handed off across work items with no scoreboard
guarantee* (the §8 OM multi-draw determinism bug; the RASTER multi-core/
multi-drawcall dropped-draw-call bug). Every FF↔SIMT interface MUST satisfy:

- **C1 Scope-partition** arguments to their natural home (per-thread → register
  window; per-warp → uniform channel / `vx_wgather`; per-call → source register;
  per-dispatch → DCR).
- **C2 Single-issue** — one architectural op per logical operation (macro-op
  expansion internally is fine).
- **C3 Scoreboard-ordered handoff** — every FF↔SIMT value moves through
  scoreboarded registers and the op retires under the scoreboard. (A register
  *window* alone — C2 — is not enough; the *handoff* must be scoreboarded.)
- **C4 No shared mutable side-band** state that outlives a single scoreboarded op
  (no per-(warp,pid,lane) CSR latch, no cross-unit shared window, no cross-core
  arbiter sticky state).
- **C5 Explicit lifecycle & completion** — completion by handle/count, never a
  value-encoded sentinel; a draw/frame epoch has a **single owner** (one FSM),
  with epoch-tagged work items.

PRISM RTU ISA v2 is the in-house reference instance. Enforcement is mechanized:
PR checklist + SimX structural assertion (every FF op has a scoreboarded rd/handle
or is provably side-effect-free) + a CI parity matrix (multi-core × multi-cluster
× multi-drawcall, SimX↔RTL). This law is what stops the recurring-patch failure
mode by construction.

### 1.4 ABI surface (graphics, gfx-only)

- **FF ops (CUSTOM1 = 0x2B):** `vx_rast` / `vx_rast_begin` (raster pull — *to be
  replaced*, §4), `vx_tex4` single/quad (TEX, windowed), `vx_om4` (OM, windowed).
  Legacy `vx_tex`/`vx_om` retained only until the driver fully emits the v4 forms.
- **On-wire formats (`sw/common/vx_gfx_abi.h`, `gfx_frontend_abi.h`):**
  `rast_prim_t` (edges + Q-format attribute deltas, 120 B), 12-byte
  `rast_bin_header_t`, `pipe_arg_t` (front-end stage args), fixed-point `fixed_t<F>`.
- **DCRs:** raster (TBUF/PBUF/TILE_COUNT/STRIDE/SCISSOR), om (CBUF/ZBUF addr+pitch,
  writemasks, depth/stencil/blend state), tex (per-stage addr/format/filter/wrap/mip).

---

## 2. Current state (what is done)

Reflects code as of 2026-06-20 (past the Jun-17 review synthesis — several review
P0/P1 items have since landed). Grades: ✅ done · ⚠️ partial · ❌ missing. Items
marked **(verify)** moved recently and need a parity check.

| Layer | Component | State | Notes |
|---|---|:--:|---|
| **SimX** | RASTER model | ✅ | producer FSM, TE/BE walker, 12-byte bin header, perf counters |
| | OM model | ✅ | R-M-W pipeline + **same-pixel interlock** (`om_core.cpp` `collides_with_inflight`) |
| | TEX model | ✅ | trilinear, window `vx_tex4` single/quad |
| | §8 multi-draw determinism | ⚠️ | OM R-M-W innocent; residual hazard is the SFU **window handoff** (`sfu_unit.cpp`) — C3/C4 fix pending (verify) |
| **Front-end (on-device, `sw/gfx/pipe_frontend.h`)** | expand_k (VS assembly) | ✅ | 1 thread/vertex → `setup_vertex_t` |
| | setup_k (clip+setup) | ⚠️ | near-plane clip + setup + scan/emit; **no cull modes**, clip→bin feedback partial |
| | binning_k (bin-sort) | ✅ | 9-stage parallel sort-middle → dense primbuf + headers + PID array |
| **Runtime (`sw/runtime`)** | DrawCommands (CP batch) | ✅ | one doorbell/one completion; multi-stage no host round-trip |
| | FrontEndPool | ⚠️ | device-resident pool + 9-launch emitter; **16 separate allocs**, not a pooled slab |
| | FF register emitters | ✅ | `program_raster/om/tex(DrawCommands&)` |
| | host `Binning()` | ✅ | retained only as coverage **oracle**, not the runtime path |
| **RTL (`hw/rtl`)** | RASTER front-end | ⚠️ | `VX_raster_mem/te/be/core/arb`; reads 12-byte header **(verify)**; **dispatch interface still pull/sentinel/arb** (§4 defect) |
| | OM unit | ❌/⚠️ | interface + wrapper exist; **datapath not built out** — SimX authoritative |
| | TEX unit | ❌/⚠️ | interface + wrapper exist; **datapath not built out** — SimX authoritative |
| | cluster/graphics wrap | ✅ | `VX_graphics.sv`, `VX_cluster.sv`; per-cluster FF + r/o/t-cache + DCR fan-out |
| **Kernel/ABI** | intrinsics + ABI types | ✅ | `vx_graphics.h`, `vx_gfx_abi.h`, `gfx_frontend_abi.h` |
| **Mesa (`vortexpipe`)** | front-end draw path | ⚠️ | `vp_raster.cpp` emits expand→setup→bin→fs as **one batch** (verify it's the default, not host `Binning`) |
| | FS NIR→LLVM | ⚠️ | emits kernel + (verify) `vx_om4`/`vx_tex4` vs legacy; rv32/rv64 |
| | residency / thin driver | ⚠️ | color/depth pinned `VX_MEM_PHYS`; non-gfx still llvmpipe-hosted |
| | SW fallback | ⚠️ | `sw_om` software path only; no full sampler/raster SW fallback |
| **Tests** | unit + pipeline + draw3d | ✅ | `gfx_raster/om/tex/tex4*`, `gfx_pipeline_*`, `gfx_draw3d` (CGLTrace golden) |
| | multi-core × multi-drawcall parity | ❌ | **not in CI**; the gap that hid both interface bugs |
| | Vulkan CTS | ❌ | vortexpipe is a lavapipe guest; no standalone ICD/CTS harness |

**Headline reading:** the *functional model (SimX)* and the *on-device front-end +
runtime* are substantially built and the driver path is largely wired; the two
big deficits against the north star are (a) **RTL OM/TEX datapaths are not built
out** (the U55C FF acceleration is currently RASTER-only in hardware), and (b)
**no Vulkan CTS harness on hardware**. The interface-law violations (§3) are the
correctness blockers in between.

---

## 3. What to adjust (in-flight defects — doctrine compliance + known bugs)

These are corrections to existing code, ordered by blocking-ness. Each is an
instance of the C1–C5 law (§1.3).

1. **OM §8 / window handoff (C3,C4).** Give `vx_om4` a scoreboard handle (parity
   with `vx_tex4`/RTU `wait`) and stop reading the shared window mid-sequence;
   **per-unit, scoreboard-retired windows** — remove the cross-unit shared
   graphics window entirely (retires the "gfx aborts without RTU" coupling). Do
   the *scoreboarded* variant, not the snapshot end-patch. → SimX gfx 6/7 → 7/7.
2. **RASTER dispatch interface v2 (C1–C5).** Replace the `vx_rast` pull + `pos_mask==0`
   sentinel + per-(warp,pid,lane) CSR latch + cross-core `VX_raster_arb`
   sticky/flush with a **push/launch** model (§4). This is the root fix for the
   multi-core/multi-drawcall dropped-draw-call bug, not another arb patch.
3. **pid width reconcile.** Producer `PIPE_PRIM_BITS=20` vs consumer
   `VX_RASTER_PID_BITS=16` — pick one width across producer/SimX/RTL + bound-check;
   silent aliasing past 65 535 prims today.
4. **RTL RASTER header parity (verify).** Confirm `VX_raster_mem.sv` reads the
   12-byte `rast_bin_header_t` (absolute `pids_offset`) the runtime emits; diff
   SimX↔RTL on a binned scene.
5. **Mesa FS ABI (verify/finish).** FS must emit `vx_om4`/`vx_tex4` (windowed),
   not legacy `vx_om`/`vx_tex`; the driver default path must be FrontEndPool/
   DrawCommands (no host `Binning`, no per-draw readback).
6. **Build-contract fix.** `vp_compile.c` must honor the install-tree (pkg-config)
   contract and the correct env var (`VP_VORTEX_PATH`).
7. **RASTER cycle model in SimX.** The walker currently contributes ~0 modeled
   cycles; drive it one pipe-stage/tick for throughput parity (needed for any
   perf claim and for honest scheduling on U55C).

---

## 4. RASTER ↔ SIMT dispatch v2 (the interface redesign)

The last FF unit still on the pre-doctrine pattern, and the only one whose
*control* path is broken. Decision (highest-performance, correct-by-construction):

- **Push, not pull.** A hardware **Fragment Work Distributor (FWD)** pulls covered
  quads from the (unchanged) TE/BE raster math, packs `NUM_THREADS` covered quads
  into a fragment **wave**, and **launches** it onto a shader core — payload
  (coverage, x/y, pid, barycentrics) delivered as wave **input registers** (C1/C2/
  C3). The shader never polls; `vx_rast` and the bcoord CSRs are removed.
- **Single-owner epoch FSM** (`QUIESCED → FILLING → DRAINING → QUIESCED`): a draw
  is a CP-issued, epoch-tagged transaction; completion is
  `producer_drained ∧ launched == retired`, counted — no sentinel (C5).
- **Static screen-space tile ownership** (tile → one core): raster order and OM
  read-modify-write become correct *by construction* (no pixel touched by two
  cores), subsuming the cross-core OM-ordering concern into one front-end property
  (C4).
- **Unify with the compute/CTA dispatch path** (`scheduler_cta_encapsulation`,
  `kmu_cta_dispatch` redesign) so fragment work is "just another work source" for
  the warp scheduler.
- **Interim (de-risk only):** a single **scoreboard-ordered** `frag.payload` pull
  op (handle-terminated) is doctrine-compliant and reuses today's SFU-PE path —
  acceptable if the launch-payload scheduler hook lags. The pull/poll/sentinel
  protocol is retired either way.

SimX-first: model FWD as the producer SimObject emitting fragment waves into the
existing wave-dispatch model, 1:1 with the RTL FWD for trace-diffable parity.

---

## 5. What remains (features to complete)

Beyond the §3 adjustments, to reach the north star:

### 5.1 Hardware FF datapaths on U55C (critical path)
- **OM v2 RTL** — mobile-class fixed-point ROP: depth/stencil/blend, write-masks,
  MRT; `vx_om4` windowed interface (C1–C5). *Currently SimX-only.*
- **TEX v2 RTL** — fixed-point sampler: point/bilinear, integer-mip then trilinear,
  wrap modes; `vx_tex4` single/quad. *Currently SimX-only.*
- **RASTER FWD RTL** (§4) + 300 MHz closure for the graphics cluster on U55C.
- **Composable FF taps** — advanced features (aniso, MSAA, programmable blend) as
  a thin SW layer over FF taps, shrinking both dedicated HW and the pure-SW path.

### 5.2 On-device front-end completeness
- Cull modes (front/back) + near-plane sub-triangle clip feedback in setup_k.
- Guardband clipping; min-z for Hi-Z; robust overflow/back-pressure on the binning
  queues (no host restart exists under full residency — overflow must be handled
  device-side).
- Parallel binning hardening (segmented queues / two-level bins if per-CTA fan-out
  is too wide); API-draw-order preservation through to OM.

### 5.3 CP graphics front-end (autonomous draw)
- The CP sequences VS → setup → bin → raster → FS → OM and programs FF config
  device-side, as a self-contained device program over resident memory — the
  "no host in the loop" mechanism. Built on the RTL CP.

### 5.4 On-device SIMT software fallback (mandatory completeness path)
- Device-side software rasterizer / sampler / ROP (`libgfx_sw`) covering exotic
  formats, blend/logic-op modes, MSAA resolve, etc. Three-tier per-unit dispatch:
  native FF → HW-composed → SIMT software. Retire the llvmpipe runtime path; keep
  `MESA_VORTEX_STRICT` mandatory in CI meanwhile.

### 5.5 Full-residency memory model
- Persistent device-resident allocator over one pinned slab (two-heap split:
  FF-pinned-PA vs shader-paged-VA); collapse FrontEndPool's 16 allocations into a
  pooled slab; plan against `VX_CAPS_VM_PINNED_*`. Only egress is present.

### 5.6 Compiler stage coverage
- Extend `vp_nir_to_llvm` beyond VS/FS/compute to GS, tessellation (control/eval),
  and task/mesh, plus device-side amplification glue.

### 5.7 Conformance + hardware bring-up
- Stand up the Vulkan CTS harness against vortexpipe (graphics path), first on
  SimX/rtlsim, then on the U55C bitstream at 4 cores.
- U55C graphics bitstream: integrate RASTER+OM+TEX FF into the XRT AFU, close
  timing at the target clock, validate on-card.

---

## 6. Schedule (phases to the north star)

Each phase is independently green (SimX-first, then RTL parity, then hardware).
Effort is relative, not calendar — gated on validation, per project rule
(*defer synth until rtlsim-green*).

| Phase | Goal | Contents | Exit gate |
|---|---|---|---|
| **P0 — Interface law** | stop the recurring bug | adopt C1–C5 (§1.3); SimX structural assertion + PR checklist; **CI parity matrix** (multi-core × multi-cluster × multi-drawcall) | matrix runs (red allowed) |
| **P1 — Correctness green (SimX)** | gfx suite 7/7 on SimX | §3.1 OM window fix; §3.3 pid width; §3.5 mesa `vx_om4`/`vx_tex4`; §3.6 build fix; decouple gfx window from RTU | gfx-on-SimX 7/7; matrix green on SimX |
| **P2 — RASTER dispatch v2** | kill the multi-core/multi-drawcall class | §4 FWD in SimX (push/launch, epoch FSM, static tiling); retire `vx_rast` pull | box/2-drawcall + multi-core byte-exact vs SimX gold |
| **P3 — RTL FF datapaths** | FF accel exists in hardware | OM v2 RTL, TEX v2 RTL, RASTER FWD RTL; §3.7 cycle model; SimX↔RTL parity on the matrix | rtlsim parity green on the matrix |
| **P4 — Autonomy + residency** | true GPU posture | CP graphics front-end (§5.3); residency allocator (§5.5); thin driver (no host binning/readback); SW fallback v1 (§5.4) | a frame renders device-resident, host-untouched, on rtlsim |
| **P5 — U55C bring-up** | 4 cores on-card | graphics AFU integration; 300 MHz closure; on-card vecadd-of-pixels → draw3d | draw3d correct on U55C @ 4 cores |
| **P6 — Conformance** | the north star | Vulkan CTS harness; SW fallback completeness for failing cases; compiler stage coverage (§5.6) as CTS demands | **Vulkan CTS pass on U55C @ 4 cores, FF raster+om+tex, on-device** |

Critical path to the north star runs P3 (RTL OM/TEX) → P5 (U55C) → P6 (CTS); P0–P2
unblock correctness and must precede P3 so the hardware is built against a
doctrine-clean interface.

---

## 7. Validation criteria (gates)

- **Functional (SimX):** `gfx_raster/om/tex/tex4*`, `gfx_pipeline_*`, `gfx_draw3d`
  match golden ±tolerance; **multi-core × multi-cluster × multi-drawcall** matrix
  byte-exact vs SimX gold (the box 2-drawcall case that fails today must pass).
- **Parity (SimX↔RTL):** the same matrix diffs clean on rtlsim; FF op
  scoreboard/handle assertion holds (no C3/C4 violation).
- **Hardware (U55C):** `gfx_draw3d` and the pipeline tests render correctly on-card
  at 4 cores via the xrt driver, using FF RASTER+OM+TEX; timing closed at target
  clock.
- **Conformance (the definition of done):** Vulkan CTS passes on U55C @ 4 cores,
  all rendering on-device, FF-accelerated, host as driver only.

---

## 8. Where the detail now lives (archived)

Consolidated from, and superseded by, this doc (history in
[../archives/](../archives/)):

- Vision: `gfx_v2_true_gpu_charter.md`
- Interface law: `gfx_v2_ff_simt_interface_doctrine.md`, `raster_sfu_interface_v2.md`,
  `gfx_v2_window_p1_0.md`, `gfx_v2_custom1_isa_allocation.md`
- Front-end: `gfx_v2_vertex_setup_pipeline.md`, `gfx_v2_tile_binning_redesign.md`,
  `gfx_v2_cp_graphics_frontend.md`
- FF units: `gfx_v2_ff_expansion_roadmap.md`, `gfx_v2_tex_v2.md`, `gfx_v2_tex4_p1/p2.md`,
  `gfx_v2_om_v2.md`, `gfx_v2_om4_p3.md`
- Completeness: `gfx_v2_software_fallback.md`, `gfx_v2_residency_allocator.md`,
  `gfx_v2_compiler_stage_coverage.md`, `gfx_v2_vortexpipe_driver.md`
- Eval/conformance: `gfx_v2_perf_evaluation.md`, `vulkan_conformance_proposal.md`
- Reviews: `archives/reviews/*` (the 8 area reviews + `review_v2.1_recommendations.md`)
