# gfx_v2 — RASTER ↔ SIMT Dispatch Interface v2 (subsystem doc)

**Status:** DRAFT subsystem design under the FF↔SIMT doctrine.
**Scope:** the control + data contract between the fixed-function rasterizer and
the SIMT shader cores — i.e. how fragment work and its attributes get from RASTER
into a fragment shader. This is charter **§6.3** ("Rasterizer front-end
redesign") made concrete on the interface axis, and it brings RASTER into
compliance with the FF↔SIMT doctrine.
**Parent docs:**
[gfx_v2_true_gpu_charter.md](gfx_v2_true_gpu_charter.md) §6.3, §8.4;
[gfx_v2_ff_simt_interface_doctrine.md](gfx_v2_ff_simt_interface_doctrine.md)
(C1–C5).
**Reference instance:** PRISM RTU ISA v2
([rtu_isa_v2_proposal.md](rtu_isa_v2_proposal.md)) — the doctrine-compliant FF
unit this design mirrors. SOTA: NVIDIA GigaThread/raster-tile front end, AMD
RDNA scan-converter→packer→wave-launch, Mali/Adreno tile ownership.
**Date:** 2026-06-20.

> RASTER is the worst-graded unit in the doctrine §3 audit — it violates **all
> five** clauses, and it is the only unit whose *control* path (poll / sentinel /
> cross-core arbiter) the v2.1 reviews do not address. This doc fixes that.

---

## 1. The current interface and how it violates the doctrine

### 1.1 Programming model (per quad)

```c
vx_rast_begin();                       // OR-reduced "begin" pulse
for (;;) {
    uint32_t pos_mask = vx_rast();     // POP: returns ONLY {pos_y,pos_x,mask}
    if (pos_mask == 0) return;         // value-encoded sentinel → exit
    uint32_t pid = csr_read(VX_CSR_RASTER_PID);     // side-band #1
    F0 = csr_read(VX_CSR_RASTER_BCOORD_X0); ...     // side-band #2..#13
    // interpolate → vx_tex → vx_om
}
```

One logical fragment quad is delivered across **1 SFU result + up to 13 CSR
reads**, with the CSR payload latched out-of-band into per-`(warp,pid,lane)`
LUTRAM at pop-retire. Cross-core distribution + termination ride a shared
arbiter (`VX_raster_arb`) carrying sticky-done / `frame_drained` / `flush`.

### 1.2 Doctrine violations (the bug surface)

| Clause | Violation | Symptom |
|---|---|---|
| **C1** scope-partition | pid/bcoords (per-thread) routed through CSR, not the register window | bloated per-quad cost |
| **C2** single-issue | pop + 13 CSR reads per quad | ~14 ops/quad |
| **C3** scoreboarded handoff | CSR latch is **not** scoreboarded against the CSR read | cross-warp/thread hazard |
| **C4** no shared side-band | per-`(warp,pid,lane)` CSR latch + cross-core arb sticky state | multi-core corruption |
| **C5** lifecycle/completion | `pos_mask==0` **sentinel**; no single owner of the draw boundary | **dropped 2nd draw call at cores≥2** |

The dropped-draw-call failure (top face of the box missing at cores=2, while the
draw call in isolation is byte-exact vs SimX gold) is a direct C5 violation: the
sentinel from draw call *N* leaks into *N+1* because no single FSM owns the
boundary, and a core exits before rendering its share. The C4 CSR latch and the
arbiter's flush epicycles are the patches that have accreted trying to hold C5
together without owning it.

---

## 2. v2 design — push/launch Fragment Work Distributor

Flip pull → **push**. The cluster rasterizer becomes a **Fragment Work
Distributor (FWD)**: it pulls covered quads from the (unchanged) TE/BE raster
math, packs `NUM_THREADS` covered quads into a **fragment wave**, and **launches**
that wave onto a shader core through the warp-dispatch path — payload delivered as
wave inputs, completion counted. Shader threads never poll the rasterizer; the
`vx_rast` loop, the bcoord CSRs, and the arbiter's sticky/flush logic are
**removed**.

```
            cluster                                   per-core
 ┌─────────────────────────────┐        ┌────────────────────────────────────┐
 │ RASTER pipeline (TE/BE)      │ quads  │ warp scheduler (compute + fragment) │
 │   ↓ covered quads            ├───────▶│   ← FWD launches fragment waves     │
 │ Fragment Work Distributor    │ wave   │   payload = wave inputs (C1/C2/C3)  │
 │  - epoch FSM (C5, single own)│ launch │   FS runs → vx_tex4 → vx_om4        │
 │  - static tile→core map (C4) │        │                                     │
 │  - launched/retired counters │◀───────┤   retire ⇒ count++                  │
 └─────────────────────────────┘ retire └────────────────────────────────────┘
```

### 2.1 How v2 satisfies each clause

- **C1 scope-partition.** Coverage/x/y/pid/barycentrics are per-thread → the
  fragment wave's **register window** (the FS's input allocation, like
  `gl_FragCoord`/`gl_PrimitiveID`/barycentric inputs). Per-dispatch raster config
  stays in DCRs. No per-thread datum rides a CSR.
- **C2 single-issue.** The shader issues **zero** pop/poll ops — work arrives as a
  launched wave; the payload is just its input registers. (Interim variant §5
  keeps exactly one scoreboarded op.)
- **C3 scoreboarded handoff.** Payload lands in scoreboarded registers at wave
  launch; the FS reads them under normal scoreboard semantics. No un-scoreboarded
  side channel.
- **C4 no shared side-band.** The per-`(warp,pid,lane)` CSR LUTRAM is deleted.
  **Static screen-space tile ownership** (each tile → one core) means no pixel is
  ever touched by two cores, so the cross-core arbiter and its sticky/flush state
  are gone — and OM read-modify-write becomes correct *by construction* (this
  subsumes the OM cross-core ordering concern into one front-end property).
- **C5 lifecycle/completion.** A draw is an **epoch-tagged** transaction owned by
  the FWD FSM: `QUIESCED → FILLING → DRAINING → QUIESCED`. Completion is
  `producer_drained ∧ launched == retired`, reported once. No sentinel; stale-epoch
  waves are dropped structurally.

### 2.2 Lifecycle FSM (single owner — C5)

```
QUIESCED ──(CP draw cmd, epoch e)──▶ FILLING ──(producer emits all quads)──▶ DRAINING
   ▲                                                                          │
   └───────────────(launched[e] == retired[e]  ∧  producer_drained[e])────────┘
                       └─ completion reported to CP/host exactly once
```

`begin` is no longer an OR-reduced pulse cores race on; the **CP issues the draw**
(charter §6.4) carrying the epoch. Cores only ever receive *launched waves*, which
exist only after the producer is armed — so a core cannot observe a half-armed
producer, and cannot "finish early" (it does not decide completion; the FWD does).

### 2.3 Distribution (C4 / raster order)

**Static screen-space tile ownership.** Tiles are statically mapped to cores; all
fragments of a tile (hence any pixel) shade and ROP on one core. Consequences:
- raster order / API-draw order preserved per pixel for the OM with no cross-core
  machinery;
- OCACHE locality per core;
- deterministic, matching the SimX model.

Load imbalance on geometry-skewed scenes is the known cost; a coarse work-steal
refinement (with the stolen tile's ownership transferred atomically) is a later
perf option, not needed for correctness.

### 2.4 Payload delivery

The fragment-wave payload (≤ 4-bit coverage, x, y, pid, 3×4 barycentrics) lands as
the wave's input registers at launch (the C1 "per-thread register window"). This
is the SOTA endpoint and removes all shader-side load/poll cost. It requires a
scheduler hook to seed a per-lane register block at wave launch — the one new
launch-path capability v2 introduces (see §4 / §5 interim).

---

## 3. Dispatch integration (charter §8.4)

Fragment waves enter the core through the **warp scheduler**, built on the same
occupancy/retirement primitives as compute CTAs so the two converge. Per the
true-GPU goal, the endpoint is **unification with the CTA/compute dispatch path**
("fragment work is just another work source"); the implementation may start as a
thin fragment-launch port that *reuses* those primitives and merges once the
KMU/`VX_cta_dispatch` redesign ([kmu_cta_dispatch_redesign.md]) lands. Which lands
first is an implementation detail, not a design fork — both target the same
scheduler contract.

---

## 4. SimX-first plan & RTL parity

SimX is gold and already models the correct functional result; v2 changes the
*mechanism*, not the image.

1. **SimX v2:** refactor `sim/simx/raster/*` so the **producer + FWD** is the
   SimObject and fragment waves enter the existing wave-dispatch model, mirroring
   the RTL FWD 1:1 for trace-diffable parity. Graphics suite green in SimX at
   multi-core × multi-cluster × **multi-drawcall** (the doctrine §4 CI matrix).
2. **RTL FWD + payload staging:** bring rtlsim to SimX parity on that full matrix.
3. **Delete legacy:** remove `vx_rast`, the raster bcoord CSRs, and the
   `VX_raster_arb` sticky/flush machinery; update `vx_graphics.h`, the mesa/
   vortexpipe FS lowering, and the gfx kernels.

The functional result must not change (still matches the golden ±tolerance);
only the dispatch mechanism does.

---

## 5. Interim stepping stone (de-risk only)

If the register-prestage scheduler hook (§2.4) is not ready when the rest is, a
**doctrine-compliant pull** is acceptable as an interim landing: a single
`frag.payload rd0..rdK` op that returns the entire stamp for the lane in one
**scoreboarded** issue (C2/C3), terminated by an explicit handle/`done` flag
(C5), with the FWD still owning distribution and the epoch FSM (C4/C5). This
reuses today's SFU-PE result path and immediately retires the sentinel, the CSR
latch, and the cross-core arbiter — i.e. it fixes the *bugs* even before the full
push lands. The pull/poll/sentinel protocol is retired either way.

---

## 6. Perf / area / timing (U55C @ 300 MHz)

- **Shader-side:** ~14 ops/quad → ~0 (push) or 1 (interim). Removes CSR-port
  pressure and the poll loop.
- **Removed HW:** per-`(warp,pid,lane)` raster CSR LUTRAM; the arbiter's
  sticky-done/activity/flush + fan-out RR + OUT_BUF flush; the raster-core
  `fetch_triggered` epicycle. The cluster arb collapses to a producer→FWD stream.
- **Added HW:** FWD packer (FIFO) + launched/retired counters + payload staging
  write port. Net area expected neutral-to-positive; control state moves from
  three modules into one well-scoped FSM.
- **Timing:** removes today's arb critical paths (combinational `done_all`,
  `all_active_served` reduction across outputs, RR mux). Packer is FIFO+counter;
  payload staging is a write port. No obvious new 300 MHz hazard. Per charter
  invariant + project rule: **defer synth until rtlsim-green**; this is a
  design-time estimate.

---

## 7. To validate (not open choices)

1. Register-prestage hook at wave launch — confirm the scheduler can seed a
   per-lane payload block (else fall to §5 interim).
2. Static-tile-ownership balance on the gfx test scenes — confirm acceptable
   utilization; quantify the skew case.
3. SimX↔RTL parity on the doctrine §4 matrix (multi-core × multi-cluster ×
   multi-drawcall), including the box (2 draw calls) that fails today.
4. OM cross-core RMW concern is fully subsumed by tile ownership — confirm no
   residual OM ordering case remains once RASTER owns tiles.

---

## 8. Recommendation

Adopt the push/launch FWD as the RASTER↔SIMT interface, with static tile
ownership and the epoch-FSM lifecycle. It brings RASTER into full C1–C5
compliance, makes the dropped-draw-call and cross-core-OM classes impossible by
construction, cuts ~14 ops/quad to ~0, and deletes more control state than it
adds. Sequence SimX-first; use the §5 scoreboarded-pull interim only if the
launch-payload hook lags.
