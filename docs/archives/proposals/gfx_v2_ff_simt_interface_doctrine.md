# gfx_v2 — Fixed-Function ↔ SIMT Interface Doctrine

**Status:** DRAFT addendum for review — *doctrine only; touches no unit yet.*
Open questions resolved into decisions (§6); remaining items are validation, not
choices.
**Scope:** the kernel-visible contract by which any fixed-function unit (RASTER,
TEX, OM, PRISM RTU) exchanges work and data with the SIMT shader cores. Defines
the *law* every FF↔SIMT interface must obey; the per-unit design docs own
bringing each unit into compliance.
**Relation to charter:** addendum to
[gfx_v2_true_gpu_charter.md](gfx_v2_true_gpu_charter.md). The charter fixes *who
does what*; this addendum fixes *how FF and SIMT exchange work* — the dimension
where the program keeps re-incurring the same class of bug.
**Evidence base:**
[review_v2.1_recommendations.md](../reviews/review_v2.1_recommendations.md) §3
(the §8 multi-draw determinism bug) and the RASTER↔shader multi-core /
multi-drawcall failure in
[raster_sfu_interface_v2.md](raster_sfu_interface_v2.md).
**Reference:** PRISM RTU ISA v2 ([rtu_isa_v2_proposal.md](rtu_isa_v2_proposal.md))
— the in-house, validated instance of this doctrine; AMD `image_bvh_intersect_ray`,
NVIDIA Uniform Register File (Turing+), Intel `MemRay`/`MemHit`.
**Date:** 2026-06-20.

---

## 1. Why this addendum exists

The same defect has now surfaced **three** times, in three units, across two
trees and two ABI generations:

| Instance | Where | Mechanism |
|---|---|---|
| RTU Phase-1 | SimX + kernel | ray issued field-by-field through per-(warp,lane) special registers; `get_after` scoreboard dance — *fixed by RTU ISA v2* |
| §8 multi-draw bug | gfx SimX (`sfu_unit.cpp`) | `vx_om4`/`SETW` carry **no rd → not scoreboarded**; the shared per-(warp,lane) graphics window is overwritten by the next fragment's `SETW` while the prior fragment's OM reads still drain → batch-shape-dependent depth divergence |
| RASTER multi-core / multi-drawcall | vortex_ci RTL (rtlsim) | `vx_rast` returns `pos_mask` but pid+bcoords latch through an **un-scoreboarded CSR side channel**; termination is a value-encoded **sentinel** (`pos_mask==0`); a cross-core arbiter carries sticky-done/flush state with no single frame owner → a core exits early and drops a draw call's fragments |

These are not three bugs. They are **one disease**:

> **un-ordered shared FF↔SIMT side-band state, handed off across work items
> (fragments / draws / warps / cores) with no scoreboard guarantee, producing
> deterministic but batch-/timing-dependent divergence.**

Each has been (or is about to be) fixed *locally* — a snapshot here, a flush
there. That is patching ends. The disease recurs because there is **no stated
law** that an FF↔SIMT interface must obey, so every newly-migrated unit is free
to re-introduce it. RTU ISA v2 discovered the cure empirically; this addendum
**names it** and makes it binding for all units.

---

## 2. The doctrine

Every FF↔SIMT interface — existing or new — must satisfy all five clauses. They
are not style preferences; each is the direct negation of a shipped bug. (C4 and
C5 were one clause in the first draft; they are split because *state-sharing* and
*lifecycle/completion* are distinct failure modes and a one-clause-per-bug-shape
doctrine is more enforceable.)

### C1 — Scope-partition the arguments
Classify every argument by its SIMT scope and give it its natural, cheapest home;
never route a value through a wider scope than it needs.

| Scope | Natural home |
|---|---|
| per-thread (divergent every lane) | register window (the compiler's allocation) |
| per-warp (uniform per issue) | **uniform-register channel** (endpoint), `vx_wgather` lane-pack (interim) |
| per-call (per operation, warp-uniform) | a source register operand |
| per-dispatch (constant for the launch) | DCR, host/CP-programmed |

*Negates:* RTU Phase-1's flattening of all scopes onto one per-lane special-register
port. *Reference:* RTU ISA v2 §2.

### C2 — Single-issue
One **architectural** instruction per logical operation (trace, sample, fragment
pop, ROP write). Internal macro-op / sequencer expansion is expected; what the
kernel issues and the scoreboard tracks is **one** op.

*Negates:* the N-op `set`/`get`/`set` marshalling protocols. *Reference:* RTU v2
`trace`+`wait`; windowed `vx_tex4`/`vx_om4`.

### C3 — Scoreboard-ordered register handoff
Every value the shader supplies to, or receives from, an FF unit moves through
**scoreboarded registers**, and the op **retires under the scoreboard**. No
architectural path may order a producer's write and a consumer's read of the same
datum by program order, wall-clock, or "it usually drains in time" alone.

*Negates:* the §8 bug exactly — `vx_om4`/`SETW` have no `rd`, so nothing orders
the window write against the draining reads. *Reference (correct):* RTU v2 `wait`
— "the window write **is** the scoreboard-ordered writeback" (rtu_isa_v2 §5.2);
`vx_tex4`'s rd-sync handle.

> **C2 + C3 are distinct.** A register-*window* ABI (C2) is necessary but **not
> sufficient**: TEX/OM moved to windows yet still reproduced the hazard because
> the window *handoff* was not scoreboarded (C3). Both must hold.

### C4 — No shared mutable side-band state
No CSR, window, or unit-internal datum may be **shared** across ops, warps,
cores, or units in a way that **outlives a single scoreboarded op**. State that
must persist (a pending async operation) is named by an explicit **handle**, not
an implicit "current" register the next op clobbers. A datum is owned by exactly
one in-flight op or one explicit handle at a time.

*Negates:* the RASTER per-(warp,pid,lane) CSR latch; the §8 shared graphics
window; the **single graphics window shared by RtuUnit/TexUnit/OmUnit** (review
T5 / P0-3).

### C5 — Explicit lifecycle & completion
Termination and completion are signaled by an **explicit handle or count**, never
a value-encoded sentinel. Any frame / draw / epoch lifecycle has a **single
owner** (one FSM), never a set of modules that must independently agree across a
boundary. Work items carry an **epoch tag**; stale-epoch packets are dropped
structurally, not by flush heuristics.

*Negates:* RASTER's `pos_mask==0` sentinel, the no-single-owner draw boundary
(the dropped-draw-call bug), and the arbiter's sticky-done / `frame_drained` /
`flush_trigger` agreement machinery. *Reference:* RTU v2's explicit
`wait`-on-handle.

---

## 3. Compliance audit of the four FF units

Graded against C1–C5. ✅ compliant · ⚠️ partial · ❌ violates.

| Unit | C1 scope | C2 single-issue | C3 scoreboarded | C4 no shared state | C5 lifecycle | Verdict |
|---|:--:|:--:|:--:|:--:|:--:|---|
| **PRISM RTU (v2)** | ✅ | ✅ | ✅ | ✅ | ✅ | **Reference — compliant** |
| **TEX (`vx_tex4`)** | ✅ | ✅ | ⚠️ rd-handle, input via shared `SETW` window | ❌ shared gfx window | ✅ | **Partial (C4)** |
| **OM (`vx_om4`)** | ✅ | ✅ | ❌ no `rd` (§8) | ❌ shared gfx window | ✅ | **Violates C3,C4** |
| **RASTER (`vx_rast`)** | ❌ pid/bcoords via CSR | ❌ pop + 13 CSR reads | ❌ CSR latch unscoreboarded | ❌ CSR latch + cross-core arb | ❌ sentinel, no single owner | **Worst — violates all five** |

Per-unit notes:

- **RTU** proves the doctrine is implementable on Vortex *today*: it reuses
  multi-register window ops + `vx_wgather` (rtu_isa_v2 §4), ~2 issued ops/ray,
  scoreboard-ordered writeback. It is the template.
- **TEX/OM**: the *windowing* (C1/C2) is right; the regression is C3/C4 — shared
  staging window, and (OM) an unscoreboarded write op.
- **RASTER** is the only unit still entirely on the pre-doctrine pattern and the
  only one violating C5. The reviews' RASTER items (P0-4 header, P1-3 cycles)
  touch *format* and *timing*, **not** this control contract — they will not stop
  the recurrence. RASTER needs a dispatch redesign under this doctrine; see §6.5
  and [raster_sfu_interface_v2.md](raster_sfu_interface_v2.md).

---

## 4. Enforcement (decided)

Doctrine prose is necessary but not sufficient — the disease recurred *under
review*. Enforcement is therefore mechanized, not advisory:

1. **PR checklist.** Every FF↔SIMT change answers C1–C5 explicitly in the PR
   description; a no on any clause blocks merge.
2. **SimX structural assertions.** SimX asserts, for every FF op, that it either
   (a) has a scoreboarded `rd`/handle, or (b) is provably side-effect-free; and
   that no FF datum is read by an op other than the one (or handle) that owns it.
   **A C3/C4 violation fails a test, not just a review** — this is what would have
   caught §8 automatically.
3. **CI parity matrix.** FF↔SIMT interfaces are exercised at the configurations
   that actually stress them — **multi-core × multi-cluster × multi-drawcall** —
   on both SimX and (where built) RTL, diffed for parity. The current gap
   (multi-core rtlsim never even built; multi-drawcall never in CI) is the
   process hole that let both bugs ship; closing it is part of the doctrine.

These land as **P0-0** (doctrine adoption + the §3 audit + items 1–3) above the
existing P0 list.

---

## 5. Resolved cross-cutting decisions

Made per the true-GPU goal (best performance + completeness), not the minimal
patch.

- **D-TEXOM — full decoupling, not the minimal snapshot.** TEX/OM get
  **per-unit, scoreboard-retired windows**; the cross-unit shared graphics window
  is removed entirely, retiring T5 / P0-3 and the "`vx_om4` aborts without RTU"
  coupling. Reframes review **P0-1**: do the *scoreboard-handle* variant for
  `vx_om4` (parity with `vx_tex4`/RTU `wait`), **not** the snapshot-at-F==0
  end-patch. The snapshot would satisfy the symptom; it leaves the shared window
  (a latent C4 violation) in place.
- **D-UNIFORM — uniform-register file is the C1 endpoint.** `vx_wgather`
  lane-packing is a *conforming interim*, but it is already saturated at warp=4
  (flags+cull co-pack, self-slot suppressed; rtu_isa_v2 §9 Q6) and does not
  scale. The architectural endpoint for the per-warp scope is a real
  uniform-register file (NVIDIA-since-Turing; review **P2-4**). New interfaces
  target it; existing wgather users migrate opportunistically.
- **D-RASTER — push/launch is the RASTER endpoint.** See §6.5.

---

## 6. RASTER control redesign — decided shape

RASTER is the one non-compliant *control* path the reviews don't cover. The
decision (true-GPU-aligned, highest performance, and correct-by-construction):

- **Endpoint: push/launch.** A hardware **Fragment Work Distributor** packs
  covered quads into fragment waves and **launches** them onto shader cores —
  payload (coverage, x/y, pid, barycentrics) delivered as wave inputs (C1/C2/C3),
  completion **counted** (C5), no sentinel, no cross-core arbiter (C4). This makes
  the dropped-draw-call class structurally impossible.
- **Unify with the compute/CTA dispatch path** (charter §8.4) so fragment work is
  "just another work source" for the warp scheduler — built with the same
  occupancy/retirement primitives, not a parallel launcher.
- **Static screen-space tile ownership** for distribution: each tile → one core,
  so raster order and OM read-modify-write are correct *by construction* (no pixel
  is touched by two cores). This subsumes the cross-core OM-ordering concern into
  one front-end property.
- **Stepping stone (optional, de-risking only):** a scoreboard-ordered windowed
  `frag.payload` pull op (single issue, handle-terminated) is doctrine-compliant
  and reuses today's SFU-PE result path; acceptable as an interim landing if the
  push path needs a scheduler hook that isn't ready. The pull/poll/sentinel
  protocol is retired regardless.

Detailed in [raster_sfu_interface_v2.md](raster_sfu_interface_v2.md), to be
recast as the RASTER-compliance subsystem doc under this doctrine.

---

## 7. Non-goals

- Does **not** touch FF datapaths, BVH formats, binning structure, or the FF-vs-SW
  split — charter §6/§8 concerns. This is purely the *interface contract*.
- Does **not** require a register window where a single register suffices (C2 is
  "one issued op," not "always a window").
- Does **not** change per-dispatch DCR programming (already the C1-correct home).

---

## 8. To validate in the subsystem docs (not open choices)

1. Confirm the scoreboard-handle `vx_om4` closes the §8 146-byte diff → 0 and
   matches RTL ordering (review P0-1 acceptance).
2. Confirm per-unit TEX/OM windows build and run **without** `-DVX_CFG_EXT_RTU_ENABLE`
   (retires T5).
3. Confirm the RASTER FWD launch hook against the CTA-dispatch path (charter §8.4
   / kmu_cta_dispatch redesign) — reuse vs. interim parallel port is an
   implementation detail of the subsystem doc, governed by which lands first.
4. Confirm the CI parity matrix (multi-core × multi-cluster × multi-drawcall)
   green on SimX, then RTL parity.

---

## 9. Recommendation

Adopt C1–C5 as a binding gfx_v2 invariant (this addendum), publish the §3
compliance table, mechanize enforcement (§4), and route the existing P0-1/P0-3
fixes through the *doctrine-compliant* variants (§5). Then bring RASTER dispatch
under the doctrine via its subsystem doc (§6). With the law named, the four units
audited, and violations caught by test rather than review, the recurring-patch
failure mode closes by construction.
