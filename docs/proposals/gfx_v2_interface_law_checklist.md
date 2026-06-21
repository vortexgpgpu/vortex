# FF↔SIMT interface-law PR checklist (gfx_v2 §1.3, P0)

**Status:** CURRENT. The mechanized enforcement surface for the binding interface
law. Companion to [gfx_v2_true_gpu.md](gfx_v2_true_gpu.md) §1.3 (the law) and §3
(the in-flight defects it governs).

The graphics program kept re-incurring one bug class: an *un-ordered shared
FF↔SIMT side-band handoff* — a value moved across work items through shared
mutable state with no scoreboard guarantee (the §8 OM multi-draw determinism bug;
the RASTER multi-core / multi-drawcall dropped-draw-call bug). This checklist plus
the SimX structural assertion plus the CI parity matrix make a recurrence
*fail the build or the matrix*, not slip through review.

## When this applies

Any PR that adds or changes a fixed-function op or its FF↔SIMT interface:

- a new CUSTOM1 op (TEX / OM / RASTER / RTU / graphics-window), or a change to an
  existing one's operands, destination, or completion;
- any state read or written across the FF↔SIMT boundary (register windows,
  uniform channels, DCRs, cross-unit/cross-core latches);
- the SimX model, RTL, kernel intrinsic, or driver emitter for any of the above.

## The five clauses (C1–C5)

Tick each, or justify the exception inline in the PR description.

- [ ] **C1 — Scope-partition.** Every argument lands in its natural home:
  per-thread → register window; per-warp → uniform channel / `vx_wgather`;
  per-call → source register; per-dispatch → DCR. No per-thread value smuggled
  through a per-warp channel or vice-versa.
- [ ] **C2 — Single-issue.** One architectural op per logical operation. Internal
  macro-op expansion (sequencer uops) is fine; multiple *architectural* ops to
  express one operation is not.
- [ ] **C3 — Scoreboard-ordered handoff.** Every value crossing FF↔SIMT moves
  through a scoreboarded register and the op retires under the scoreboard
  (destination register / completion handle). A register *window* alone (C2) is
  **not** sufficient — the *handoff* itself must be scoreboarded.
- [ ] **C4 — No shared mutable side-band.** No state that outlives a single
  scoreboarded op: no per-(warp,pid,lane) CSR latch, no cross-unit shared window,
  no cross-core arbiter sticky state.
- [ ] **C5 — Explicit lifecycle & completion.** Completion by handle/count, never
  a value-encoded sentinel (e.g. `pos_mask==0`). A draw/frame epoch has a single
  owner (one FSM) with epoch-tagged work items.

## Mechanized gates (must pass before merge)

1. **SimX structural assertion** — `sim/simx/gfx_doctrine.h`. Every FF op decoded
   in `EXT2` is classified by `gfx_doctrine::classify()` into exactly one handoff
   class and checked against its decoded destination:
   - **Scoreboarded** — retires under a destination register/handle (C2/C3).
   - **SideEffectFree** — no architectural effect the SIMT pipeline does not
     already order.
   - **KnownViolation** — an enumerated §3 defect (today: `vx_om4`, `SETW`);
     warns once, passes the default build, **aborts under
     `VX_GFX_STRICT_DOCTRINE=1`**.
   - **Unclassified** — aborts unconditionally. *A new FF op cannot reach the
     pipeline without declaring its handoff class here.* This is the line that
     stops the recurrence.
   - **PR rule:** a new FF op adds its `classify()` entry in the same PR. A new
     op is **never** `KnownViolation` — that class is frozen to the existing §3
     debt. As each §3 defect is fixed, its entry moves to `Scoreboarded` /
     `SideEffectFree` and `VX_GFX_STRICT_DOCTRINE=1` is flipped on in CI.

2. **CI parity matrix** — `ci/testcases/graphics_parity.yaml`, category
   `graphics_parity`. Sweeps multi-core × multi-cluster × multi-drawcall and runs
   each cell on **both** SimX and rtlsim against the shared golden (both
   byte-matching the same reference ⇒ SimX↔RTL parity). This is the dimension
   that hid both interface bugs (single-core/single-cluster tests passed while
   the cross-core / multi-drawcall handoff dropped work). P0 gate: the matrix
   *runs* (red cells allowed — they are the §3 work). P1/P2 gate: it goes green.

## Reviewer prompt

> Does this op hand a value across the FF↔SIMT boundary through anything other
> than a scoreboarded register? If yes, it violates C3/C4 — request the
> scoreboarded variant, not an end-of-sequence patch.
