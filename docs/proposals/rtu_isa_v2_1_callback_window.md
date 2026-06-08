**Date:** 2026-06-07
**Status:** implemented in SimX — addendum to
[rtu_isa_v2_proposal.md](rtu_isa_v2_proposal.md).
**Branch:** `prism`

# PRISM RTU — ISA ABI v2.1 (callback-side window read + sync trace)

## 1. Motivation

ABI v2 collapsed the *issue* path: `vx_rt_trace2` passes the ray through the
`f0–f7` register window and `vx_rt_wait2` retires the hit through a register
window, eliminating the per-word `vx_rt_set` / `vx_rt_get` marshalling on the
normal trace→wait flow. It deliberately **retained** the per-(warp,lane)
register file as the in-trap callback payload window (v2 §5.5) — but it did
**not** apply the same window collapse to the *callback read* path. A dispatcher
that needs several candidate fields still reads them one slot at a time:

| Dispatcher | Reads | Phase-1 cost |
|---|---|---:|
| Intersection (IS, ray-sphere) | object-ray origin+dir (slots 8–13) | 6× `vx_rt_get` + 6× `fmv` = **12 ops** |
| Closest-hit (fat shader) | t/u/v + IDs + attrs | 6–10× `vx_rt_get` |

This is exactly the field-by-field marshalling v2 removed from the trace path,
just confined to the trap. v2.1 closes it.

## 2. GETWF — FP windowed regfile read

A new macro-op reads `count` **contiguous float slots** starting at `start`
into an FP register group, in one fetched instruction — the callback-side
analog of `WAIT2`'s writeback window. It expands (per-warp sequencer) into
`count` micro-ops, each streaming one regfile slot into one FP register, NaN-
boxed; no `int→float` conversion (the slots already hold f32 bits).

**Encoding** (additive; CUSTOM1, funct3 = 6 — the callback-op group):

| field | meaning |
|---|---|
| funct2 = 2 | sub-op GETWF (funct2 0 = `CB_RET`, 1 reserved, 3 free) |
| funct7[6:2] | window **start slot** (0–31) |
| rs2 index | window **count** (1–8; the register field is an immediate) |
| rd | FP window **base register** (writes rd..rd+count-1) |

No source operands — the data comes from the RTU regfile (staged by the
yielding trace's `apply_callback_payload`). Soundness: a dispatcher already
treats its target registers as scratch (there is no register-value save across
the async trap — only the scoreboard snapshot); writing an FP group is no
different from the individual gets it replaces. Dispatchers that run real FP
(the IS shader) carry `__attribute__((interrupt("machine")))`, so the compiler's
prologue already saves the window registers.

**Kernel API.** A typed accessor for the common object-ray case:

```c
vx_objray_t objray;
vx_rt_get_objray(&objray);   // 1 macro-op -> f0..f5; was 6 get + 6 fmv
```

Ported the `rtu_smoke_proc` intersection dispatcher to it (12 ops → 1 fetched
instruction). Further windows (e.g. a hit window for fat CHS) add their own
typed accessor over the same GETWF encoding.

## 3. Sync trace (§9.3) — resolved, no new opcode

v2 §9.3 asked whether a fused `vx_rt_trace_sync(cfg, ray, &hit)` earns an
opcode. **It does not.** A true fused instruction would have to *park*
mid-macro-op (between arm and writeback), adding sequencer/scoreboard
complexity, and it would forfeit the async overlap that motivates the
trace/wait split — to save a single instruction fetch (the handle never leaves
a register). So `vx_rt_trace_sync` ships as a thin inline wrapper
(`trace2` + `wait2`); when the kernel has independent work, it issues the two
separately and the compiler schedules into the gap.

## 4. Trap-safe `wait2` (prerequisite for retiring v1 TRACE/WAIT)

The Phase-1 single-issue `vx_rt_trace` / `vx_rt_wait` (CUSTOM1 funct3=5 sub-op
2/3) are made retirable but **kept for now** — the Mesa/Vulkan RT lowering
(`vp_nir_lower_ray_tracing_to_rtu`, the `vortex_rt_trace`/`vortex_rt_wait` NIR
intrinsics) still emits them, so they cannot be removed until the step-6 Mesa
migration moves to the v2 ISA. All hand-written kernels now use the v2 ISA. The
reason v1 `WAIT` survived is that it is a *single* parked op the callback trap
can revive, whereas the v2 `WAIT2` was a *parking macro-op* that could not
survive the async trap. v2.1 fixes that so wait2 is callback-safe:

- **`WAIT2` split into block + writeback.** `WAIT2` (funct3=7 sub-op 1) is now a
  SINGLE-OP block — it reuses the v1 park/revive path, so it survives a callback
  trap exactly like the old `WAIT`. The hit window is delivered by two *separate*
  non-blocking windowed reads the `vx_rt_wait2` intrinsic emits next: `GETWF`
  (t/u/v → f0..f2) and **`GETW`** (the GP twin: primitive/instance/geometry IDs
  → t3..t5), each scoreboard-chained on the status word so they run post-terminal.
  (`GETW` is funct3=6 sub-op 3; it is now exercised, so it's implemented — not a
  skeleton.)

- **Two scheduler fixes were required for any macro-op to survive a callback
  trap** (these are the real enablers, not the ABI shape):
  1. *Resume-on-trap.* A `wstall` macro-op (GETWF/GETW/TRACE2) suspends the warp
     until it commits. If the async trap flushes it mid-flight it never commits,
     so its `resume_warp` never fires — the warp hangs. `raise_async_trap` now
     resumes the warp it is taking over.
  2. *mret/trap serialization.* A callback trap raised the *same cycle* an `mret`
     retired corrupts the warp's tmask/PC (restored vs newly-trapped contexts
     race), skipping the next dispatcher's `cb_ret`. The RTU callback drain now
     defers a trap one cycle past an `mret` (the reformation multi-group path,
     `rtu_smoke_reform_sbt`).

With these, all callback kernels (AHS/IS/CHS/MISS/SBT), reformation
(`reform`/`reform_mw`/`reform_sbt`), and recursion run on `wait2`; the recursive
dispatcher's nested ray is now `trace2`/`wait2` in-trap. `SET`/`GET` and (for
now) v1 `TRACE`/`WAIT` remain in the ISA — the latter pending step-6 Mesa.

## 5. Status

- Implemented in SimX: `WAIT2` block + `GETWF`/`GETW` windowed reads and the two
  scheduler fixes. `vx_rt_get_objray` / `vx_rt_trace_sync` in the kernel header.
- **24/24 `tests/raytracing/*` pass on SimX.** All hand-written kernels on v2.
- **v1 TRACE/WAIT retirement is deferred**: `tests/vulkan/rt*` (rtquery*) reach
  the RTU through Mesa, whose lowering still emits the v1 ops. Retire them only
  after step-6 migrates `vp_nir_lower_ray_tracing_to_rtu` to `trace2`/`wait2`.
  (NOT yet run on SimX here.)
- Out of "in SimX" scope: RTL decode for funct3=6 (GETWF/GETW) and the v2
  funct3=7 ops; a writeback `SETW`. Trap-per-yield latency is microarchitecture,
  not ABI.
