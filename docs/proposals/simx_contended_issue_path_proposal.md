# SimX Contended Issue-Path Modeling — Investigation Record

## Problem

Two `model_parity` cases are `known_issue`-marked after the LSU/divide
timing alignment (see `simx_lsu_timing_alignment_proposal.md`):

| case | simx cycles | rtlsim cycles | gap | tolerance |
|---|---|---|---|---|
| `core:parity-sgemm-mc` (n128, NT8/NW8, 2 cores, L2) | 802,943 | 1,103,199 | 27.2% | 10% |
| `dxa:model_parity-wgmma-sp-dxa` | 61,800 | 74,866 | 17.5% | 10% |

Retired instructions match exactly on both; the divergence is purely the
cycle model under full-occupancy contention. The RTL's producer-commit →
dependent-dispatch turnaround measures 55/92/146 cycles (p50 across the
three dependent-FMADD chains) against SimX's 8/11/25.

## Root cause (established)

A cycle-accurate audit of the hardware issue→execute→commit→scoreboard loop
at the parity shape (NUM_WARPS=8, NUM_THREADS=8 ⇒ ISSUE_WIDTH=1: one issue
slice, one commit beat per cycle for the whole core) identified the
serializers SimX under-models, each verified by trace measurement:

1. **Per-fragment load commit.** A coalesced load returns N response beats;
   each takes one grant on the single core-wide commit port and only the
   last clears the scoreboard. Modeling this in SimX reproduces the
   hardware's beat demand almost exactly (fp16 tensor workload: 16,764
   hardware commit beats vs 17,519 modeled; TCU/ALU/FPU/SFU class counts
   identical). SimX today retires every load in one beat.

2. **Fixed-priority commit arbitration** in EX-unit order — ALU > LSU >
   SFU > FPU > TCU — so a hot ALU stream starves load writebacks, delaying
   dependent warps' scoreboard clears. SimX arbitrates round-robin.

3. **Constant post-dispatch overhang.** SimX's FU-output→commit path takes
   3 channel hops where the hardware commits in one registered stage;
   per-class dispatch→commit latency measures a uniform ~+4 cycles vs the
   RTL (ALU p50 6 vs 2). Conversely SimX's issue side is optimistic
   (same-cycle dispatch-credit return where the hardware takes two
   registered stages; a bypass that issues a suppressed warp when every
   ready warp faces a full queue). These errors cancel on uncontended
   chains — which is exactly why the suite passes at low occupancy and
   diverges at 8-warp contention, where the optimistic terms scale and the
   pessimistic ones do not.

## What was tried, and why it is not landed

The mechanisms were implemented and validated individually (fragment-beat
commits with writeback-less beats, fixed-priority commit selection, a
collapsed 1-stage commit path, per-FU commit-side elasticity, 2-cycle
credit return, hard suppression). Eleven configurations were swept against
sentinel cases and twice against the full 29-case parity suite. Findings:

- The best configuration brings `parity-sgemm-mc` to **+3.9%** and every
  original sentinel into tolerance — the mechanisms demonstrably close the
  gap — but every configuration that fixes the target cases regresses
  other passing cases by 5–30% in one direction or the other.
- The regressions are unmaskings, not new errors. Example:
  `parity-sgemmx` passes today at +6.7% only because SimX's shallow
  2-entry FU output channels accidentally reproduce ~1M cycles of
  writeback serialization the hardware also has; giving any FU realistic
  commit-side elasticity releases that overlap and exposes a latent ~30%
  model divergence. Similarly, the small tensor cases pass only because
  the optimistic issue side compensates the +4-cycle post-dispatch
  overhang; adding the faithful credit/suppression timing without first
  removing the overhang over-serializes them by 5–15%.
- Per AGENTS.md, trading two tracked xfails for five-to-seven untracked
  regressions is not landable. The experiments are reverted; this document
  and the measurement tooling are the record.

## Follow-up program (the actual fix)

Closing the two known_issues requires removing the error cancellations in
dependency order, validating the full parity sweep at each step:

1. **Align the constant post-dispatch depth** stage-by-stage
   (operand-collector, dispatch, FU transit, commit) so each class's
   uncontended dispatch→commit latency matches the hardware without relying
   on the optimistic issue side. This is where `parity-sgemmx`'s latent
   divergence must be resolved.
2. **Land the faithful issue side** (two-stage credit return, hard
   suppression) once step 1 removes the overhang it compensates.
3. **Land the contention serializers** (fragment-beat commits,
   fixed-priority commit arbitration, calibrated commit-side elasticity).
   With steps 1–2 in place these have nothing left to unmask, and the
   measured +3.9% result for `parity-sgemm-mc` bounds the achievable gap.

Measurement tooling (all reusable from `--debug=3` traces on both drivers,
which share UUIDs): per-UUID commit-beat counting per FU class, per-class
dispatch→commit latency tables, load segment decomposition
(dispatch→request→first/last response→commit), response inter-beat
spacing, and producer-commit→consumer-dispatch turnaround distributions.

## Current CI state

Both cases remain `known_issue` xfails at tolerance 0.10
(`ci/testcases/core.yaml`, `ci/testcases/dxa.yaml`), with this document as
the triage record and the follow-up scope.
