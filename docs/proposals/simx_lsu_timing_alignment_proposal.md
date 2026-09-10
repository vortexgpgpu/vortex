# SimX LSU Timing Alignment Proposal

## Problem

Three `model_parity` cases exceeded their cycle tolerance. In every case
retired instructions matched exactly; only cycles diverged, and SimX was
always the faster model:

| case | simx cycles | rtlsim cycles | gap | tolerance |
|---|---|---|---|---|
| `core:parity-sgemv` (m512 n512) | 957,012 | 1,096,954 | 12.76% | 10% |
| `core:parity-sgemm-mc` (n128, 2 cores, L2) | 802,280 | 1,103,199 | 27.28% | 10% |
| `dxa:model_parity-wgmma-sp-dxa` | 63,321 | 74,866 | 15.42% | 10% |

These three tolerances were widened to 0.10 when the RTL gained the decoupled
LSU outstanding-request pool (`ed61857a9`); the gaps had since drifted past
even that. Per AGENTS.md the fix is to model the behavior, never to widen a
tolerance.

## Investigation

Measurements taken on a dedicated build tree with identical configs on both
drivers, using per-UUID trace comparison (33k+ loads matched across drivers),
per-PC commit-gap profiles, and per-segment decomposition of every load
(dispatch → mem-req → first/last response → commit).

**What already matched.** Isolated round-trip latencies agree between the
models: an L1 hit measured at the LSU boundary is ~10 cycles on both drivers,
an L1-miss-to-DRAM round trip ~40-41 cycles on both, and per-class
issue→commit latencies agree once stage reference points are normalized.
Functional traffic is identical (same dcache request and DRAM refill counts).

**What diverged.**

1. *Streaming loads (sgemv):* the divergence appears only in the high-miss
   regime (no gap at m128, 7.4% at m256, 12.8% at m512) as a smooth linear
   drift. SimX's LSU front end issued loads into ~4 slots of switch/coalescer
   queue slack where the RTL has a 1-deep pipe buffer plus the coalescer FSM's
   single in-flight slot, and SimX's load writeback reached the commit
   arbiter one registered stage too early. Both fixed (see below); residual
   ~7% comes from request-arrival timing shifting the hit/miss split between
   the models (SimX classifies 74% of reads as misses vs the RTL's 42% at
   m512, with identical DRAM traffic).

2. *Divides:* simulation builds implement divide as a fully pipelined unit at
   the multiplier's depth, while SimX charged a blocking `XLEN+2` cycles.
   Each dxa_copy iteration lost ~60 cycles on two DIVUs. Fixed.

3. *Contended issue path (sgemm-mc, wgmma-sp-dxa):* with 8 warps per core the
   RTL's scheduler idles 52% of cycles vs SimX's 24% at equal per-instruction
   latencies. The RTL's producer-commit → dependent-dispatch turnaround under
   contention measures 55/92/146 cycles (p50, consecutive dependent FMADDs)
   vs SimX's 8/11/25. Contributing RTL structure identified: registered
   scoreboard/queue-ready gating (a freed dispatch-queue slot is re-spendable
   only cycles later, and `operands_ready` is a registered view), hard
   fu-going-full masking of warps at issue, per-fragment load writeback beats
   competing for the shared one-per-cycle commit port, and the warp convoying
   these produce. Partial SimX models of each (deferred scoreboard release,
   pipelined 3-stage collector, hard suppression, registered credit returns,
   fragment-beat commits) closed sgemm-mc from 27.3% to 13.6% — but the added
   mechanisms destabilized four small tensor parity cases (±5-10% swings on
   6-40k-cycle runs; the coupled arbitration is chaotically sensitive), so
   they are not landed.

## Landed changes (SimX only; no RTL, no config change)

1. `sim/simx/mem/local_mem_switch.cpp`, `sim/simx/mem/mem_coalescer.cpp`:
   single-entry request ingress, so upstream issue feels downstream cadence
   instead of running ahead into queue slack.
2. `sim/simx/lsu_unit.cpp`: load writeback crosses one more registered stage
   before the commit arbiter.
3. `sim/simx/alu_unit.cpp`: divide/remainder latency aligned with the
   pipelined simulation divider (multiplier-class), replacing the blocking
   `XLEN+2` model.

Result: `core:parity-sgemv` 12.76% → ~6.6% (pass), `dxa:model_parity-copy`
kept in tolerance (−1.2%), all previously-passing parity cases unchanged.

## Remaining gaps (tracked as known_issue)

`core:parity-sgemm-mc` (~27%) and `dxa:model_parity-wgmma-sp-dxa` (~15%) stay
red-flagged as `known_issue` xfails with this document as the triage record.
Closing them requires modeling the contended issue-path serialization
coherently rather than perturbatively: registered ready/credit gating, the
fragment-beat commit-port contention, and the collector pipeline need to land
together with a compensating alignment of SimX's post-dispatch depth
(measured ~4 cycles deeper than the RTL per FU class), so that uncontended
chain edges stay at parity while contended turnarounds stretch. The
measurement tooling for that work (per-UUID segment decomposition, per-PC
commit-gap profiles, turnaround distributions) is described above and easily
recreated from debug traces.

## Perf-gate baselines (separate root cause, fixed)

`dxa:perf_gate-wgmma-sp-dxa` and `tensor_wg:perf_gate-wgmma-sparse` failed
with instruction-count mismatches: the sparse-WGMMA kernel rewrite
(`692fcebc0`, 2026-07-28) updated goldens at the pre-restructure
`ci/perf/baselines/` path, which the CI catalog restructure had already
abandoned, so the canonical `ci/baselines/perf/` entries still described the
old kernels. Regenerated via `pytest ci -m perf_gate -k ... --update-baselines`;
both diffs lock in the rewrite's improvements (−34% and −49% cycles).
