# SimX Contended Issue-Path Modeling

## Problem

Two `model_parity` cases were `known_issue`-marked after the LSU/divide
timing alignment (see `simx_lsu_timing_alignment_proposal.md`):

| case | gap at triage | tolerance |
|---|---|---|
| `core:parity-sgemm-mc` (n128, NT8/NW8, 2 cores, L2) | 27.2% | 10% |
| `dxa:model_parity-wgmma-sp-dxa` | 17.5% | 10% |

Retired instructions matched exactly on both; the divergence was purely the
cycle model under full-occupancy contention: the hardware's producer-commit
→ dependent-dispatch turnaround measured 55/92/146 cycles (p50) against
SimX's 8/11/25.

## Root causes (established by cycle audit + trace forensics)

At the parity shape (ISSUE_WIDTH=1: one issue slice, one commit beat per
cycle core-wide), the hardware serializers SimX lacked, each verified by
per-UUID trace measurement on both drivers:

1. **Per-fragment load commit.** A coalesced load returns N response beats;
   each takes one commit-port grant and only the last clears the
   scoreboard (measured beat demand: 16,764 hardware vs 17,519 modeled on
   the tensor fp16 workload, per-FU class counts identical). SimX retired
   every load in one beat.

2. **Fixed-priority commit arbitration** (ALU > LSU > SFU > FPU > TCU): a
   hot ALU stream starves load writebacks and every dependent warp's
   scoreboard clear. SimX arbitrated round-robin.

3. **Commit-path depth.** SimX took 3 channel hops from FU output to
   retirement where the hardware commits in one registered stage —
   a uniform ~+4-cycle per-class overhang (ALU dispatch→commit p50 6 vs 2),
   historically cancelled by an optimistic issue side. The two must move
   together; the issue side's same-cycle credit return and its
   transient-stall bypass are retained as the deliberate counterweight to
   the remaining constant overhang.

4. **Execution-unit output windows.** The FPU admits at most
   `VX_CFG_FPU_QUEUE_SIZE` operations between operand acceptance and
   result handoff (tag queue), however deep its pipelines are — measured
   directly: hardware FMA streams sustain exactly 2 in flight, dispatching
   a new op at the instant an old one commits, and the fp32 tensor
   workload never exceeds depth 2. Results leave the sub-unit pipelines
   out of order (a converter finishing under a longer multiply-add), so
   slots free by exit time. The TCU bounds results awaiting the consumer
   by its landing-queue depth `2^clog2(TCU_LATENCY+1)`. SimX's uniform
   2-entry output channels approximated these bounds by accident and
   collapsed when any channel was widened, which is how a latent ~30%
   divergence on `parity-sgemmx` stayed hidden.

## Landed model (SimX only; no RTL, no config change)

- `lsu_unit.cpp`: every memory response fragment retires through its own
  writeback beat (non-final fragments as writeback-less beats), a full
  writeback path backpressures the response stream, and the commit-side
  channel holds 6 beats so a fragment burst queues at the port instead of
  throttling memory.
- `core.cpp`: per-FU commit staging with fixed-priority selection inside
  commit(); a granted beat retires one registered stage after its FU
  presents it.
- `fpu_unit.{h,cpp}`: explicit tag-queue admission bound
  (`VX_CFG_FPU_QUEUE_SIZE`) with out-of-order slot release at each
  result's pipeline-exit time; output capacity = bound + one skid entry.
- `tcu/tcu_unit.cpp`: output capacity = the landing-queue depth.
- `sfu_unit.cpp`: 6 beats of commit-side elasticity (texture/raster
  request queues run deeper than the default channel).
- `func_unit.h`, `sim/common/simobject.h`: channel-capacity
  parameterization for the above.

## Results

`parity-sgemm-mc` 27.2% → **+0.3–4%** (tolerance tightened to 0.05);
`parity-sgemmx`'s latent divergence surfaced at ~30% and closed to under
10%; the tensor, graphics, raytracing, vm, and core suites hold within
tolerance (full 29-case sweep gates the change).

The wgmma/DXA family retains tracked gaps (`wgmma-sp-dxa` ~15%,
`wgmma-dxa` ~5–7%, `wgmma-dxa-mcast` ~12%): their residuals are
insensitive to every issue/commit/window mechanism above, pointing at the
DXA producer path's own timing model; they are `known_issue`-marked with
this document as the triage record.

## Method (reusable)

Per-UUID trace matching from `--debug=3` logs on both drivers (identical
UUIDs): per-class dispatch→commit latency tables, per-PC weighted latency
deltas, per-warp inter-dispatch gap histograms with stalled-at-PC
attribution, load segment decomposition (dispatch→request→responses→
commit), commit-beat counting per FU class, and event-sweep in-flight
depth profiles. Sentinel cases are never sufficient: SimX timing is
chaotically coupled, so every change is gated on the full parity sweep.
