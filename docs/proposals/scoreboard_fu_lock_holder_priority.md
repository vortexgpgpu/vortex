# Scoreboard FU-Lock: Holder-Priority Arbitration

**Status:** proposed (investigation complete, implementation pending)
**Scope:** `hw/rtl/core/VX_scoreboard.sv`, `sim/simx/core.cpp`
**Prerequisite reading:** the two existing lock designs compared below.

## 1. Problem statement

The issue-stage FU lock serializes multi-uop TCU sequences (WMMA/WGMMA): once a
sequence's first uop issues, no other warp may start a sequence on that FU until
the unlock uop retires. Two implementations of this lock exist in the code
lineage, and measurement shows each wins on a different workload class. This
proposal records the evidence, explains the mechanism, and specifies a third
design — holder-priority arbitration — expected to dominate both.

## 2. The two existing designs

### 2.1 Holder-exclusive per-warp mask (current on this branch)

A one-hot per-warp register (`fu_locked`, `'1` = open) gates the issue-arbiter
request inputs:

```
arb_valid_in[w] = staging_valid[w] && operands_ready[w] && fu_locked[w]
```

While a sequence holds the lock, **only the holder warp may request the
arbiter at all** — for any FU. The entire issue slice is dedicated to the
sequence, which therefore streams into the TCU back-to-back.

### 2.2 Per-FU readiness lock (upstream master)

A per-FU bit vector (`fu_locked[NUM_EX_UNITS]`, no owner) is folded into the
registered readiness term:

```
operands_ready_n = data_ready && ~fu_goingfull[ex_sel]
                && ~(fu_locked_n[ex_sel] && fu_lock_sel)
```

Only *sequence-starting* uops targeting a locked FU are blocked. Other warps'
ordinary uops (loads, ALU, and mid-sequence uops) continue to compete for
issue slots while a sequence is in flight.

## 3. Measured evidence

All numbers are rtlsim cycles on identical RTL except the scoreboard (and its
matching SimX issue model), identical apps, identical retired-instruction
counts. Configs are the `perf_gate` cell definitions in `ci/testcases/`.

| cell | per-FU (upstream) | holder-exclusive | delta |
|---|---|---|---|
| wgmma-fp16-ss, 128³ dense SS | 801,428 | **758,510** | holder-excl −5.4% |
| wgmma-fedp2k-rs, 64³ dense RS | 166,490 | **159,683** | holder-excl −4.1% |
| wgmma-sparse, 128³ 2:4 | **784,147** | 812,012 | per-FU −3.4% |
| fp16 / fp16-mc-nt4 / fp16-mc-nt16 / sgemm2 (WMMA) | — | — | identical |
| mxfp8 / sparse-fp16 (WMMA) | — | — | identical |

Net across the WGMMA suite the holder-exclusive design is ~22k cycles better,
which is why it is the one currently adopted. WMMA cells are insensitive:
their sequences are short enough that neither policy's difference is visible.

## 4. Mechanism

- **Dense sequences have no bubbles.** The holder warp has a ready uop every
  cycle, so any slot granted to another warp *displaces* a TCU uop, and the
  TCU is the bottleneck. Interleaving is pure loss: +42.9k cycles on the
  dense 128³ cell, +6.8k on FEDP2K-RS.
- **Sparse sequences have intrinsic bubbles** (metadata-dependent gaps where
  the holder is not ready). The per-FU design backfills those dead slots with
  other warps' work: −27.9k cycles on the sparse cell.
- **Negative result — check placement is not the lever.** A surgical hybrid
  was implemented and measured: upstream's per-FU lock state, but tested
  combinationally at the arbiter request (prism's gate position, registered
  state) instead of inside the registered readiness term. It recovered
  **zero** cycles (dense stayed at 801,428). The registered-readiness lag on
  lock handoff is not the cost; the absence of holder exclusivity is. Do not
  re-attempt placement-only variants.

## 5. Proposed design: holder-priority

Keep the holder-exclusive design's one-hot holder register, but open the
arbiter to other warps **only in cycles where the holder has no ready
request**, still excluding new sequence-starts:

```
// holder_req: the lock holder has a request this cycle (flop-sourced).
wire holder_req = |(lock_holder_onehot & raw_arb_valid);
wire lock_active = ~&fu_locked;   // some sequence in flight

arb_valid_in[w] = staging_valid[w] && operands_ready[w]
               && ( fu_locked[w]                       // holder always may
                 || (lock_active && ~holder_req && ~stg_fu_lock[w])
                 || ~lock_active );
```

Semantics:

- Holder ready → holder is the only requester (dense keeps its streaming,
  −5.4% / −4.1% wins preserved).
- Holder stalled → other warps may issue ordinary uops to any FU (sparse
  bubbles get backfilled, targeting the per-FU design's −3.4%).
- `stg_fu_lock[w]` exclusion keeps new sequence-starts blocked while a lock
  is active, preserving the sequence-atomicity contract; ownerless per-FU
  state is not needed because only one lock can be active per issue slice.

Success means winning **every** row of the table in §3 simultaneously.

### SimX lockstep

The SimX issue model must move in the same commit (`model_parity` enforces
this). The current model mirrors holder-exclusive via the suppress-fold
credit gating in `sim/simx/core.cpp`; holder-priority adds the same
fallback: when the locked warp has no ready uop this cycle, unsuppress the
other warps except sequence-starts.

## 6. Timing risk — the reason this is not a trivial change

The issue-stage has a documented history of timing sensitivity at NT16:

- The binding path routes **through** the GTO issue arbiter
  (staging → arbiter → wide arb_data mux → out_buf), route-dominated.
- `operands_ready[w]` must remain a **clean flop**; any combinational logic
  added on it lands directly on that path.

Holder-priority adds a cross-warp term to `arb_valid_in`: `holder_req` fans
in from the holder's request (flop-sourced: staging valid ∧ registered
readiness ∧ registered lock) and fans out to all `PER_ISSUE_WARPS` request
inputs. That is one added AND-OR level at the arbiter inputs plus a
PER_ISSUE_WARPS-wide broadcast. At NT16/ISSUE_WIDTH≥4 this may be the same
kind of fanout that previously cost WNS at this stage.

**Mitigations to evaluate, in order:**
1. Source `holder_req` entirely from registers (as sketched) — no
   readiness-cone logic in the term.
2. If routing-bound: register `holder_req` (one-cycle-late fallback opening).
   Dense is unaffected (holder keeps priority); sparse backfill arrives a
   cycle late per bubble — measure whether the win survives.
3. If still failing: gate the feature by `ISSUE_WIDTH`/NT via a localparam so
   small configs get holder-priority and the NT16 gate config keeps the
   current design.

**Timing acceptance:** the `core` fpga_gate must hold its current golden
(recorded with the holder-exclusive design) within tolerance. A tcu-gate run
is unaffected (the scoreboard is outside the tcu DUT).

## 7. Validation plan / acceptance criteria

1. `run-tcu` + `run-tcu-dsp` unittest suites green (no datapath impact
   expected; cheap sanity).
2. Perf cells, all must hold or improve vs current goldens:
   - `wgmma-fp16-ss` = 758,510 (must not regress)
   - `wgmma-fedp2k-rs` = 159,683 (must not regress)
   - `wgmma-sparse` ≤ 812,012, target ≈ 784,147 (the point of the change)
   - all WMMA cells unchanged
3. `model_parity` cases green with the matching SimX change.
4. `core` fpga_gate PASS against the current golden.
5. Re-record only the cells that improve; a red cell means the design failed,
   not that the baseline moved.

## 8. Upstream considerations

Upstream master carries the per-FU design (its dense WGMMA is 5.4% slower
than holder-exclusive on identical RTL). This branch's scoreboard is
therefore a deliberate divergence, and every future merge will conflict on
this file — automated resolution must not silently re-take upstream's
version (the `wgmma-fp16-ss` perf_gate ratchet at 758,510 is the enforcement
backstop). If holder-priority validates, propose it upstream: it strictly
dominates both existing designs and would dissolve the divergence.
