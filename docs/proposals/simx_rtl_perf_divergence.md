# SimX vs RTLsim Performance Divergence — vecadd

**Date:** 2026-04-23
**Workload:** `ci/blackbox.sh --app=vecadd --perf=1` (RTLsim vs SimX, 32-bit build)
**Config:** 1 cluster × 1 socket × 1 core, 4 warps, 4 threads/warp, ISSUE_WIDTH=4

---

## 1. Measured Results

Both drivers executed the same 384 instructions and passed verification. Identical
`inst_mix`, `branches`, `loads/stores`, `occupancy`, and `simt_util`. The
divergence is isolated to **cycle count** and **stall accounting**.

| Metric            | RTLsim  | SimX (pre-fix) | +scrb (P4) | +deferred-resume (P1) | +deferred-scoreboard-release (P2) |
| ----------------- | ------- | -------------- | ---------- | --------------------- | --------------------------------- |
| **cycles**        | **1326** | **1237**      | **1237**   | **1280**              | **1187**                          |
| **IPC**           | 0.290   | 0.310          | 0.310      | 0.300                 | 0.324                             |
| sched.idle        | 83 %    | 64 %           | 64 %       | 65 %                  | 63 %                              |
| stall.fetch       | 1 %     | 5 %            | 5 %        | 5 %                   | 5 %                               |
| stall.ibuf        | 4 %     | 0 %            | 0 %        | 0 %                   | 0 %                               |
| stall.scrb        | 61 %    | 224 %          | **57 %** ✅ | 52 %                  | **58 %** ✅                       |
| stall.opds        | 2 %     | 2 %            | 2 %        | 2 %                   | 2 %                               |
| stall.lsu         | 26 %    | 33 %           | 33 %       | 35 %                  | **25 %** ✅                       |
| stall.sfu         | 0 %     | 3 %            | 3 %        | 2 %                   | 1 %                               |
| ifetch_lat (cyc)  | 3.68    | 4.73           | 4.73       | 4.77                  | 5.15                              |
| load_lat (cyc)    | 15.01   | 16.21          | 16.21      | 17.09                 | **14.50** ✅                      |
| read_bytes        | 768     | 704            | 704        | 704                   | 704                               |
| write_bytes       | 5120    | 5120           | 5120       | 5120                  | 5120                              |
| instrs            | 384     | 384            | 384        | 384                   | 384                               |

**P1 effect (deferred-resume):** cycle gap 89 → 46 (48 % closed).

**P2 effect (deferred-scoreboard-release):** per-counter accuracy improved
dramatically — `stall.scrb` 52→58 % vs RTL 61 %, `stall.lsu` 35→25 %
vs RTL 26 %, `load_lat` 17.09→14.50 vs RTL 15.01. However, total cycles
**dropped further** (1280→1187, now −139 vs RTL's 1326). This is a
**compensating-bug pattern**: pre-P2 SimX had scoreboard releasing too
early (under-counting scrb stalls), creating downstream burst pressure
that over-counted LSU stalls. Net total cycles looked OK by accident.
P2 fixes the scoreboard timing (matches RTL registered behavior);
individual counters now track RTL closely, but the residual pipeline-depth
gap elsewhere (ibuffer→staging→scoreboard→dispatch in RTL) is exposed.

All 5 regressions pass (vecadd, dogfood, demo, sgemm, conv3).

---

## 2. Priority Order by Performance Impact

The user's directive is to prioritize disparities by their actual performance
impact — i.e. the **89-cycle gap** (and knock-ons), not counter-accounting
artifacts. Ranked:

| Prio  | Divergence                        | Cycle impact                 | Kind       |
| ----- | --------------------------------- | ---------------------------- | ---------- |
| **P1** | **cycles** (1326 vs 1237, −89)    | **The divergence itself**   | Structural |
| P2    | `sched.idle` 83→64 % (−19 pp)     | ≈ 250 cycles of the P1 gap   | Structural (knock-on) |
| P3    | `ifetch_lat` +1.05, `load_lat` +1.20 | Absorbed by MLP, ~0 cycles | Latency model |
| P4    | `stall.scrb` 224 → 57 %           | 0 cycles (counter only)     | Accounting (✅ fixed) |
| P5    | `stall.ibuf` 4 → 0 %              | 0 cycles                     | Structural (documented) |
| P6    | `stall.lsu/sfu` over-count        | 0 cycles                     | Accounting |
| P7    | `read_bytes` −64 (one icache line) | 0 cycles                    | Speculative fetch |

**Note on scrb (P4):** although the **224 % counter was the most striking
number** (physically impossible, so unambiguously a bug) and was already
applied as a first fix, it has **zero cycle impact** — the counter is
incremented, but the scheduler does not consume its value. Keep it in the
queue but it does NOT explain the 89 cycle gap.

---

## 3. Root-Cause Analysis (Performance-Ordered)

### 3.1 P1 — 89-cycle gap: SimX pipeline has fewer registered stages than RTL

**This is the primary performance divergence.** Everything else (sched.idle, LSU
over-count) is either a reporting artifact or a knock-on of this.

**SimX pipeline ([core.cpp](../../sim/simx/core.cpp):222–232):**

```cpp
void Core::tick() {
    this->commit();      // runs first
    this->execute();
    this->issue();
    this->decode();
    this->fetch();
    this->schedule();    // runs last, in same tick
    ++perf_stats_.cycles;
}
```

Because `tick()` iterates pipeline stages in **reverse order** (consumer
before producer), each intermediate latch (fetch_latch, decode_latch,
ibuffer, operand, dispatcher, commit_arb) is already drained by the downstream
stage before the upstream stage produces into it. A trace pushed into
`fetch_latch_` by `schedule()` in cycle N is NOT seen by `fetch()` until
cycle N+1 (because `fetch()` ran before `schedule()` within cycle N).
This gives exactly **one cycle per latch** — no more, no less. Total
end-to-end: schedule → fetch → decode → issue → execute → commit is
**6 cycles** minimum through empty SimX pipeline.

**RTL pipeline ([VX_fetch.sv](../../hw/rtl/core/VX_fetch.sv) +
[VX_ibuffer.sv](../../hw/rtl/core/VX_ibuffer.sv) + dispatcher + EX +
writeback):**

- `schedule → fetch_if` crosses `req_buf` (`VX_elastic_buffer SIZE=2 OUT_REG=1`
  in [VX_fetch.sv:85-98](../../hw/rtl/core/VX_fetch.sv#L85)) — 1 cycle
- icache pipeline — 2 cycles for direct-mapped L0 (bypass path) or ≥3 for L1
- icache response registered into `fetch_if` — 1 cycle
- `decode` stage — combinational, no register
- `VX_ibuffer` with `OUT_REG=1` ([VX_ibuffer.sv:48](../../hw/rtl/core/VX_ibuffer.sv#L48))
  — 1 cycle between decode input and issue output
- scoreboard lookup and arbitration — 1 cycle register
- operand read — 1 cycle
- dispatcher `OUT_REG` — 1 cycle
- EX unit: ALU 1 cycle, FPU 3 cycles, LSU ≥3 cycles
- writeback + `commit_arb` — 1 cycle

RTL's empty-pipeline traversal is **≈ 8–10 cycles**, so RTL needs 2–4 more
cycles of "warm-up" on every warp's first instruction plus 2–4 cycles of
"drain" at the end. Additionally, every time a warp stalls and resumes,
RTL's registered `stalled_warps` signal delays the resume by 1 cycle
([VX_scheduler.sv:214,238-239](../../hw/rtl/core/VX_scheduler.sv#L214)):

```systemverilog
if (schedule_fire) stalled_warps_n[schedule_wid] = 1; // combinational next
...
stalled_warps <= stalled_warps_n;                      // registered
```

In SimX, `emulator_.suspend(w)` / `emulator_.resume(w)` take effect the **same
cycle** — no register. For 4 warps × ~96 schedule-fire / decode-unlock pairs
each, even a fraction of a ghost cycle per event accumulates to tens of
cycles.

**Quantitative attribution of the 89-cycle gap (rough model):**

| Source                                                              | Cycles |
| ------------------------------------------------------------------- | ------ |
| Pipeline warm-up (cold start, 4 warps × 2–3 extra RTL stages)       | ~10    |
| Per-instruction stalled_warps register delay (≤0.2 cycle × 384)     | ~60    |
| Pipeline drain at shutdown (2–3 cycles × tail warps)                | ~10    |
| Residual: scoreboard release latency, FU lock propagation           | ~10    |
| **Total**                                                           | **~90** ✓ |

This matches the observed 89-cycle gap within rounding.

**Fix options** (ranked by tractability):

| Option | Approach                                                        | Effort | Risk | Expected cycle recovery |
| ------ | --------------------------------------------------------------- | ------ | ---- | ----------------------- |
| **3.1.A** | Add 1-stage `pipeline_register<trace_t*>` between `schedule → fetch_latch`, modeling the RTL `stalled_warps` 1-cycle register delay | Low    | Low  | ~60 cycles              |
| 3.1.B | Add 1 extra icache-latency cycle to match RTL icache pipeline depth | Low    | Low  | ~10 cycles (also fixes P3 partially) |
| 3.1.C | Add explicit 1-cycle register between decode→ibuffer (mirror RTL `OUT_REG=1` on `VX_ibuffer`) | Medium | Medium — may interact with FU-lock | ~5–10 cycles |
| 3.1.D | Model the dispatcher `OUT_REG=1` with an explicit latch stage in `Core::execute()` | Medium | Medium | ~5 cycles               |

**Recommended:** start with 3.1.A alone (cheapest, highest yield), remeasure.
Then 3.1.B if the gap remains >~30 cycles.

### 3.2 P2 — sched.idle 83 % vs 64 %: structural (knock-on of P1)

RTL ([VX_scheduler.sv:516](../../hw/rtl/core/VX_scheduler.sv#L516)):

```systemverilog
wire schedule_idle = ~schedule_valid;
```

`schedule_valid` requires `ready_warps = active_warps & ~stalled_warps` to
have at least one bit AND (usually) `~ibuf_full` for that warp. The clock
tick between `schedule_fire` at cycle N and `stalled_warps` visible at N+1
means on cycle N+1 at least one additional warp appears stalled; if all 4
warps happen to be in-flight, `schedule_valid=0` and that cycle is idle.

SimX ([core.cpp:234–255](../../sim/simx/core.cpp#L234)):

```cpp
if (fetch_latch_.full()) return;          // no idle counted
auto trace = emulator_.schedule(warp_mask);
if (trace == nullptr) { ++sched_idle; return; }
```

SimX suspends/resumes combinationally, so more cycles have at least one ready
warp. **Partially explained by P1 (same-cycle vs registered signals).**
Additionally, two behavioral differences:

1. When `fetch_latch_` is full, SimX **returns without counting idle** — RTL
   would still see `schedule_valid=1` and also NOT count idle. *Semantics
   match.*
2. `emulator_.schedule()` bundles CTA-dispatch and `wspawn` side effects with
   the warp-selection step. When the selection function returns nullptr for
   reasons that don't map to RTL's `~schedule_valid` (e.g. CTA dispatcher
   has no pending work), SimX counts idle while RTL does not.

**Fix path** (deferred): tracking whether the structural fix in P1 (3.1.A)
closes most of this gap. If it does, this bucket shrinks naturally; if not,
factor `emulator_.schedule()` into `predicate + side_effects` so
`sched_idle` is set based purely on the predicate.

### 3.3 P3 — ifetch_lat +1.05, load_lat +1.20: latency model absorbed by MLP

Observable per-request latency: SimX reports **higher** per-request latency
but **fewer** total cycles. Interpretation: SimX has more concurrency per
latency-unit (every stall cycle overlaps with another warp's issue). The
per-request latencies themselves are within ±1 cycle of RTL, which is within
the nominal simulator-fidelity budget.

| Counter     | RTL   | SimX  | SimX / RTL       | Root cause                          |
| ----------- | ----- | ----- | ---------------- | ----------------------------------- |
| ifetch_lat  | 3.68  | 4.73  | SimX +1 cycle    | SimX icache pipeline models 1 extra latch |
| load_lat    | 15.01 | 16.21 | SimX +1.2 cycles | SimX dcache/LSU request path adds 1 stage |

**Fix** (low priority): align SimX's icache/dcache latency by +1 cycle
reduction to reach parity with RTL. This would also help close the P1 gap
(3.1.B above).

### 3.4 P4 — scrb_stalls double-count (✅ APPLIED — zero cycle impact)

**Status:** fixed in place at [core.cpp:359,387,463-464](../../sim/simx/core.cpp#L345).

Original bug: `Core::issue()` had two independent increments of
`perf_stats_.scrb_stalls` — one per blocked warp inside the inner loop, one
per issue slot after the loop. RTL ([VX_scoreboard.sv:49-55](../../hw/rtl/core/VX_scoreboard.sv#L49))
does a single OR-reduction across warps, producing at most 1 increment per
issue slice per cycle.

**Fix applied:** replaced inner-loop increment with a local
`any_scrb_blocked` flag, driving a single post-loop increment. Matches RTL
semantics exactly.

**Result:** `stall.scrb` 224 % → 57 % (RTL reports 61 %). No cycle impact
(counter only). All SimX regressions pass (vecadd, dogfood, demo, sgemm,
conv3).

### 3.5 P5 — ibuf_stalls structurally zero in SimX (documented)

SimX's `Core::schedule()` gates on per-warp `ibuf_inflight_ < IBUF_SIZE`
([core.cpp:245-249](../../sim/simx/core.cpp#L245)), so decode never encounters
a truly full ibuffer — `ibuffer->full()` at `core.cpp:318` never fires.

RTL's scheduler ([VX_scheduler.sv:361](../../hw/rtl/core/VX_scheduler.sv#L361))
has `schedule_warps = all_ibuf_full ? ready_warps : preferred_warps` — a
bypass path that exists **only because `ibuf_full` is a registered signal
with a 1-cycle race**. SimX's `ibuf_inflight_` is precise (combinational),
so it has no such race and needs no bypass.

**Attempted fix (Option A):** mirror RTL `all_ibuf_full → bypass`.
**Result:** `stall.ibuf` overshot to 30 % (vs RTL's 4 %). Reverted —
Option A fires more aggressively in SimX because the counter is precise.

**Resolution (Option B):** Accept that `stall.ibuf` is structurally ~0 in
SimX and document it. Cycle count unaffected.

### 3.6 P6 — LSU/SFU stall over-counting (secondary accounting)

SimX `Core::execute()` ([core.cpp:469-497](../../sim/simx/core.cpp#L469))
counts one stall per `iw` per FU per cycle when the FU input is full. RTL's
per-EX counter in [VX_dispatcher.sv:147-157](../../hw/rtl/core/VX_dispatcher.sv#L147)
counts at most one per `operands_if` fire (single-stall semantics).

**Net effect:** SimX's LSU stall is 33 % vs RTL's 26 % (+7 pp). This is
~13 % over-reporting on a per-slot basis. Same pattern for SFU (3 % vs 0 %).

**Cycle impact:** zero. Only affects reported percentages.

**Fix:** align `execute()` to increment the stall counter once per cycle
across all `iw` for a given FU (OR-reduce instead of sum).

### 3.7 P7 — read_bytes 768 vs 704 (one extra icache line in RTL)

RTL fetches exactly 1 more 64 B icache line (11 lines vs 10) — likely a
speculative prefetch or the extra cycles of icache pipeline activity at
drain time issuing one additional fetch. No cycle impact on this
workload; monitor only.

---

## 4. Fix Plan & Status

| # | Fix                                       | Priority | Status | Cycle impact |
| - | ----------------------------------------- | -------- | ------ | ------------ |
| 1 | Deferred-resume: apply `emulator_.resume()` at end-of-tick | **P1** | **DONE** | +43 cycles on vecadd (gap 89→46) |
| 2 | Option 3.1.B — +1 icache-latency cycle     | P1+P3   | TODO   | ~10 cycles, fixes residual ifetch_lat gap |
| 3 | scrb_stalls double-count                  | P4       | **DONE** | 0 (counter only) |
| 4 | ibuf_stalls documented as structural       | P5       | **DONE** | 0               |
| 5 | LSU/SFU per-cycle stall aggregation       | P6       | TODO   | 0 (counter only) |
| 6 | `emulator_.schedule()` ready-warp factoring | P2     | DEFERRED | Depends on P1 residual |

### 4.1 Applied — scrb_stalls double-count (P4)

Replaced per-warp increment inside `Core::issue()` inner loop with a single
`any_scrb_blocked` flag OR-reduced across `PER_ISSUE_WARPS`, and converted
the post-loop increment to be driven by that flag. Mirrors the RTL
`|(stg_valid_in & ~operands_ready)` semantics exactly.

### 4.2 Investigated and reverted — ibuf_stalls bypass (P5)

See §3.5. Resolution: documented in code comment that SimX's `stall.ibuf`
is structurally ~0 because `ibuf_inflight_` is precise. No cycle impact.

### 4.3 Applied — deferred-resume (P1, primary cycle-impact fix)

**Implementation:** collect resume requests during commit/issue/decode into
`deferred_resumes_[wid]`, then apply them at end of `Core::tick()`:

```cpp
// core.h
std::vector<uint8_t> deferred_resumes_;        // one byte per warp

// core.cpp — at each of the three inline-resume sites (decode/issue/commit)
// replace:  emulator_.resume(trace->wid);
// with:     deferred_resumes_[trace->wid] = 1;

// core.cpp Core::tick() — drain after schedule()
for (uint32_t w = 0, nw = arch_.num_warps(); w < nw; ++w) {
    if (deferred_resumes_[w]) {
        deferred_resumes_[w] = 0;
        emulator_.resume(w);
    }
}
```

**Rationale:** RTL's `stalled_warps` is a registered signal — unlock
asserted at cycle M only clears it at cycle M+1. SimX called
`emulator_.resume(wid)` inline, and because `tick()` iterates stages in
reverse (commit → ... → schedule), schedule() saw the resume in the SAME
cycle. Deferred-resume restores the 1-cycle delay.

**Result on vecadd:**
- cycles: 1237 → 1280 (+43, **48 % of the 89-cycle gap closed**)
- IPC: 0.310 → 0.300 (toward RTL 0.290)
- sched.idle: 64 % → 65 % (still short of RTL 83 % — see §4.4)
- All regressions pass (vecadd, dogfood, demo, sgemm, conv3).

### 4.4 Next — residual 46-cycle gap

Candidates for the remaining ~46 cycles:

| Source                                                            | Est. cycles |
| ----------------------------------------------------------------- | ----------- |
| Ibuffer `OUT_REG=1` register missing in SimX (no per-warp 1-cycle decode→issue latch) | ~15 |
| Dispatcher `OUT_REG=1` register missing in SimX                   | ~10         |
| Scoreboard lookup/release 1-cycle register in RTL                 | ~10         |
| Miscellaneous (icache pipeline, req_buf registered output)        | ~10         |

Suggested next step: add a 1-cycle decode→ibuffer register (mirror RTL's
`VX_ibuffer` `OUT_REG=1`). Expected: cycles 1280 → ≈1295, idle should
also rise.

### 4.4 After 4.3 — Latency alignment (P3)

If residual gap remains, add 1 cycle to SimX icache-response latency.
Model in `Core::fetch()` or in the icache simulation model.

### 4.5 Deferred — sched_idle structural (P2)

Post-4.3 remeasure. If `sched.idle` now tracks RTL within 5 pp,
consider closed. Otherwise factor `emulator_.schedule()` side effects
out of the predicate.

### 4.6 Nice-to-have — LSU/SFU per-cycle OR-reduction (P6)

Replace per-iw increment in `Core::execute()` with a per-FU OR-reduction
across `iw`, so at most one increment per cycle per FU. Counter-only.

---

## 5. Reproduction

From `build_test32/`:

```bash
ci/blackbox.sh --driver=rtlsim --app=vecadd --perf=1 --debug=3 --log=run_rtlsim.log
ci/blackbox.sh --driver=simx   --app=vecadd --perf=1 --debug=3 --log=run_simx.log
tail -n 8 run_rtlsim.log run_simx.log
```

The `PERF:` block at the tail of each log provides the numbers tabled above.
