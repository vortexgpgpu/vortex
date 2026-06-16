# Schedulable Convergence Stack (SCS) — Deadlock-Free SIMT Without ITS

**Scope:** [hw/rtl/core/VX_ipdom_stack.sv](../../hw/rtl/core/VX_ipdom_stack.sv), [VX_split_join.sv](../../hw/rtl/core/VX_split_join.sv), [VX_schedule.sv](../../hw/rtl/core/VX_schedule.sv), [hw/rtl/VX_gpu_pkg.sv](../../hw/rtl/VX_gpu_pkg.sv), [sim/simx/wctl_unit.cpp](../../sim/simx/wctl_unit.cpp), [sim/simx/scheduler.h](../../sim/simx/scheduler.h), [sim/simx/decode.cpp](../../sim/simx/decode.cpp), LLVM-Vortex divergence pass (`~/dev/llvm_vortex`)
**Reference:** ElTantawy & Aamodt, *MIMD Synchronization on SIMT Architectures*, MICRO 2016 (multi-path execution); Diamos et al., *Execution of Divergent Threads Using a Convergence Barrier*, US 2016/0019066A1 (ITS, the rejected baseline)
**Branch:** `feature_scs` (off the `diverge` baseline)
**Status:** Proposal — for review
**Tests:** [tests/opencl/lockht](../../tests/opencl/lockht), [tests/opencl/lclist](../../tests/opencl/lclist)

---

## 1. Motivation

Vortex resolves control-flow divergence with a hardware IPDOM
reconvergence stack ([VX_split_join.sv](../../hw/rtl/core/VX_split_join.sv),
[VX_ipdom_stack.sv](../../hw/rtl/core/VX_ipdom_stack.sv)). The compiler
emits `vx_split`/`vx_join`; the hardware pushes a `{mask, reconv_PC}`
entry on divergence and pops it at the immediate post-dominator. This is
area-efficient, deterministic, scales well, and is optimal for
throughput on structured code.

Its one defect is **forward progress**: any algorithm where one thread
of a warp must make progress for a sibling thread to proceed
*deadlocks*, because the stack runs one divergent path to its
reconvergence point before scheduling the sibling. Two regression tests
in this tree exercise the failure deterministically:

- [tests/opencl/lockht](../../tests/opencl/lockht) — per-bucket spin-lock
  hash table; deadlocks when two warp lanes contend the same bucket lock.
- [tests/opencl/lclist](../../tests/opencl/lclist) — hand-over-hand
  (lock-coupling) sorted linked list; deadlocks at the first `lock(head)`.

The mainstream fix is NVIDIA's Independent Thread Scheduling (ITS):
per-thread program counters plus a per-warp convergence-barrier table.
**We reject ITS for Vortex on two grounds:**

1. **Area.** Per-thread PCs + per-thread state (Ready/Yield/Blocked/Exited)
   + a ~16-entry named-barrier table *per warp* is a large, always-resident
   state increase that pressures Fmax and occupancy on the U55C
   (300 MHz target).
2. **Compiler complexity.** ITS requires a convergence-optimizer pass that
   *allocates* named barrier resources and reasons about their lifetimes
   across nested/irreducible control flow — a non-trivial, bug-prone
   register-allocation-class problem.

This proposal keeps the IPDOM stack and its compiler interface almost
unchanged, and obtains **unconditional deadlock-freedom** with a small
hardware addition plus a *simple* compiler addition (yield insertion +
loop structuring — no resource allocation).

---

## 2. Key insight

**The IPDOM stack already stores a PC per divergent subgroup.** Each
entry is a `{mask, PC}` for a path that is not currently executing. The
deadlock is therefore *not* a reconvergence-detection failure — the
stack knows exactly where every path is and where it will merge. It is a
**scheduling-order** failure: the stack is strict-LIFO, so only the top
entry runs, to completion, before any sibling.

So we do **not** need per-thread PCs to break the deadlock. We need to
let the warp *switch among the `{mask, PC}` entries it already has* when
the running entry stalls. That is the entire idea.

The granularity of independent progress becomes **per-split** (per
divergent subgroup), not per-thread. ITS pays for `warp_size`
independent contexts, always, in hardware. SCS provides a small number
`K` of independently *schedulable* splits, and (Section 5) spills beyond
`K` to memory so the guarantee is unconditional while the *hardware*
cost stays IPDOM-class.

---

## 3. Architecture overview

Three layers; the compiler does the heavy lifting (per the directive that
software is cheaper than hardware), the hardware stays minimal.

### 3.1 Hardware — Schedulable Convergence Stack
Replace the strict-LIFO stack with a small **schedulable split table** of
`K` entries per warp (`K` ≈ current `DV_STACK_SIZE`, e.g. 8). Each entry,
as today, holds `{active_mask, PC, reconv_PC}`; we add per entry:

- a 2-bit **status**: `RUNNABLE`, `YIELDED`, `WAIT_MERGE`;
- the table gains a **round-robin pointer** (one per warp).

Behavior:
- **Convergent / single-split code:** exactly one entry, no switching —
  bit-identical to today's IPDOM. No new per-cycle cost.
- **`vx_split`:** push a child entry (as today); the parent becomes
  `WAIT_MERGE` for that reconv_PC.
- **`vx_join` / reaching `reconv_PC`:** merge the split into its parent
  when *all* siblings sharing that `reconv_PC` have arrived; otherwise the
  arriving split becomes `WAIT_MERGE`. (This is today's two-phase join,
  decoupled from LIFO order.)
- **Switch trigger:** when the running split stalls (Section 3.2), set it
  `YIELDED` and advance the round-robin pointer to the next `RUNNABLE`
  split. A `YIELDED` split returns to `RUNNABLE` when any sibling commits
  an architectural change it might be waiting on (coarse: clear all
  `YIELDED` in the warp on any successful store/atomic-mutation by the
  warp; fine: see §3.2).

### 3.2 Hardware — stall watchdog (safety net)
A per-warp counter that detects "no forward progress":
- increments on a back-edge taken with **no thread exiting** the loop, or
  on an **atomic that returned contended / unchanged** (`amoswap`/`CAS`
  failure is the cleanest signal);
- resets on any thread exiting a loop or any successful memory mutation.

On crossing threshold `T` (configurable, ~64–256 issue slots), the
running split is forced `YIELDED` and the warp rotates. This guarantees
progress *even for code the compiler did not instrument* (indirect calls,
separately-compiled libraries) — best-effort, bounded by `K` (and made
unconditional by §5).

### 3.3 ISA — one new instruction, rest reused
- **`vx_yield`** (new SFU op): deschedule the current split, rotate to the
  next `RUNNABLE` split; no-op if none. Single cycle, allocates nothing.
- **`vx_split` / `vx_join`:** unchanged — the compiler emits them exactly
  as today.
- **No `vx_pred`-based opt-out loop is required** — hardware
  schedulability replaces the page-10 software emulation.

### 3.4 Compiler — yield insertion + loop structuring (no allocation)
The LLVM-Vortex divergence pass gains two cheap responsibilities:
1. **Yield insertion:** place `vx_yield` on the back-edge of every loop
   that may block on another lane — conservatively, any loop containing an
   atomic RMW whose result feeds the loop condition, or any loop the pass
   cannot prove lane-independent. (It already identifies loops for
   `vx_split`/`vx_join`.)
2. **Bounded-split structuring:** ensure a lane's *wait state* lives in
   **registers** (per-lane, free in SIMT) and that blocking acquisition is
   a *single* loop with a `vx_yield`, rather than nested spins that push a
   new split per attempt. This keeps the number of coexisting splits a
   **small compile-time constant** (Section 5), so the hardware table
   rarely spills.

Crucially, neither task allocates a resource or reasons about lifetimes —
it is yield placement plus a standard single-loop transform. This is the
"compiler is cheap" trade the reviewer asked for.

---

## 4. Worked examples

### 4.1 `lockht` (single-bucket contention)
1. Lanes contend `bucket_lock[b]`; `amoswap` yields one winner, rest fail.
2. Divergent spin loop → split: `{winner @ post-loop}` (WAIT_MERGE) and
   `{losers @ loop}` (RUNNABLE, executing).
3. Losers' CAS keeps failing → watchdog (or `vx_yield`) fires → losers
   `YIELDED`, rotate to the winner split.
4. Winner runs CS, releases the lock, reaches `vx_join` → WAIT_MERGE.
5. Lock now free; rotate back to losers → one wins, repeats.
6. When all have merged at the join → reconverge. **Two coexisting
   splits suffice (K=2).**

### 4.2 `lclist` (hand-over-hand)
- Per-lane state (`pred`, `curr`, held-lock id) is in registers; the
  traversal + acquisition is one structured loop with a `vx_yield` on the
  acquire back-edge.
- The lane holding `lock(head)` is a RUNNABLE split; the spinners are
  another. The watchdog/yield rotates so the holder advances (acquire
  next, release head), unblocking the spinners.
- Because acquisition is a *single* yielding loop (not nested spins), the
  coexisting-split count is bounded by static control-flow nesting, **not**
  by traversal depth or warp size.

---

## 5. The bounded-`K` limitation and its solution

**Limitation.** A `K`-entry hardware table guarantees forward progress
only while the number of splits that must coexist is ≤ `K`. Adversarial,
un-instrumented divergence (e.g. a maximally-divergent goto soup) can in
principle demand up to `warp_size` distinct contexts — the same worst
case that forces ITS to use per-thread PCs.

**Two-part solution.**

### 5.1 Compiler bounds the common case to a small constant
For analyzable code, coexisting splits = (static divergent-nesting depth)
+ (distinct blocking points), **independent of warp size**:
- Structured/reducible control flow has IPDOM stack depth bounded by
  source nesting — already why `DV_STACK_SIZE` is a small constant today.
- The §3.4 single-loop structuring keeps each blocking region to O(1)
  splits (active-in-loop vs. yielded), instead of O(depth) nested spins.
- Per-lane progress state is in registers, so "where each lane is" costs
  **no split** — it is ordinary SIMD register state.

Result: for the `lockht`/`lclist` class the compiler holds coexisting
splits at a handful, so the hardware table never spills.

### 5.2 Hardware makes it unconditional: memory-backed split spill
Treat the `K`-entry table as a **cache over a warp-private, memory-backed
split pool**. When a split must be created and the table is full:
- evict the least-recently-run **non-`WAIT_MERGE`** split to a
  warp-private spill region (a `{mask, PC, reconv_PC}` descriptor — tiny);
- the round-robin scheduler cycles over *all logical* splits; selecting a
  spilled split swaps it into a hardware slot (evicting another), exactly
  like register spill/fill.

Consequences:
- **Unconditional deadlock-freedom.** Every logical split — including a
  spilled lock *holder* — is eventually scheduled, so held resources are
  always released. This matches ITS's guarantee.
- **IPDOM-class hardware area.** The resident state is `K` entries (a
  small constant), not `warp_size` contexts. The unbounded part lives in
  cheap memory and is materialized *only* under pathological divergence.
- **Graceful degradation.** Spill costs bandwidth/latency, i.e. a
  slowdown, never a hang. The compiler (§5.1) keeps the common case
  entirely in-table, so spill is a rare safety valve.

This is the crux of why SCS dominates ITS on the reviewer's two axes:
ITS pays `warp_size`-context area in *hardware, always*; SCS pays
`K`-context area in hardware and pushes the rare overflow to *memory*.

---

## 6. RTL microarchitecture and area/timing

### 6.1 Baseline (what exists today)
The IPDOM stack ([VX_ipdom_stack.sv](../../hw/rtl/core/VX_ipdom_stack.sv))
is **BRAM-backed**: a single `VX_dp_ram` of `DV_STACK_SIZE × NUM_WARPS`
entries, each `1 + NUM_THREADS + PC_BITS` bits (fallthrough flag + union
mask + reconvergence PC), with `DV_STACK_SIZE = NUM_THREADS − 1` and
`PC_BITS ≈ XLEN`. Per-warp `wr_ptr`/`empty`/`full` are flip-flops.
[VX_split_join.sv](../../hw/rtl/core/VX_split_join.sv) drives push on
divergent `vx_split` and pop on `vx_join`. The dp_ram read is registered
(`RADDR_REG=1`).

### 6.2 SCS RTL deltas (per core)
| Module | Change |
|---|---|
| `VX_ipdom_stack.sv` → `VX_cstack.sv` | Same dp_ram store, but (a) entry widened by one **resume_PC** field (`+PC_BITS`); (b) `wr_ptr`-only addressing replaced by an explicit **entry-index** select driven by the scheduler; strict-LIFO push/pop becomes table insert/merge. |
| **`VX_cstack_sched`** (new, small) | Per-warp **status array** (`2 × DV_STACK_SIZE` bits: RUNNABLE/YIELDED/WAIT_MERGE) in FF/LUTRAM, + a **round-robin priority encoder** over the active warp's `DV_STACK_SIZE` entries to pick the next runnable split. |
| **watchdog** (new, tiny) | One saturating counter per warp (`~10 b`) + threshold compare; increments on back-edge-with-no-exit / contended-atomic, resets on loop-exit or successful mutation. |
| `VX_split_join.sv` | `vx_join`/`reconv_PC` match becomes order-independent (merge when all siblings of a `reconv_PC` have arrived) instead of LIFO pop. |
| `VX_scheduler.sv` | Split selection folds into the existing **pipelined** warp-select stage (decision registered one cycle ahead — see §6.4). |
| `decode.cpp`/decode RTL | Add `vx_yield` (SFU op); reuse SPLIT/JOIN encodings. |
| **`VX_cstack_spill`** (optional, v2) | Small FSM that evicts/refills `{mask,PC,reconv_PC}` descriptors to a warp-private memory region via an existing LSU port; staging register only. |

No per-thread PC, no named-barrier table, no compiler resource allocator.

### 6.3 Area model and estimate
Let `N_T = NUM_THREADS`, `N_W = NUM_WARPS`, `P = PC_BITS`,
`D = DV_STACK_SIZE = N_T−1`.

| State | Baseline IPDOM | SCS v1 (no spill) | ITS (rejected) |
|---|---|---|---|
| Stack/split store (BRAM) | `D·N_W·(1+N_T+P)` | `D·N_W·(1+N_T+2P)` | n/a (needs fast access) |
| Per-thread PC (FF/LUTRAM) | 0 | **0** | `N_T·P·N_W` |
| Per-thread state (FF) | 0 | **0** | `~3·N_T·N_W` |
| Named-barrier table (FF/LUTRAM) | 0 | **0** | `~16·(N_T+8)·N_W` |
| Split status (FF/LUTRAM) | 0 | `2·D·N_W` | (in barrier table) |
| Watchdog (FF) | 0 | `~10·N_W` | `~10·N_W` (yield ctr) |
| RR / select logic (LUT) | small | `~D`-wide pri-enc | `~N_T`-wide arbiter |

**Worked example** — representative synthesis config `N_T=32, N_W=32,
P=32, D=31` (XCU55C, per core):

- Baseline split store: `31·32·65 = 64.5 Kbit` → **~2 RAMB36**.
- SCS split store: `31·32·97 = 96.2 Kbit` → **~3 RAMB36** (**+1 RAMB36**).
- SCS added fast state: status `2·31·32 = 1984 b` + watchdog `~320 b`
  ≈ **~2.3 Kbit FF/LUTRAM**; RR pri-enc + control ≈ **~300–400 LUT**.
- **ITS added fast state**: PC `32·32·32 = 32.8 Kbit` + state `~3.1 Kbit`
  + barrier table `16·40·32 = 20.5 Kbit` ≈ **~56 Kbit of FF/LUTRAM** per
  core — i.e. **~25× more on-chip fast state than SCS**, and it is the
  expensive kind (registers/LUTRAM, not BRAM), exactly what pressures
  Fmax/occupancy.

Per-core SCS overhead vs. baseline: **+1 RAMB36, +~2.3 Kbit FF, +~0.4 K
LUT** — well under ~1–2 % of a Vortex core. Status can be moved to
distributed RAM (LUTRAM) to trade FF for LUT if FF-bound at many-core
scale. The optional spill FSM adds **~300 LUT + ~100 FF + one staging
register** and **no** extra memory macros (it uses DRAM via an LSU port).

### 6.4 Timing
- The split store remains a registered BRAM read — unchanged from
  baseline, comfortably inside the 3.33 ns (300 MHz) budget.
- The only new combinational node is the **`D`-wide round-robin priority
  encoder** (`D=31` → ~5 LUT levels, ~1.0–1.5 ns). **Design rule:**
  register the split-schedule decision one cycle ahead in the existing
  pipelined warp-select stage (the scheduler already pipelines warp
  selection), so the encoder is **never in series** with the dependent
  BRAM read or PC mux in the same cycle.
- Watchdog increment/compare is parallel and off the critical path.
- **Estimate:** 300 MHz on the U55C is maintained with baseline margin,
  provided the §6.4 design rule holds. The single timing risk is a
  same-cycle schedule→read collapse; the mitigation (decoupled schedule
  stage) is standard and already used for warp selection.

### 6.5 Caveats
These are pre-synthesis estimates from the existing module structure and
config formulas; Phase 4 replaces them with post-place-and-route numbers
(`PREFIX`-isolated build per the synthesis-run convention). The ITS column
is a structural estimate of the patent's data structures, not a built
design, included only for relative scale.

---

## 7. Correctness argument (forward progress)

Claim: under SCS every warp makes progress unless the program deadlocks
on a real MIMD machine.

Sketch:
1. Every split is, infinitely often, selected by round-robin (fairness),
   including spilled splits (§5.2). So no split is starved.
2. A split blocked on a resource is `YIELDED`/rotated; the split able to
   release that resource is therefore eventually scheduled and runs to its
   release (a bounded code region between yields).
3. Reconvergence (`vx_join` / `reconv_PC`) is order-independent (merge on
   all-siblings-arrived), so progress in any order still converges.
4. Therefore the only non-progress is a cyclic resource dependency with no
   release — i.e. a genuine MIMD deadlock, which no execution model fixes.

(A mechanized version of this argument, and the `K`/spill bound, are an
explicit deliverable in Phase 0.)

---

## 8. Comparison

| Dimension | IPDOM (today) | **SCS (this proposal)** | ITS |
|---|---|---|---|
| Per-thread PC | no | **no** | yes (large) |
| Resident per-warp HW state | `{mask,PC}` stack | + status bits + RR ptr (**few bits**) | per-thread PC + state + ~16-barrier table |
| Compiler obligation | split/join | split/join + `vx_yield` + loop structuring (**no allocation**) | barrier alloc + lifetime + yield placement |
| Convergent-code perf | optimal | **identical** | barrier-instruction tax |
| Divergent-stall perf | deadlock | switch on stall (in-table) / spill (rare) | scheduler flexibility + tax |
| FPGA Fmax / occupancy | best | **≈ IPDOM** | worse |
| Deadlock-free | no | **yes (memory-backed, unconditional)** | yes |

---

## 9. Validation plan

Per the SimX-as-oracle methodology:

- **Phase 0 — model + proof.** Formal/semi-formal forward-progress
  argument; settle `K`, spill format, watchdog threshold `T`, yield-on-
  resume policy.
- **Phase 1 — SimX.** Extend [wctl_unit.cpp](../../sim/simx/wctl_unit.cpp)
  SPLIT/JOIN to the schedulable table; add the watchdog + RR scheduler in
  [scheduler.h](../../sim/simx/scheduler.h); add `vx_yield` in
  [decode.cpp](../../sim/simx/decode.cpp). Acceptance: `lockht` and
  `lclist` **PASS** in SimX (today they hang). Add a deliberately
  over-`K` stress kernel to exercise spill.
- **Phase 2 — compiler.** LLVM-Vortex pass: `vx_yield` insertion + single-
  loop structuring; re-run `lockht`/`lclist` from source (no hand ASM).
- **Phase 3 — RTL.** Implement the schedulable table + spill in
  [VX_ipdom_stack.sv](../../hw/rtl/core/VX_ipdom_stack.sv) /
  [VX_split_join.sv](../../hw/rtl/core/VX_split_join.sv) /
  [VX_schedule.sv](../../hw/rtl/core/VX_schedule.sv); verify via xrt.
- **Phase 4 — timing.** Close 300 MHz on the U55C; confirm area delta vs.
  baseline IPDOM is within noise and far below an ITS estimate.

Regression: keep `lockht`/`lclist` out of the default CI list until
Phase 1 passes, then register them as must-pass.

---

## 10. Risks & open questions

1. **Yield-on-resume granularity.** Coarse "clear all YIELDED on any warp
   store" is simple but can over-wake (perf). A finer address-watch is
   more precise but adds state. Start coarse; measure.
2. **Watchdog threshold `T`.** Too low → needless switching; too high →
   latency before a deadlock breaks. Make it a `VX_config` knob; sweep in
   SimX.
3. **Spill bandwidth.** Pathological divergence could thrash the spill
   region. Acceptable (slowdown, not hang); compiler §5.1 should keep it
   rare. Quantify on the over-`K` stress kernel.
4. **Memory-model interaction.** Yields must respect existing
   `mem_fence`/acquire-release ordering at split boundaries; audit against
   the AMO path.
5. **Irreducible control flow.** Compiler structuring assumes reducible
   CFGs; irreducible regions fall back to the HW watchdog + spill
   (best-effort but still deadlock-free via §5.2).
6. **Indirect/opaque callees.** Cannot be compiler-instrumented; rely on
   the watchdog + spill safety net.

---

## 11. Phasing summary

| Phase | Deliverable | Gate |
|---|---|---|
| 0 | Forward-progress proof, parameter choices | reviewed |
| 1 | SimX SCS model | `lockht`+`lclist` PASS in SimX |
| 2 | LLVM-Vortex yield/structuring pass | PASS from source |
| 3 | RTL SCS + spill | PASS via xrt |
| 4 | U55C timing/area | 300 MHz, area ≈ IPDOM |
