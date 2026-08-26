# RFC: L2 hang diagnosis — response credits, arbitration, and DTCU completion delivery

> **Status (2026-08-24): root-cause claim superseded by instrumented reproduction.**
> The `AgedPriority` proposal below was based on timeout signatures, not an arbiter trace.
> A DEBUG=1/3 trace of the deterministic `a4/r8/m14` specimen disproves strict-priority
> starvation as its proximate cause.  Keep the completion-delivery proposal as a performance
> design option, but do not use aging as the correctness fix for this failure.  The validated
> cache backpressure fix and evidence are recorded in the addendum immediately below.

2026-08-22. Companion data: [`exp1_392_final_annotated_20260822.csv`](../exp1_392_final_annotated_20260822.csv)
(392-point grid; 53 cells annotated as non-terminating). Mechanism write-up first reported in the
2026-08-21 investigation; the original RFC turned it into two proposed design changes plus the
compiler-side options they enable.  Those proposals are retained below as design discussion; the
validation addendum records the subsequently implemented response-credit fix and supersedes the
original root-cause claim.

## 2026-08-24 validation addendum

### Reproduction and direct observations

The canonical test machine from the test Makefile was used without blackbox machine overrides:
1 cluster, 4 cores, 4 sockets × 1 core, 16 warps/core, 32 threads/warp, 8 effective L1/L2 request
lanes in this build, and a 1 MiB L2.  `--debug=1` was augmented with 100 K-cycle counters at the
cluster L2 fan-in and cache-bank queue snapshots; the earlier DEBUG=3 trace supplied individual
request/response tags.

For `app=4, mode=14, M=512, N=256, K=128`, the unmodified model reached this fixed point:

| cycle | observation |
| --- | --- |
| 400 K | DTCU row 3 still makes progress (`accepts=56,917`); no unbounded row-0 grant run |
| 500 K | L2 output 0 row 3 has waited 86,567 cycles, but row 0 on that output is idle; the grant is selected and rejected because the downstream channel is full |
| 500 K | socket-2 dcache bank 3 has `mem_req=2`, `mem_rsp=2`, `pipe=2` (all full); its pipe head is a new load miss requiring an egress request |
| 500 K | L2 bank 1 has `core_rsp=2`, `mem_rsp=2`, `pipe=2` (all full); its replay head cannot return to that dcache |
| 600 K | every field and head tag above is unchanged; arbiter `accepts` are unchanged while `output_blocked` grows by exactly 100,000 |

The DEBUG=3 tag path closes the loop: L2 bank 1's final queued response has bank-crossbar tag
`0x405`, which routes to L2 input 5 and then row 0, i.e. socket 2's L1 response path.  At the same
time the socket-2 dcache pipe is waiting for its full egress queue to drain into L2.

### Actual dependency cycle

```text
dcache pipe head needs mem_req_out
  -> dcache mem_req_out is full behind L2 request backpressure
  -> full dcache pipe cannot admit an already-arrived fill response
  -> that response blocks the L2 bank core-response queue
  -> the full L2 bank stops accepting the dcache's requests
  -> dcache mem_req_out can never drain
```

This is deadlock from coupled request/response queues, not livelock and not strict-priority
starvation.  In the steady state row 0 has no request on the DTCU's output, so an aged or
round-robin L2 fan-in has nothing different to grant and cannot break the cycle.

### Implemented fix and result

The validated fix is the response-credit invariant from `gfxw_v2` commit `b0886abd7`, ported to
`dtcu` as commit `2fd19c731`.  A cache bank can have up to `MSHR_SIZE` ordinary miss responses and
`AMO_PASSTHRU_CAP` uncached/AMO passthrough responses outstanding at once, so its lower-memory
response input is now sized to their sum:

```cpp
mem_rsp_in(this, config.mshr_size + AMO_PASSTHRU_CAP)
```

Previously `mem_rsp_in` had only `MSHR_SIZE` entries.  Passthrough responses could therefore use
response slots that the MSHR accounting implicitly assumed were available, close the coupled
request/response cycle above, and leave the L2 arbiter selecting a request that its full output
could not accept.  The fix adds the missing response credits; it does not add an escape FIFO or
change the normal two-entry bank pipeline and arbitration order.

The release (`-O2 -DNDEBUG`) build now completes the canonical pipe reproducers and the related
single-buffer path:

```text
[MOTI] app=4 M=512 N=256 K=128 mode=14 ... cycles=697449 errors=0
[MOTI] app=4 M=512 N=256 K=128 mode=15 ... cycles=760833 errors=0
[MOTI] app=4 M=512 N=256 K=128 mode=6  ... cycles=368661 errors=0
PASSED
```

These runs validate the cache response-credit repair for this reproduced failure family.  The
full 53-cell timeout set must still be re-run before claiming that all historical classes share
one cause.  Completion delivery can still remove polling traffic and improve performance, but it
is no longer justified as the correctness repair for class A by this evidence.  Likewise, an L2
fairness policy should be introduced only if a trace shows a continuously requesting lower row
losing accepted grants to higher-priority rows; the canonical trace instead shows downstream
output blockage.

## 1. The evidence, closed out

The full sweep — apps (1,2,3,4,5,6,9) × shapes (r2,r4,r8,r12,r16,s256,s512,s1024) × modes
(1,2,6,7,8,14,15) — completed 339/392 cells. The remaining **53 cells timed out at 12 h**, were
re-run standalone with a 36 h budget, and **0 of 53 completed in 27.2 h** before the retry was
closed. Three of them (`a4/r8/m14`, `a4/r8/m15`, `a2/r16/m1`) had already been recorded as
`hang_reproduced` in the 2026-08-18 stable run — three independent reproductions. These are not
slow points: at the simulator's measured throughput (~5 M cycles/h at these configs, e.g. s1024 m8's
19.25 M-cycle run finishing inside 4 h) a 12 h budget covers tens of millions of cycles, while the
expected costs — bounded by completed sibling cells — are 0.3–3 M cycles. The cells make no forward
progress at all.

Three distinct mechanisms, with distinct signatures:

| class | cells | mechanism | determinism |
| --- | --- | --- | --- |
| **A** | a4×m14/15 everywhere ≥ r8; s1024 m7/m14/m15 spread | consumer/poller traffic (L2 row 0) starves the engine's operand fetch (rows 2/3) under strict priority; the engine's `done` never sets; consumers wait forever on a producer they starved | deterministic for a4×pipe (all shapes, all runs) |
| **B** | m6 at r8+ (19 cells) | core store/epilogue streams (row 0) starve DXA's B-tile fetch (row 1); multi-CTA phase overlap decides whether row 0 ever drains | nondeterministic (same cell passes some runs) |
| **C** | m1/m2 cells at s256/r16/s1024 (~11) | kernel has **no unbounded loop** ([`kernel_m1.cpp`](../kernel_modes/kernel_m1.cpp) is a bounded K-loop); the simulated machine ticks forever (live probe: 2 worker threads at 100 % CPU for 31 min+, no completion) — a model-level lost-completion / stall-release bug | nondeterministic; `a2/r16/m1` reproduces every run |

This RFC addresses A and B. C is a separate simulator-debugging task and is called out in §8.

## 2. Root cause, in the code as it stands

**The arbiter has no fairness.** `PriorityArbiter::grant` ([`sim/simx/types.h:954`](../../../../sim/simx/types.h))
returns the lowest-index requester, statelessly:

```cpp
uint32_t grant(const BitVector<>& requests) override {
  for (uint32_t i = 0; i < size_; ++i)
    if (requests.test(i)) return i;
  return -1;
}
```

**And the row order puts the cores above every engine.** The L2 fan-in
([`sim/simx/cluster.cpp:122`](../../../../sim/simx/cluster.cpp)) is: row 0 = sockets (cores),
row 1 = DXA gmem, row 2 = DTCU_cluster TMA, row 3 = DTCU_socket TMA fan-in — deliberately matching
the RTL `VX_mem_arb` priority ordering. Consequence: as long as row 0 has a request in flight,
rows 1–3 are **never** granted. Any workload in which core traffic is sustained *and* the cores'
own progress depends on an engine (poll a `done` the engine must set; sleep on a barrier the DXA
must release) is a priority-inversion cycle with unbounded waiting. That is exactly modes 14/15
(class A) and mode 6 (class B).

**The DTCU's completion is a memory word the waiters must come and fetch.**
`DtcuTma::issue_done_flag` ([`sim/simx/dtcu/dtcu_tma.cpp:266`](../../../../sim/simx/dtcu/dtcu_tma.cpp))
writes a fire-and-forget 4-byte masked store into the descriptor's `done` word on the engine's read
port, resolving at the LLC. Every waiter then polls it with `dtensor_check()` — an AMO that by
design bypasses the local L1 and resolves at the LLC. `kernel_m7.cpp` spins on it **bare**
(`while (0 == dtensor_check(da));`); the pipelined modes add `MOTI_PIPE_BACKOFF` register-spins
between probes ([`k_dtcu_desc.h`](../kernel_modes/k_dtcu_desc.h) `moti_wait_desc`). The polling
itself is row-0 traffic aimed at the same L2 the engine fetches through: waiting *creates* the
traffic that starves the thing being waited for.

**The DXA already has the correct delivery structure — and it is one-sided.** The device declares
`bar.expect_tx(n)` before issuing ([`sw/kernel/include/vx_barrier.h:49`](../../../../sw/kernel/include/vx_barrier.h));
`vx_barrier_expect_tx` reaches `Core::barrier_event_attach`
([`sim/simx/wctl_unit.cpp:118-133`](../../../../sim/simx/wctl_unit.cpp),
[`core.cpp:1035`](../../../../sim/simx/core.cpp)); and when the DXA's LMEM write carrying
`dxa_notify_done` lands, a `tx_callback` fires `core->barrier_event_release(bar_id)`
([`cluster.cpp:196-215`](../../../../sim/simx/cluster.cpp)). Waiters sleep in the barrier unit and
generate **zero** memory traffic. So the machine has both halves of NVIDIA's answer already built —
delivery for the DXA, and nothing for the DTCU — which is why m6 hangs *less* than the pipe modes
(one starvation ingredient instead of two) but still hangs: delivery removes the **waiter's**
traffic, and does nothing about **third-party** core traffic outranking the producer at the arbiter.

## 3. What the commercial answer looks like (and which halves we are missing)

NVIDIA's tensor pipeline (Ampere `cp.async.wait_group` → Hopper TMA + `mbarrier` + warp
specialization → Blackwell `tcgen05` + TMEM) rests on three decisions:

1. **Completion travels TO the waiter's local scope** — TMA arrives at an mbarrier in the
   consumer's own shared memory (one write per completion, by the producer).
2. **Waiting is a hardware sleep**, not a memory access (`mbarrier.try_wait`, scoreboard waits).
3. **No strict-priority row where core traffic always beats the async engine** — steady-state
   tensor traffic is bulk and the arbitration is effectively fair.

The DXA implements (1)+(2) for itself. Nothing implements (3), and the DTCU implements none.
Proposal §4 supplies (3); proposal §5 gives the DTCU (1)+(2) by copying the DXA's own plumbing.

## 4. Proposal 1 — `AgedPriority` arbiter: bounded waiting at the L2 fan-in

### Design

A new arbiter type beside `Priority`/`RoundRobin` in [`types.h`](../../../../sim/simx/types.h):

```cpp
class AgedPriorityArbiter : public IArbiterImpl {
public:
  AgedPriorityArbiter(uint32_t size, uint32_t max_age)
    : size_(size), max_age_(max_age), age_(size, 0) {}

  uint32_t grant(const BitVector<>& requests) override {
    // 1) starvation override: oldest input at/over the age bound wins
    uint32_t victim = -1u, victim_age = 0;
    for (uint32_t i = 0; i < size_; ++i) {
      if (requests.test(i) && age_[i] >= max_age_ && age_[i] >= victim_age) {
        victim = i; victim_age = age_[i];
      }
    }
    uint32_t g = (victim != -1u) ? victim : lowest_index(requests);   // 2) else RTL priority
    for (uint32_t i = 0; i < size_; ++i)                              // 3) age the losers
      age_[i] = (requests.test(i) && i != g) ? age_[i] + 1 : 0;
    return g;
  }
  ...
};
```

- `grant()` is invoked once per arbitration tick, so `age_[i]` counts **ticks spent requesting
  without a grant** — precisely the bounded-waiting quantity. Worst-case wait becomes
  `max_age + (rows-1)` ticks instead of ∞.
- Under light contention the behaviour is bit-identical to today's `PriorityArbiter` (ages never
  reach the bound), so the RTL-priority modelling argument in the cluster.cpp comment is preserved
  where it was ever valid.
- Wiring: `MemArbiter::Create(sname, ArbiterType::AgedPriority, ...)` at the **L2 fan-in only**
  ([`cluster.cpp:153`](../../../../sim/simx/cluster.cpp)) to start; `max_age` from a new config
  `-DVX_CFG_L2_ARB_MAX_AGE=<W>` (default 256, see below). A follow-up audit should list every other
  `ArbiterType::Priority` instance sitting on a path where an engine's progress gates core progress
  (the socket dcache fan-in that carries the socket engine's D port,
  [`socket.cpp:188`](../../../../sim/simx/socket.cpp), is the first candidate).

### Choosing W — and why it is a compiler-shaped number

W trades engine wait (want small) against perturbing the RTL priority model (want large). The
useful bound is the consumers' *slack*: a granted engine line only delays a core request by one
tick, and the consumers' per-slice epilogue is thousands of cycles long, so W anywhere in
64–1024 is invisible to consumer throughput while capping engine wait. This is statically
computable — per-slice consumer work = `rows/slice × N/thread × streams` — which is exactly why
§6 argues the *policy* (W per kernel, via a DCR at launch) belongs to the compiler even though the
*mechanism* is hardware. Validation sweep: W ∈ {64, 256, 1024} × {a2,a4} × {m6,m14} at r8.

### What it buys (predictions, falsifiable)

- All 22 class-A/B cells terminate. The winner map de-contaminates: r8's row should largely flip
  to wgSB (its base ≈283 K is the fastest at r8 and it forfeited 5/7 apps), and s1024's m7 column
  gets real numbers.
- Cells that already finish move ≲1 % (the bound almost never engages for them).
- a4×pipe becomes *finite but still slow* — aging bounds waiting, it does not reduce the consumer
  traffic. That is proposal 2's job.

## 5. Proposal 2 — DTCU completion delivered like the DXA's: `expect_tx` / barrier release

### Descriptor and ISA surface

Extend `dtensor_desc_t` with a notify target (two words in the existing 64 B line):

```
uint32_t notify_mask;   // bit c: release a barrier on core c on completion. 0 = legacy polling.
uint32_t notify_bar;    // encoded barrier id, same encoding DXA uses
                        // (low byte = cta_no, bits[30:8] = bar_no — bar_decode_id())
```

The kernel already builds its own descriptors (`moti_fill_desc`), so no `kernel_arg_t` change and
no host change: the kernel writes its own `bar.id()` — the same call m6 already makes for DXA.

### Engine side (simulator)

`issue_done_flag()` keeps the memory store — the flag word stays the architectural completion for
legacy pollers and for the host. Ordering is already correct for delivery too: the store is only
issued after **every D line is ACKed** (`dtcu_tma.cpp:770-777`), so firing the notify at the same
point inherits the data-before-flag guarantee. Two wiring options:

- **(i) Direct callback (modelling shortcut, recommended first).** At cluster/socket construction
  time — where the cores are in scope — install a closure on the Dtcu:
  `dtcu->set_notify([cores](uint32_t mask, uint32_t bar){ for c in mask:
  cores[c]->barrier_event_release(bar_decode_id(bar, VX_CFG_NUM_BARRIERS)); })`, invoked from
  `issue_done_flag()`. This is the same modelling level as the DXA's `tx_callback` in
  [`cluster.cpp:207`](../../../../sim/simx/cluster.cpp) — a completion edge, not new datapath.
- **(ii) Doorbell packet (RTL-faithful follow-up).** A 1-flit message per set bit in
  `notify_mask`, riding the socket engine's existing `d_req_out` into the socket dcache slot
  (which already reaches per-core LMEM inputs), DXA-style with a `notify_done` flag; the cluster
  engine, which has no per-core port, gets a dedicated doorbell channel — the honest cost of the
  cluster placement, and worth exposing since **placement-as-port-topology is this harness's
  thesis**: the cluster engine paying extra for completion multicast is a *finding*, not plumbing.

The multicast case (m15: producer on core 0, consumers on all cores) is why `notify_mask` is a
mask. The DXA multicast already releases per-core barriers today, so (ii) has an in-tree precedent.

### Device side

`moti_wait_desc()`'s poll loop is replaced by the m6 pattern, one `expect_tx` per slice:

```cpp
// consumer_warp == 0, lane == 0, before the producer can start slice t:
ready.expect_tx(1);          // was: moti_wait_desc(da)  [AMO poll + BACKOFF spins]
ready.arrive_and_wait();     // barrier completes when cw warps arrived AND engine released the tx
```

`MOTI_PIPE_BACKOFF` is deleted rather than tuned — the knob exists only because polling exists.
`kernel_m7/m8`'s bare submitter spin becomes a single-warp barrier with one expected tx. Host-side
`dtensor_check` polling (if any path still uses it) is unaffected: the flag word is still written.

### What it buys (predictions, falsifiable)

- Class A's waiter-traffic half disappears entirely: zero poll AMOs, zero backoff sensitivity.
  At r8/relu the pipe adder Δ(m14) = +93.9 K should drop toward and below the two-launch
  Δ(m7) = +49.3 K; at s1024, m14's 1.53 M (already the winner) improves and m7/m15 stop dying.
- Combined with §4, the pipelined modes finally measure what they were built to measure —
  overlap — instead of measuring arbitration pathology.

## 6. Compiler-side options (what remains software after the hardware is honest)

1. **W selection per kernel.** Bounded-wait W is a slack computation over static quantities
   (consumer per-slice work, stream count, N) — emit it as a launch-time DCR write. This is the
   Aérgia/criticality argument done statically: consumer traffic is *elastic* (a stalled warp
   absorbs it), engine fetches are *critical* (everything transitively waits), and the compiler can
   prove which is which from the kernel structure instead of inferring it in hardware.
2. **Throttle synthesis as the no-HW fallback.** Until §4/§5 exist, the livelocks are held off by
   hand-tuned constants with measured cliff edges: `MOTI_PIPE_BACKOFF` 32 → 45,645 cycles vs
   4096 → 808,638; consumer width 4 warps finishes where 8 livelocks; slice-on-tiles vs rows is
   4× arithmetic. Every one of these is derivable from shape + machine constants. A compiler that
   emits them per-shape turns "nondeterministically hangs" into "reliably slow" — strictly weaker
   than §4/§5 but deployable today.
3. **Broadcast-operand promotion (fixes the a5 pathology §8.2).** `s[N]` is a read-only uniform
   vector; one LMEM copy per CTA removes the measured re-fetch storm (r4/m6: 17,261 DRAM reads vs
   5,903 for relu — 8 cache lines fetched thousands of times — and 10× SFU barrier-idle
   amplification). Classic scratchpad promotion, entirely compiler-side.
4. **Phase-tagged QoS.** With §4's mechanism in place, a launch-time weight ("engine-favoured
   during GEMM phases") is the static analogue of slack-aware NoC prioritisation — optional, and
   only worth pursuing if the W sweep shows a measurable gap between W=64 and W=1024.

## 7. Validation matrix

| experiment | proves |
| --- | --- |
| §4 alone, W∈{64,256,1024}: r8 × {a2,a4} × {m6,m7,m14} | termination of A/B; finishing cells unmoved; W insensitivity band |
| §4+§5: same grid + s1024 all modes | Δ(pipe) ≤ Δ(two-launch); m7/m15 s1024 real numbers; backoff deleted with no regression |
| re-run 392 grid | de-contaminated winner map (the actual deliverable of this harness) |
| §6.3 lmem promotion, m6 × a5 × {r4,r12,s512} | a5 adder collapses toward the relu/scale floor |

## 8. Explicitly out of scope

1. **Class C (m1/m2 non-termination).** No polling, no engine, no unbounded loop in the kernel —
   the model itself stops retiring. Needs its own investigation; `a2/r16/m1` reproduces every run
   and is the specimen to debug. Nothing in §4/§5 will touch these ~11 cells.
2. **The a5×wgSB slowdown** is an L1-thrash + barrier-amplification problem (§6.3), not an
   arbitration problem; aging only removes its occasional escalation into a class-B hang.
3. **The regime structure.** r16's in-core dominance (fixed engine array vs scaling in-core
   aggregate) and r2's engine dominance are base-performance facts and will survive both proposals;
   what changes is that the *middle* of the map — r8 through s1024 — gets decided by measurement
   instead of by which mode happens to terminate.
