# Multi-Cache AMO Coherence — Design

**Scope:** the behavior of RISC-V "A"-extension atomics across a multi-cache
hierarchy (4× L1 → 1× L2 and 4× L2 → 1× L3), in both the RTL
([`hw/rtl/cache/VX_amo_unit.sv`](../../hw/rtl/cache/VX_amo_unit.sv),
[`VX_cache_bank.sv`](../../hw/rtl/cache/VX_cache_bank.sv),
[`VX_cache_tags.sv`](../../hw/rtl/cache/VX_cache_tags.sv),
[`VX_cache_mshr.sv`](../../hw/rtl/cache/VX_cache_mshr.sv)) and the SimX model
([`sim/simx/amo/`](../../sim/simx/amo/),
[`sim/simx/mem/cache.cpp`](../../sim/simx/mem/cache.cpp)).

This document builds on the single-LLC atomics design
([`atomic_memory_operations.md`](atomic_memory_operations.md)) — the decode,
LSU sideband, AMO ALU, and "atomics resolve at the last-level cache (LLC)"
foundation are described there and assumed here.

**Coherence model:** GPU-weak / PULL, matching NVIDIA, ARM Mali, and
Imagination PowerVR — atomics resolve at the LLC; inner caches are
write-through and not hardware-coherent; cross-core *plain-data* visibility
is restored by consumer-side invalidation at acquire points, not by snooping.

---

## 1. The problem

Resolving every atomic at the LLC bank is sufficient for a single L1 (the L1
*is* the LLC). Once L2 (or L2+L3) is enabled, three things must hold that a
naive single-LLC design does not provide:

1. **LR/SC forward progress under contention** — every hart must eventually
   win its `SC`, independent of how many harts contend.
2. **Atomic correctness across levels** — an atomic arriving at a non-LLC
   bank must be forwarded to the LLC (which owns the RMW) and its result
   routed back, without leaving a stale or duplicate copy behind.
3. **Issuer self-consistency** — a hart that issues an atomic and later does
   a plain load of the same address must observe its own update, even though
   its L1 is write-through and non-coherent.

A *fourth* concern — a **plain load on another core** holding a stale copy
of an atomically-updated line — is the GPU-weak model's deferred case
(Regime B, §6), resolved by a consumer acquire-invalidate exactly as
`ld.acquire` / `__threadfence` do on a real GPU.

---

## 2. Reference architecture: how GPUs do this

The cross-vendor pattern is unanimous and is **PULL**, not directory-based:

- **Atomics resolve at the shared L2 / memory partition** by dedicated RMW
  ALUs, with the target line locked for the duration (the serialization
  point). NVIDIA performs the RMW at each memory partition's Atomic
  Operation Unit; PowerVR reserves the L2 line so it "cannot be read or
  written until the atomic operation has been completely processed"; Mali
  resolves atomics near the L2 coherency point.
- **Per-core L1 caches are write-through and not hardware-coherent.** Mali:
  "coherency is guaranteed only for the LSC; the driver manages the other
  caches." NVIDIA: write-through L1 with non-coherent stores that reach the
  level "at which a cache coherency policy is enforced" (L2).
- **Cross-core visibility is restored at synchronization points** by
  flush/invalidate, not snooping — the consumer-side bulk L1 invalidate
  behind `ld.acquire` / `__threadfence`.

Sources: [GPU atomics @ memory partition (HPCA'13)](https://www2.cs.sfu.ca/~ashriram/papers/2013_HPCA_GPUCoherence.pdf),
[non-coherent write-through L1 (US 9047197)](https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/9047197),
[barrier-initiated flush/invalidate (US 9563561)](https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/9563561),
[PowerVR L2-locked atomics (US 8108610)](https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/8108610),
[Mali Bifrost (Hot Chips 28)](https://old.hotchips.org/wp-content/uploads/hc_archives/hc28/HC28.22-Monday-Epub/HC28.22.10-GPU-HPC-Epub/HC28.22.110-Bifrost-JemDavies-ARM-v04-9.pdf).

Vortex already resolves atomics at the LLC and enforces write-through inner
caches ([`Vortex.sv`](../../hw/rtl/Vortex.sv),
[`processor.cpp`](../../sim/simx/processor.cpp)); this design adds the
forwarding, self-consistency, and consumer-invalidate pieces.

---

## 3. Design decisions

| Question | Decision | Rationale |
|---|---|---|
| Coherence scope | AMO-triggered, weak (GPU) model | Plain store→load across cores stays the programmer's responsibility (fences), as on every real GPU. Smallest change. |
| How is a remote stale copy cleaned? | **PULL** | The consumer invalidates its *own* inner cache at an acquire fence; the LLC never pushes invalidations upward (what NVIDIA/Mali do; least code). |
| Topology | Same cache module at every level | The mechanism lives inside the one reusable cache; recursion to L3 is free (§5.6). |
| Reservation tracking | **Per hart** | One reservation per hart matches RISC-V and guarantees forward progress (§5.1). |

**Rejected: PUSH + directory/snoop-filter.** Back-invalidating remote inner
caches from the LLC would make atomics coherent with no software fence, but
needs an upstream back-probe channel, a per-line sharer directory, an
ack/ordering protocol, and recursive re-probing — a whole coherence
subsystem that does not scale O(cores) and that real GPUs avoid. Recorded as
future work (§7).

---

## 4. Core invariant and the two regimes

> **An atomic never leaves a copy of its line in any inner (non-LLC) cache.**

It holds because (a) the requesting cache self-invalidates the line as the
atomic is forwarded down, (b) the atomic is non-allocating, so no fill is
installed afterward, and (c) the atomic never touches sibling inner caches.
The only way an inner cache can hold a line being atomically updated is a
**plain load** on that core. This splits all behavior into two regimes:

- **Regime A — atomic-only sharing** (spinlock, atomic counter; the common
  case). No inner cache ever holds the line; every atomic serializes at the
  LLC. **Fully implemented and validated.** Zero new coherence traffic.
- **Regime B — a plain load cached the line on another core.** That copy is
  stale after a remote atomic. Resolved by the consumer's acquire-invalidate
  (§7), exactly as a GPU requires `ld.acquire` / `__threadfence`. **Deferred**
  until a mixed atomic/plain-load workload needs it.

---

## 5. Mechanisms (implemented)

Each mechanism is realized in both SimX and RTL. The SimX model was brought
up first as the goal-reference oracle; the RTL mirrors its behavior at the
cache-bank microarchitecture.

### 5.1 Per-hart LR/SC reservations

A shared, capacity-bounded reservation table with LRU eviction does not
guarantee forward progress: once contending harts exceed the table size, a
hart's reservation can be evicted by another hart's `LR` between its own `LR`
and `SC`, so the `SC` never succeeds (`lrsc_counter` hangs at ≥ 64 harts).

The reservation store is **per hart** — each hart owns exactly one
reservation, never displaced by another hart's `LR`, broken only by a
committed write to the line. This matches RISC-V (a reservation is a property
of the hart) and guarantees a winner each retry round.

- **SimX:** a `hart_id → line` map in
  [`amo_unit.{h,cpp}`](../../sim/simx/amo/amo_unit.cpp).
- **RTL:** a directly-indexed array of `NUM_HARTS = 1 << HART_ID_WIDTH` slots
  `{valid, line_addr}` in [`VX_amo_unit.sv`](../../hw/rtl/cache/VX_amo_unit.sv);
  `reserve`/`check` index the requester's slot, `invalidate` clears matching
  slots of other harts. `VX_CFG_AMO_RS_SIZE` is retained for compatibility
  and no longer bounds correctness.

### 5.2 Plain-write reservation invalidation

The LLC breaks a hart's reservation on **any** committed write to the
reserved line, not only atomic stores, so a plain store from another hart
fails a racing `SC`. SimX pulses `invalidate(line, except=hart)` on every LLC
write-through path; RTL drives `amo_res_invalidate_w = amo_do_store_st1 ||
do_write_st1` into the AMO unit.

### 5.3 Non-LLC AMO passthrough

A non-LLC bank does not own the atomic. It forwards the AMO downstream
non-allocating and routes the result word back up.

- **SimX:** an `AmoProbe` ([`cache.cpp`](../../sim/simx/mem/cache.cpp))
  probes the local line, invalidates on hit, and forwards without installing
  a fill.
- **RTL:** the existing miss→fill→replay path is **reused** rather than
  adding a separate side-table ([`VX_cache_bank.sv`](../../hw/rtl/cache/VX_cache_bank.sv)
  `g_amo_ptw`). The AMO allocates a normal MSHR entry, flagged in a parallel
  `amo_ptw_flag[]` with the result word-select in `amo_ptw_wsel[]`. On the
  downstream fill the flagged entry captures the result word, **installs no
  line**, and replays carrying the result up to `core_rsp`
  (`eff_hit_st1 = is_hit_st1 || is_amo_replay_st1`). AMOs are excluded from
  MSHR coalescing (`amo_table` masks them out of `addr_matches`, and the
  requester is forced non-pending) so each atomic takes its own round-trip;
  at the LLC, same-line AMO coalescing is preserved (`allocate_is_amo` gated
  to non-LLC).

### 5.4 Issuer self-consistency

A hart that issues an atomic and later plain-loads the same address must see
its own update. As the atomic is forwarded, the issuing cache invalidates its
local copy so the next load misses and refetches from the LLC. In SimX this
falls out of the `AmoProbe` self-invalidate; in RTL,
[`VX_cache_tags.sv`](../../hw/rtl/cache/VX_cache_tags.sv) gains an
`invalidate` input that clears `line_valid` on a tag match (using a raw hit
that excludes a line being filled this cycle), driven by
`is_amo_fwd_st0 && is_hit_st0`.

### 5.5 AMO/fill age-ordering

A load-then-AMO (or AMO-then-load) to the same line can race the
probe/invalidate against an in-flight fill in the bank pipeline. Symmetric
age-ordering holds the younger access at admission until the older drains:

- an incoming AMO waits while a load fill is pending for its line (so its
  invalidate lands on the installed line);
- an incoming plain load waits while an AMO passthrough is pending for its
  line (so the load observes the AMO — same-hart same-address program order).

SimX defers the `AmoProbe` in `processInputs` and re-issues a vanished-line
replay as a fresh miss. RTL uses MSHR probe ports
(`probe_pending_ld` / `probe_pending_amo`,
[`VX_cache_mshr.sv`](../../hw/rtl/cache/VX_cache_mshr.sv)) to drive
`amo_input_defer` / `load_input_defer` at the bank input, plus a same-cycle
allocation guard for the admit→allocate window.

### 5.6 LLC AMO commit-window serialization (RTL)

The LLC AMO writeback path is single-outstanding (it chains only same-line
AMOs). Admitting a second AMO to a different line while the first is still in
its writeback window would let it reach S1 with the writeback busy and drop
its store. The bank closes the whole window — `amo_commit_busy = amo_wb_pending
|| amo_do_store_st1 || amo_do_store_st0` gates the AMO request path and
`core_req_ready`. All terms are zero at non-LLC banks, so this serialization
applies only at the LLC.

### 5.7 Recursion to L3 is free

The mechanism is per-cache and the same cache module is instantiated at every
level. An atomic under an L3 config travels L1→L2→L3, self-invalidating L1
**and** L2 on the way down (both act as non-LLC), and the L3 performs the RMW.
No per-level special-casing and no inter-level messaging beyond the normal
request flow. The `AMO_ENABLE` parameter on each cache is driven by
`VX_CFG_EXT_A_ENABLED` alone (not gated to the LLC), so non-LLC data caches
synthesize the passthrough; the LLC-vs-passthrough distinction is made inside
the bank.

---

## 6. Worked walkthroughs

### 6.1 4× L1 → 1× L2 (L2 = LLC)

```
   core0   core1   core2   core3
    L1_0    L1_1    L1_2    L1_3      write-through, non-coherent
      └──────┴───┬───┴──────┘
                 L2 (LLC: RMW + per-hart reservations)
                 memory
```

**Regime A — `amoadd counter` by all cores:** each core's atomic invalidates
its own L1 copy of `counter` and increments at L2; no L1 ever holds
`counter`; the final value and the unique fetch-add returns are correct with
zero cross-L1 traffic.

**Regime B — producer/consumer via a plain load:** if `core0` previously
plain-loaded `flag` (caching it) and `core3` then `amoadd flag,1`, `core0`'s
next plain load hits its stale L1 copy until `core0` issues an
acquire-invalidate (§7), after which it misses and refetches the current
value from L2. The L2 never reaches up.

### 6.2 4× L2 → 1× L3 (L3 = LLC)

```
 cluster0           ...            cluster3
 L1×4 → L2_0                       L1×4 → L2_3
    └───────────────┬─────────────────┘
                  L3 (LLC: RMW + per-hart reservations)
                  memory
```

A core in cluster0 doing `amoadd X` self-invalidates **L1 and L2_0** en route
(both non-LLC) and the L3 performs the RMW. Identical mechanism, one extra
level traversed, no new code.

---

## 7. Deferred: Regime-B acquire-invalidate

The one unimplemented piece is the consumer-side **bulk invalidate** at an
acquire synchronization point — clearing the inner cache's valid bits (no
writeback; inner caches are write-through, never dirty) so the next plain load
refetches from the LLC. The design reuses the existing flush walk with an
invalidate mode and routes a `FENCE`/acquire through the LSU as a new memory
op on the existing core→cache path — no new fabric channel. It is deferred
until a mixed atomic/plain-load workload requires fence-managed cross-core
data visibility.

---

## 8. Validation

`tests/regression/amo` across all configs, on both SimX and rtlsim:

| config | result |
|---|---|
| 1 core, no L2 | 13/13 |
| 4× L1 → 1× L2 | 12/12 |
| 4× L2 → 1× L3 | 12/12 |

The suite includes `lrsc_counter` (forward-progress repro for §5.1) and
`self_consistency` (per-hart private 64 B line; exercises §5.4/§5.6 and fails
a design lacking the local-invalidate and commit-window serialization).
`atomic_critical` is skipped on multi-core: it relies on plain load/store
inside a critical section, which needs Regime-B coherence (§7).
`rv32ua`/`rv64ua` ISA conformance (LR/SC + all AMOs) passes. The full
`regression --cache` suite (L1/L2/L3 enable+disable, banking, ways,
replacement policy, writeback, clustering, reduced line-size) passes,
confirming the `AMO_ENABLE` plumbing and bank/tag/MSHR changes do not regress
non-AMO cache behavior.

Outstanding: xrt sign-off at the multi-core configs and U55C @ 300 MHz timing
closure. The new state is the per-hart reservation array (scales with
`NUM_HARTS`) and the per-MSHR-entry passthrough word table; if the §5.1
`invalidate` compare or the §5.4 local-invalidate is critical, it can be
pipelined off the SC-success / hit path (both tolerate an extra cycle).

---

## 9. Out of scope / future work

- **Regime-B acquire-invalidate** (§7) — the consumer-side bulk invalidate
  for fence-managed cross-core plain-data visibility.
- **PUSH + directory/snoop-filter** — fence-free cross-core plain-data
  coherence; requires an upstream back-probe channel, per-line sharer
  tracking, an ack/ordering protocol, and recursive re-probing. Not what GPUs
  do; revisit only if a strong cross-core guarantee is ever required.
- **Per-line acquire invalidate** — finer-grained than the whole-cache flash
  invalidate, if profiling shows bulk invalidation hurts hit rate.
- The RTL AMO unit testbench and AMO performance counters
  ([`atomic_memory_operations.md`](atomic_memory_operations.md) §6.3–§6.4).
