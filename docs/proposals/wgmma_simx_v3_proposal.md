# WGMMA SimX v3 — Proposal

**Date:** 2026-04-28
**Status:** Draft
**Related:**
[simx_v3_proposal.md](simx_v3_proposal.md),
[simx_rtl_perf_divergence.md](simx_rtl_perf_divergence.md),
`project_wgmma_cta_limit`.

---

## 1. Constraints (load-bearing)

Any proposal that breaks one of these is wrong.

1. **NoC-only memory access.** Every fetch — including tile A, B, and
   metadata — flows through the channel hierarchy
   (`LmemReq`/`LmemRsp` against LMEM). No `core_->mem_read`, no
   synchronous backing-store reads. **Note:** LMEM is a banked
   scratchpad with no cache layer, no MSHR, and no coalescing — every
   request consumes its own LMEM-port cycle.
2. **Functional and timing coupled.** Per
   [simx_v3 §3.3](simx_v3_proposal.md#33-functional-and-timing-meet-in-exactly-one-place),
   data flows on the same packets that carry timing. The functional value
   of an A or B element *is* the bytes the response packet delivers at the
   cycle it arrives. No separate synthetic delay, no second source of
   truth.
3. **Out-perform RTL within constraints 1 and 2.** Goal: faster simulator
   wall-clock and smaller working-set than a literal RTL transcription
   would give. Silicon constraints driving RTL's per-block tile-buffer
   slot do not bind C++. Mirror the RTL at module *correspondence*, not at
   internal storage and fetch policy.

---

## 2. Why the current SimX WGMMA is broken

[sim/simx/tcu/tensor_unit.cpp](../../sim/simx/tcu/tensor_unit.cpp):

- **Rule 1:** A and B fetches abort
  ([line 1052-1058](../../sim/simx/tcu/tensor_unit.cpp#L1052)); sparse SS
  metadata aborts ([line 903-904](../../sim/simx/tcu/tensor_unit.cpp#L903)).
- **Rule 2:** the timing layer schedules `ready_cycle`; the functional
  layer is the abort. Non-overlapping, not just decoupled.
- **Rule 3:** when functional fetches do run (reference branch via
  `core_->mem_read`), they materialize full host-side A and B tiles —
  same working-set order as RTL.

| # | Defect | Reference |
|---|--------|-----------|
| D1 | Iteration order m-inner / k-middle / n-outer; comment claims "matching RTL" | [tensor_unit.cpp:1394-1399](../../sim/simx/tcu/tensor_unit.cpp#L1394) vs [VX_tcu_uops.sv:128](../../hw/rtl/tcu/VX_tcu_uops.sv#L128). RTL is k-outer / n-middle / m-inner. k and n are swapped. |
| D2 | Unit-global `TileBufferState` (one struct for all Q blocks) | [tensor_unit.cpp:533-550](../../sim/simx/tcu/tensor_unit.cpp#L533) |
| D3 | B-cache key is descriptor only — false hits across CTAs | [tensor_unit.cpp:959-964](../../sim/simx/tcu/tensor_unit.cpp#L959) |
| D4 | No Q-warp lock-step coupling — deadlock unreproducible | per-warp `fu_lock` only at [tensor_unit.cpp:1420-1422](../../sim/simx/tcu/tensor_unit.cpp#L1420) |
| D5 | Sparse SS metadata path also aborts | [tensor_unit.cpp:903-904](../../sim/simx/tcu/tensor_unit.cpp#L903) |

---

## 3. Critique of the original optimization hint

The hint identifies the right axes (working set, redundant fetches, reuse,
lock-step). Several details fail under the three rules.

- **F1. Iteration order.** "k_step ⇒ m_steps ⇒ n_steps" is ambiguous. The
  right order is **k-outer / n-middle / m-inner** — same shape as RTL —
  but for a different reason: with k outer, A[*,k] and B[k,*] blocks are
  reused across the entire inner (m, n) sweep, which is what makes a
  small in-flight working-set viable. With k inner the reuse window
  collapses and every uop needs a fresh fetch.
- **F2. Cross-warp dedup as a *per-uop descriptor scan* is wrong, but
  as a *structural shared buffer* it is the design.** The hint conflates
  two things. (a) Detecting at runtime that two Q lanes happen to share
  a descriptor and fusing their LmemReqs is a per-uop scan paid every
  cycle for a rare event — wrong tradeoff. (b) Holding a single shared
  B buffer that all Q blocks read from, valid by construction whenever
  the TB uses M-direction warp tiling (the production-common case
  where every warp sees the same B descriptor) — cheap, structural,
  halves SMEM B traffic. (b) is what §4.1 adopts. (a) is rejected.
- **F3. The strongest "minimize working set" is a *small structured
  buffer*, not a full tile.** Per-TB working set can shrink from RTL's
  ~6 KB per-block-tile-buffer down to ~1.1 KB by holding only the
  active k-stripe of A per block plus a single shared B row across Q.
  There is no LMEM cache to fall back on, so this storage *is* the
  design. The hint stops short of this distinction.
- **F4. Double-buffering** beyond what channel pending-count already
  gives is host memory wasted for no simulator-time gain.
- **F5. Descriptors and registers** are not the working-set axis.
- **F6. Lock-step Q is worth modeling — as a probe, not a coalescer.**
  The probe reproduces the documented `NUM_WARPS > ISSUE_WIDTH` deadlock;
  using it as a memory coalescer adds cross-warp synchronization that
  hurts simulator runtime.

The genuine insight: **a small *structured* TCU buffer — per-block A plus
one shared B — captures the entire usable reuse pattern at a fraction of
RTL's per-block full-tile storage. There is no LMEM cache to lean on; the
buffer is the design.**

---

## 4. Proposed Design (implementation-independent)

This section describes *what to build*. It is written without commitment to
SimX vs RTL — the same design applies to both. §5 then describes the SimX
form concretely.

### 4.1 Layering decision: minimal structured TCU buffer

The current Vortex design (both RTL and the half-built SimX) places **tile
persistence inside the TCU per-block tile buffer**: each of the Q blocks
holds the entire `(M_STEPS × K_STEPS)` A tile and `(K_STEPS × N_STEPS)` B
tile for the duration of the WGMMA. The proposal **preserves the
principle** of holding tile data inside the TCU but **dramatically
reduces the working set** by structuring the storage:

> **The TCU holds a minimal structured tile buffer:**
> - **Per-block A buffer** — `M_STEPS` bank-rows per block (the active
>   k-stripe's A rows for that warp; ≈ `M_STEPS × NT × 4 B` per block).
> - **Shared B buffer** — `1` bank-row across all Q blocks, optionally
>   `2` with ping-pong (≈ `NT × 4 B` for the whole TCU).
>
> There is **no LMEM cache layer** between the TCU and SMEM. LMEM is a
> banked scratchpad accessed via a single arbitrated port; **there is no
> MSHR and no coalescing**. Reuse comes entirely from the TCU's own
> structured buffers, never from cache-side magic.

Working-set total per active TB:
`Q × M_STEPS × bank-row + 1 × bank-row`
(or `+ 2 × bank-row` with ping-pong)
= **≈ 1.1 KB at Q = 4, NT = 32, M_STEPS = 2** — vs RTL's
per-block full-tile `Q × (M_STEPS × K_STEPS + K_STEPS × N_STEPS) ×
bank-row ≈ 6 KB`. **About 5× smaller.**

Consequences:

- The shared B buffer is the entire B-reuse mechanism. When all Q warps
  share the B descriptor (the production-common case for CTA-level GEMM
  with M-direction warp tiling), one LmemReq populates one bank-row
  consumed by all Q blocks. There is no per-block B copy.
- Per-block A buffer holds only the *current k-stripe's* A rows. At
  k-transition the buffer is overwritten; A rows for prior k are dropped.
- Ready-to-advance for block `b` becomes: "block `b`'s `A_w[m, k]` row is
  resident in its A buffer AND the shared B buffer holds `B[k, n]`".

This is a layering choice, not a SimX-only optimization. **The same
shape applies in RTL** — same SRAM savings, same shared-B sharing path,
same cross-WGMMA opportunity (§4.7).

### 4.2 Fetch granularity: bank-row, not element, not tile

Tile data moves between LMEM and the TCU's buffers at **bank-row
granularity** — one `NUM_BANKS`-word block per LmemReq/LmemRsp packet.
With `NUM_BANKS = NT` in Vortex, one bank-row = `NT × 4 B = 128 B` at
NT = 32. This matches the LMEM port width (one bank-row per cycle,
pipelined; see [VX_tcu_tbuf_fetch.sv:293-308](../../hw/rtl/tcu/VX_tcu_tbuf_fetch.sv#L293)).

Bank-row granularity gives:

- **Far fewer channel events than per-element fetches** — critical for
  simulator wall-clock and for RTL gate/wire count.
- **Natural alignment with LMEM bank-row organization.**
- **No stride-aware sub-line packing on the channel**; format-aware
  gather (fp32/fp16/fp8 lane extraction) happens *after* the bank-row
  arrives, inside the TCU.

Multiple in-flight LmemReqs are allowed via channel pending-count; the
LMEM port arbitrates and pipelines them. **Two requests to the same
bank-row that miss simultaneously each cost their own port cycle —
LMEM has no MSHR and no coalescing.** Same-bank-row reuse is achieved
by the TCU's structured buffers, never by collapsing requests at LMEM.

### 4.3 Iteration order: k-outer for buffer reuse

Order is fixed at uop-generation time:

```
for k = 0 .. K_STEPS-1:
  for n = 0 .. N_STEPS-1:
    for m = 0 .. M_STEPS-1:
      uop(m, n, k)
```

Within fixed `k`, all `M_STEPS × N_STEPS` uops touch only A[*, k] (the
current k-stripe of A) and B[k, *] (the current k-stripe of B). For
typical WGMMA sizes (M_STEPS = K_STEPS = 2; N_STEPS ∈ {4, 8, 16}), one
k-stripe is **`M_STEPS × Q + N_STEPS` bank-rows total** — `M_STEPS × Q`
per-warp A lines plus `N_STEPS` shared B lines.

K-outer maximizes both buffer-reuse modes:

- **A reuse**: the `M_STEPS` A rows resident in each block's A buffer
  are consumed across the entire `(n, m)` inner sweep — `M_STEPS ×
  N_STEPS` uops per stripe per block — and only refilled when k advances.
- **B reuse**: each B-row stays in the shared B buffer for `M_STEPS = 2`
  consecutive uops (the m-inner sweep over fixed `(k, n)`).

This is the same order RTL chose, but for a different rationale:

- **RTL** — overlap independent `(m, n)` MMA pairs across FEDP pipeline
  latency.
- **Proposed design** — amortize each LmemReq across the maximum number
  of consuming uops. Each per-warp A row is read from LMEM once per
  k-stripe and consumed `N_STEPS` times. Each shared B row is read once
  per `(k, n)` pair and consumed `M_STEPS = 2` times.

The conclusion converges; both rationales call for k-outer.

### 4.4 Lock-step Q is a dispatch invariant

The Q hardware blocks (per-warp lanes, where Q = NUM_TCU_BLOCKS =
ISSUE_WIDTH = warps-per-TB) consume **the same uop index in the same
cycle**. This is a property of dispatch, not of memory:

- Q parallel issue slots fire Q uops (one per warp) in the same cycle.
- "Block `b` ready to advance" means: block `b`'s `A_w[m, k]` is in its
  A buffer, the shared B buffer holds `B[k, n]`, and the FU output is
  free.
- If any active block is not ready, none advances.

This invariant interacts with two finite TCU resources to produce the
documented `NUM_WARPS > ISSUE_WIDTH` deadlock:

1. **Per-warp FU lock** — a WGMMA holds the TCU's shared FU for the
   duration of its uop sequence (`fu_lock` on first uop, `fu_unlock` on
   last; see
   [tensor_unit.cpp:1420-1422](../../sim/simx/tcu/tensor_unit.cpp#L1420)).
2. **Shared B buffer** — only one B tile is held at a time for the TB.

When NUM_WARPS > ISSUE_WIDTH and two CTAs interleave onto the Q
dispatch slots, the lock-step gate may indefinitely hold both CTAs in
a state where each one's required B tile is mutually exclusive with
the currently buffered tile. The proposed design preserves this
lock-step invariant — and therefore preserves the deadlock under the
documented config.

**Open** (§4.8): with the proposed shared B buffer in place of
per-block B copies, the *exact* deadlock mechanism shifts. Need to
reproduce in Phase D before declaring deadlock-equivalent with RTL
under the new structure.

### 4.5 Functional + timing coupled on every channel

The functional value of an A or B element is the bytes returned in the
`LmemRsp::data` packet at the cycle the response arrives. The cycle is
determined by LMEM port arbitration and the SMEM bank-row read pipeline.
There is no second clock for "synthetic timing" and no second data
source for "functional reads." A single packet carries both, and the
TCU's structured buffers populate from it.

This is Rule 2 by construction.

### 4.6 What changes vs the current Vortex tile-buffer design

| Aspect | Current Vortex (RTL & half-built SimX) | Proposed |
|---|---|---|
| Tile persistence | TCU per-block full slot (`(M_STEPS × K_STEPS + K_STEPS × N_STEPS) × bank-row` per block) | TCU minimal structured (`M_STEPS × bank-row` per block + `1 × bank-row` shared) |
| TCU state per active TB | `~6 KB` (Q=4, NRC=8, NT=32, fp16) | `~1.1 KB` (~5× smaller) |
| Slot allocation | FSM per block; descriptor-keyed full-tile fetch up-front | Per-block A buffer (re)alloc on k-stripe entry; shared B buffer (re)alloc on `(k, n)` change |
| Slot eviction | Implicit on first uop of different descriptor | A: overwrite on k-transition; B: overwrite on `(k, n)` change |
| B-fetch SMEM cost | `Q × K_STEPS × N_STEPS` xfers (one per block, no sharing) | `K_STEPS × N_STEPS` xfers (one per `(k, n)` for the TB) |
| First-uop latency | Whole tile must fetch (`Q × (M_STEPS × K_STEPS + K_STEPS × N_STEPS)` bank-rows) before u0 emits | Only `Q + 1` bank-rows before u0 emits — **lower** |
| Failure mode under contention | Per-block slot busy → stall until WGMMA done | Shared B buffer overwritten on dispatch alternation; deadlock under same documented config |
| Cross-WGMMA reuse | None (slot wipes on different desc) | If next WGMMA matches descriptors, A and B buffers stay valid |

**Per-WGMMA SMEM bytes moved (Q = 4, NRC = 8, NT = 32):**

- Current RTL: `Q × (M_STEPS × K_STEPS + K_STEPS × N_STEPS) × bank-row
  = 4 × 12 × 128 = 6 KB`.
- Proposed: `(Q × M_STEPS × K_STEPS + K_STEPS × N_STEPS) × bank-row
  = (16 + 8) × 128 = 3 KB`.

**Halved**, exactly because the shared B buffer fetches each B row once
per TB instead of once per block. (The reduction is *not* from any
cache-side coalescing — there is no LMEM cache.)

### 4.7 Why this is also a good RTL design

The user noted this design will eventually land in RTL. The RTL gains
match the SimX gains:

- **SRAM area saved.** Per-block tile-buffer SRAM (kilobytes × Q blocks
  × N cores) is replaced by per-block `M_STEPS × bank-row` registers
  plus one shared `bank-row` register. ≈ 5× smaller per TB.
- **Slot-allocation FSM simplified.** The
  `SEND_IDLE → FETCH_A → FETCH_B → FETCH_META → IDLE` FSM in
  [VX_tcu_tbuf_fetch.sv:227-241](../../hw/rtl/tcu/VX_tcu_tbuf_fetch.sv#L227)
  collapses to a `(re-fill A on k-step, re-fill B on (k, n)-step)` loop.
- **Single B-fetch path** for the TB — Q-fold fewer LMEM-port cycles
  spent on B in the common shared-descriptor case.
- **Cross-WGMMA reuse** comes for free if the next WGMMA's descriptors
  match.

Cost: needs explicit cross-block sharing structure for the B buffer
(routing one fetched row to all Q blocks). Mitigated by the lock-step Q
invariant — all Q blocks consume B simultaneously, so the sharing fan-out
is structural, not arbitrated.

### 4.8 Tradeoffs and design-open questions

- **No LMEM cache, no coalescing.** Every reuse claim in this section
  is predicated on the TCU's own structured buffers — never on
  cache-side merging or hit-under-miss. Same-cycle requests to the same
  bank-row from different blocks each consume their own LMEM-port
  cycle unless the TCU itself dedups them (which is exactly what the
  shared B buffer does for B; A is per-block by construction).
- **First-uop latency is LOWER than RTL today**, not higher. Proposed
  design fetches only `Q + 1` bank-rows for u0 (each warp's `A_w[0,0]`
  + shared `B[0,0]`). RTL today pre-fetches the whole tile (~`Q ×
  K_STEPS × N_STEPS` rows for B alone) before u0 emits.
- **First-WGMMA cold cost**: unavoidable on the very first WGMMA of a
  TB. Subsequent WGMMAs benefit from preserved buffers if descriptors
  match; otherwise the new tile's bank-rows replace the old at no
  incremental cost over the design's intrinsic fetch count.
- **B-buffer thrashing under multi-CTA interleaving.** Two CTAs with
  different B descriptors fighting for the shared B buffer overwrite
  it on every dispatch alternation. Acceptable for `NUM_WARPS =
  ISSUE_WIDTH` (one CTA at a time); reproduces the deadlock under the
  documented `NUM_WARPS > ISSUE_WIDTH` config (§4.4).
- **N-direction warp tiling** (warps share A descriptor, differ in B):
  the symmetric optimization is a shared **A** buffer + per-block
  **B** buffer. Out of scope for this version; the M-direction case
  (the production-common one) is the design target.
- **Lock-step Q gate granularity.** Per-uop, not per-WGMMA. Open:
  should the gate also fire on first-uop arrival (Q warps must arrive
  in the same cycle), or only on per-uop readiness? RTL implication.
  SimX adopts per-uop only; revisit if RTL-side requires arrival-time
  gating.
- **Sparse SS metadata** streams via a third per-block buffered path
  (similar to A). Same shape; implementation deferred.

### 4.9 Sample execution timeline

To make the optimization visible, walk through one TB's WGMMA execution
under the proposed design.

**Configuration:**

- `Q = 4` (warps in lock-step), `NT = 32` threads/warp.
- `NRC = 8` ⇒ `M_STEPS = 2`, `K_STEPS = 2`, `N_STEPS = 4`, total 16 uops.
- M-direction warp tiling (typical CTA-level GEMM): each warp has a
  unique A descriptor (`A_w` for `w ∈ {0..3}`); all four warps share one
  B descriptor (`B`). This is the production-common case; pure-N tiling
  is the symmetric case (shared A, unique B) and behaves the mirror.
- Iteration order (§4.3): k-outer / n-middle / m-inner. Uop sequence is
  thus `(k=0, n=0, m=0), (k=0, n=0, m=1), (k=0, n=1, m=0), …, (k=1, n=3, m=1)`.
- Block granularity (§4.2): one bank-row per LmemReq/LmemRsp.
  - **A blocks** (per warp): `A_w[m, k]` for `m ∈ {0,1}`, `k ∈ {0,1}` → 4
    distinct A blocks per warp × Q = 16 distinct A lines across the TB.
  - **B blocks** (shared): `B[k, n]` for `k ∈ {0,1}`, `n ∈ {0..3}` → 8
    distinct B lines.
- Latency model (each event = 1 cycle as stipulated):
  - **Buffer hit** (row in TCU's A or shared B buffer) = 1 cycle.
  - **Buffer miss** → 1 LmemReq → SMEM bank-row read → response = 3
    cycles end-to-end per request.
  - One MMA per cycle once operands are ready; multiple LmemReqs may
    be in flight at once via channel pending-count.
  - **Idealized assumption for this trace**: bank-row addresses do not
    bank-conflict, so multiple in-flight LmemReqs are pipelined by the
    LMEM port at full throughput. Real workloads must place A and B
    rows on non-conflicting banks; otherwise add port-serialization
    cycles (§4.8).
- **B-side dedup is at the TCU**, not at LMEM. When all Q blocks need
  the same B row (production-common shared-B-descriptor case), the TCU
  shared B buffer is consulted first; on miss, **one** LmemReq is
  issued to populate it; on subsequent uops, all Q blocks read from
  the buffer with no LmemReq. **LMEM has no cache, no MSHR, no
  coalescing** — it would otherwise see Q identical requests as Q
  separate SMEM accesses.
- **SMEM transfer width**: one transfer = `NT × 4 bytes` = **128 B**
  (NT = 32, 4 B/word; matches the LMEM port = `NUM_BANKS × 4 B` = one
  bank-row).

**Per-uop timeline.** "Cyc" is the cycle range the uop spans from operand
issue to MMA completion under lock-step (slowest lane of the Q gates the
group). The **A xfers** and **B xfers** columns count *underlying SMEM
transfers* triggered by the uop, expressed as a formula in `Q` and the
loop bounds. **A reuse** / **B reuse** indicate whether *this uop's*
A and B rows are already in the TCU's per-block A buffer / shared B
buffer when issued (Y = buffer hit, N = buffer miss → LmemReq).
"Resident after" lists the contents of the buffers at the end of the
uop (omit rows for unchanged entries).

| uop | cyc   | (k,n,m) | A xfers                  | B xfers                 | A reuse | B reuse | Resident A after        | Resident B after          | MMA |
|-----|-------|---------|--------------------------|-------------------------|---------|---------|-------------------------|---------------------------|-----|
|   0 |  1–4  | (0,0,0) | **Q** (cold `A_w[0,0]`)  | **1** (cold `B[0,0]`)   | N       | N       | `A_w[0,0]` (×Q)         | `B[0,0]`                  |  u0 |
|   1 |  5–7  | (0,0,1) | **Q** (cold `A_w[1,0]`)  | **0** (`B[0,0]` hit)    | N       | Y       | + `A_w[1,0]`            | (same)                    |  u1 |
|   2 |  8–10 | (0,1,0) | **0** (`A_w[0,0]` hit)   | **1** (cold `B[0,1]`)   | Y       | N       | (same)                  | + `B[0,1]`                |  u2 |
|   3 |  11   | (0,1,1) | **0** (`A_w[1,0]` hit)   | **0** (`B[0,1]` hit)    | Y       | Y       | (same)                  | (same)                    |  u3 |
|   4 |  12–14| (0,2,0) | **0** (`A_w[0,0]` hit)   | **1** (cold `B[0,2]`)   | Y       | N       | (same)                  | + `B[0,2]`                |  u4 |
|   5 |  15   | (0,2,1) | **0** (`A_w[1,0]` hit)   | **0** (`B[0,2]` hit)    | Y       | Y       | (same)                  | (same)                    |  u5 |
|   6 |  16–18| (0,3,0) | **0** (`A_w[0,0]` hit)   | **1** (cold `B[0,3]`)   | Y       | N       | (same)                  | + `B[0,3]`                |  u6 |
|   7 |  19   | (0,3,1) | **0** (`A_w[1,0]` hit)   | **0** (`B[0,3]` hit)    | Y       | Y       | `A_w[0..1, 0]`          | `B[0, 0..3]`              |  u7 |
|   8 |  20–22| (1,0,0) | **Q** (cold `A_w[0,1]`)  | **1** (cold `B[1,0]`)   | N       | N       | + `A_w[0,1]`            | + `B[1,0]`                |  u8 |
|   9 |  23–25| (1,0,1) | **Q** (cold `A_w[1,1]`)  | **0** (`B[1,0]` hit)    | N       | Y       | + `A_w[1,1]`            | (same)                    |  u9 |
|  10 |  26–28| (1,1,0) | **0** (`A_w[0,1]` hit)   | **1** (cold `B[1,1]`)   | Y       | N       | (same)                  | + `B[1,1]`                | u10 |
|  11 |  29   | (1,1,1) | **0** (`A_w[1,1]` hit)   | **0** (`B[1,1]` hit)    | Y       | Y       | (same)                  | (same)                    | u11 |
|  12 |  30–32| (1,2,0) | **0** (`A_w[0,1]` hit)   | **1** (cold `B[1,2]`)   | Y       | N       | (same)                  | + `B[1,2]`                | u12 |
|  13 |  33   | (1,2,1) | **0** (`A_w[1,1]` hit)   | **0** (`B[1,2]` hit)    | Y       | Y       | (same)                  | (same)                    | u13 |
|  14 |  34–36| (1,3,0) | **0** (`A_w[0,1]` hit)   | **1** (cold `B[1,3]`)   | Y       | N       | (same)                  | + `B[1,3]`                | u14 |
|  15 |  37   | (1,3,1) | **0** (`A_w[1,1]` hit)   | **0** (`B[1,3]` hit)    | Y       | Y       | `A_w[0..1, 0..1]` (×Q)  | `B[0..1, 0..3]`           | u15 |
| **Σ** | **37 cy** | —    | **Q × M_STEPS × K_STEPS = 4 Q** | **K_STEPS × N_STEPS = 8** | — | — | — | — | — |

**Total SMEM bytes moved across this WGMMA** (one transfer = `NT × 4 B`
= 128 B):

- A: `(Q × M_STEPS × K_STEPS) × NT × 4 = 4 Q × 128 B`
  = **2 048 B** at Q = 4.
- B: `(K_STEPS × N_STEPS) × NT × 4 = 8 × 128 B` = **1 024 B**.
- **Total: ≈ 3 KB** moved over LMEM for the entire 16-uop WGMMA across
  all Q warps.

Compare RTL today (per-block tile buffer, no cross-block B share):
A and B each replicated Q times → `Q × (M_STEPS × K_STEPS + K_STEPS ×
N_STEPS) × NT × 4 = 4 × 12 × 128 B = 6 KB` moved. The proposed design
halves SMEM traffic on this workload, exactly because B is fetched once
per line for the whole TB instead of once per block.

**What the table shows.**

- **Cold start (uop 0):** Q distinct A misses (one per warp's A buffer)
  + 1 shared B miss. Lock-step gate waits for the `Q + 1` fills before
  u0 advances. Under the idealized port-pipelining assumption above,
  this is one 3-cy miss latency; with serialization (§4.8) it grows to
  `Q + 1 + miss_lat` cycles.
- **B reuse via the TCU's shared B buffer (uops 1, 3, 5, 7, …):** all
  Q blocks share the B descriptor, so the shared B buffer is
  populated once per `(k, n)` and read by all Q blocks. The B-row
  miss happens once for the TB — *not* Q times — because the **TCU
  itself** dedups, not because LMEM coalesces.
- **A reuse via per-block A buffers (uops 2–7 within k=0; uops 10–15
  within k=1):** once `A_w[0, k]` and `A_w[1, k]` are resident in
  each block's A buffer (after the first two uops of the k-stripe),
  the inner sweep over `n ∈ {0..3} × m ∈ {0, 1}` hits on A every
  cycle.
- **Steady-state cost per uop within a k-stripe:** alternating
  `(B miss)` / `(full hit)` → 3-cy and 1-cy uops respectively. Six
  uops per k-stripe after the warm-up: 3 × 3 cy + 3 × 1 cy = 12 cy.
- **K-stripe transition (uop 8):** the per-block A buffers are
  overwritten by `A_w[*, 1]`; the shared B buffer is overwritten by
  `B[1, 0]`. The pattern restarts cold.

**Total wall time for the TB (this trace, idealized):** 37 cycles —
matches the per-row sum. Real port serialization (§4.8) pushes the cold
rows up; structurally the cycle counts in the "Y/Y" hit rows are
unaffected.

**Comparison with the current per-block-tile-buffer design (RTL today /
half-built SimX) for the same workload:**

| Design | SMEM bytes fetched | Cold-start latency | Per-uop latency in steady state |
|---|---|---|---|
| RTL today (per-block tile buffer, no shared B) | A: `Q × M_STEPS × K_STEPS`; B: `Q × K_STEPS × N_STEPS` (Q B copies, one per block) | Whole tile prefetched before u0 emits | 1 cy (resident) |
| Current SimX (descriptor-keyed B cache) | A: `Q × M_STEPS × K_STEPS`; B: `K_STEPS × N_STEPS` (B already shared) | ~RTL minus the (Q-1) B copies | 1 cy |
| **Proposed (this doc)** | **A: `Q × M_STEPS × K_STEPS`; B: `K_STEPS × N_STEPS`** (all unique rows fetched once each) | First-uop ≈ `Q + 1` bank-rows + miss_lat — **lower** than RTL | 1 cy hit / 3 cy miss; per-stripe amortized **2 cy/uop** |

The proposed design moves the **same total bytes** as the current SimX
B-shared variant — but does so with TCU storage that is `~5× smaller`
(§4.6) and rules out the per-block B copies of RTL today. First-uop
latency is lower because only `Q + 1` rows must land before u0 emits,
not the whole tile.

### 4.10 Request-scheduling refinements visible from the timeline

The §4.9 trace exposes four refinements at the *request-scheduling* layer.
None of them grows the TCU buffer beyond §4.1's structured shape; they
change only *when* LmemReqs are issued and how many in-flight rows the
buffer holds at once.

#### 4.10.1 TCU buffer sizing: `M_STEPS × Q + 1` rows (or `+2` with B ping-pong) per active TB

Reading the "Resident A/B after" columns: at any cycle inside a k-stripe
(uops 2–7 within k=0; uops 10–15 within k=1) the TCU must hold

- `M_STEPS × Q = 2 Q` A rows (`A_w[0, k]` and `A_w[1, k]` for each warp `w`,
  one row per block in its per-block A buffer),
- `1` B row (the current `B[k, n]` in the shared B buffer; `2` with
  ping-pong).

Anything less thrashes inside the stripe. Anything more is unused. This
fixes the **minimum sizing** of the TCU's A-buffer-per-block and shared
B buffer: at Q = 4, NT = 32 that is `9 × 128 B = 1152 B` (`1280 B` with
ping-pong) per active TB. There is no LMEM cache; the entire reuse
working set lives in these TCU buffers.

#### 4.10.2 Burst-fetch the `2 Q` A rows at k-stripe entry

Today's natural request order in the table issues `Q` A misses on uop 0,
another `Q` on uop 1 — staggered. Both batches are needed before the
inner `(n, m)` sweep over the same k-stripe can hit on A. Issuing all
`2 Q` A LmemReqs in cycle 1 of the k-stripe — driven by the uop generator
emitting an explicit "k-prologue" pseudo-uop, or by `TcuTbuf` looking
ahead to the next two uops — lets the LMEM port serve them as one
contiguous bank-row burst. Under the §4.9 idealized assumption the burst
drains in 1 + miss_lat cycles total instead of 2 separate uop windows;
under realistic single-port serialization the burst drains in `2 Q +
miss_lat` cycles either way, so the saving is the lock-step issue stall
uop 1 currently incurs (~1 cycle per k-stripe).

This is a scheduling change; channel shapes and total SMEM bytes moved
are unchanged.

#### 4.10.3 Prefetch next k-stripe during the k-transition wait (RTL-relevant)

Across the k boundary, the C accumulator register read by the first uop
of `k+1` was last written by an earlier uop of `k` (same `r = n × M_STEPS
+ m`). In **RTL**, the FEDP pipeline depth `L` (4–6 cycles) ⇒ the first
uop of the new k-stripe must wait `~L` cycles for the writeback to
retire — a window in which the TCU has no operand-emit work. **That
window is enough to issue the `2 Q` A LmemReqs (and the first B row)
for the next k-stripe**, hiding most of the next-stripe cold cost.

In **SimX** under §4.9's `1-cycle MMA` model, the RAW window is
effectively zero — this optimization is **moot for SimX timing** but
worth implementing in RTL form. SimX timing fidelity is unaffected
either way (§7's ~10 % bound).

#### 4.10.4 B ping-pong (next-B prefetch within a k-stripe)

Within a k-stripe, B is consumed sequentially: `B[k, 0]` for 2 uops
(m = 0, m = 1), then `B[k, 1]` for 2, etc. With miss_lat = 3 cy and a
2-uop consumption window of 1 cy each (steady-state hit MMA), prefetching
`B[k, n+1]` while `B[k, n]` is being consumed **partially** hides the
miss: the prefetch starts at the m=0 uop of `(k, n)`, drains in 3 cy,
and the next-B is ready by 1 cy after the m=1 uop completes — so the
miss on the next-B uop is reduced from 3 cy to 1 cy (residual stall),
not eliminated.

Net per next-B: 3 cy → 1 cy (saves 2 cy). Across N_STEPS-1 next-B's per
k-stripe × K_STEPS stripes = `(N_STEPS - 1) × K_STEPS` such savings.

Ping-pong **does** grow the shared B buffer to 2 rows (current + next).
The TCU adds one bank-row of state plus one in-flight LmemReq.

#### Combined effect on the §4.9 timeline

| Refinement | Cold cycles eliminated | Per-WGMMA savings (Q=4, NRC=8) |
|---|---|---|
| 4.10.2 — burst A at k-stripe entry      | issue stall on uop 1 of each k-stripe | ~1 cy × K_STEPS = **2 cy** |
| 4.10.3 — k-transition prefetch (RTL)    | next-stripe cold A pair                | ~`(2 cy)` per k-transition × (K_STEPS-1) = **2 cy in RTL, 0 in SimX** |
| 4.10.4 — B ping-pong                    | next-B miss → 1-cy residual stall     | (3-1) cy × `(N_STEPS-1) × K_STEPS` = `2 × 6` = **12 cy** |

Combined SimX-realistic savings on this trace: **~14 cy** vs the §4.9
baseline of 37 cy → **~23 cy** for the 16-uop WGMMA. RTL adds a further
~2 cy of k-transition hiding → **~21 cy** in RTL form. Ideal lower bound:
16 cy. The residual gap is the first-WGMMA cold start.

**These refinements are additive on top of the core design.** Phase B
(§6) can land without them; later phases may opt in as discrete commits.

---

## 5. Architecture (SimX implementation)

This section describes how §4 lands in SimX specifically.

### 5.1 Module hierarchy

```
TcuUnit               (FuncUnit; owns Q TcuCores; lock-step probe; owns TcuSharedB)
├── TcuSharedB        (shared B buffer for the TB: 1 row resident, +1 with ping-pong)
│     ├─ lmem_req_out  → LMEM (B-side fetches; one per (k, n) change)
│     └─ lmem_rsp_in   ← LMEM
└── TcuCore[b]        (per-block; one per warp lane)
    ├── TcuTbufA      (per-block A buffer: M_STEPS rows resident)
    │     ├─ alloc_in       ← from TcuCore on WGMMA arrival / k-step entry
    │     ├─ lmem_req_out   → LMEM (A-side fetches; M_STEPS per k-stripe)
    │     ├─ lmem_rsp_in    ← LMEM
    │     └─ gather_out     → TcuMma: per-uop A operand bundle
    └── TcuMma        (consumes A from TcuTbufA + B from TcuSharedB; emits result)
```

`LmemReq`/`LmemRsp` reuse the data-carrying `MemReq`/`MemRsp` from
[simx_v3 §4.3](simx_v3_proposal.md#43-memory-model); the `data` field
carries one bank-row (`NUM_BANKS × 4 B`).

### 5.2 TcuTbufA (per-block A buffer)

```cpp
class TcuTbufA : public SimObject<TcuTbufA> {
public:
  SimChannel<TbufAllocReq>     alloc_in;
  SimChannel<LmemReq>          lmem_req_out;
  SimChannel<LmemRsp>          lmem_rsp_in;
  SimChannel<TbufOperandsA>    gather_out;

  void tick() override;

private:
  // Active k-stripe's M_STEPS rows (one per m-index for this warp).
  std::array<std::shared_ptr<mem_block_t>, M_STEPS> a_rows_;
  uint32_t cur_k_         = ~0u;
  bool     valid_         = false;
  uint32_t cached_desc_a_ = 0;
  bool     a_from_smem_   = false;

  // FSM: wait for k-stripe alloc → issue M_STEPS A LmemReqs → drain responses → gather per-uop.
  enum class Phase { IDLE, FETCHING, READY };
  Phase    phase_         = Phase::IDLE;
  uint32_t reqs_in_flight_= 0;
};
```

State per block: descriptor cache (~16 B) + `M_STEPS` `shared_ptr` rows
(2 × 8 B = 16 B; the rows themselves are channel-borne and shared via
ref count) + small FSM. Per-block heap usage during a k-stripe:
`M_STEPS × NT × 4 B = 256 B` of bank-row data. Q instances per `TcuUnit`.

### 5.3 TcuSharedB (shared B buffer for the TB)

```cpp
class TcuSharedB : public SimObject<TcuSharedB> {
public:
  SimChannel<TbufAllocReq>     alloc_in;
  SimChannel<LmemReq>          lmem_req_out;
  SimChannel<LmemRsp>          lmem_rsp_in;
  SimChannel<TbufOperandsB>    gather_out;     // broadcast to all Q TcuMma

  void tick() override;

private:
  // Current B[k, n] row (and optional next-row slot for ping-pong).
  std::shared_ptr<mem_block_t> b_row_curr_;
  std::shared_ptr<mem_block_t> b_row_next_;    // ping-pong, optional
  uint32_t cur_kn_        = ~0u;
  uint32_t cached_desc_b_ = 0;
  bool     valid_         = false;
};
```

Single instance per `TcuUnit`. Heap during execution: `1 × NT × 4 B = 128 B`
(or 256 B with ping-pong). All Q `TcuMma` instances read the same `b_row_curr_`
through `gather_out` — one shared bank-row, no per-block copies.

### 5.4 TcuUnit lock-step probe

```cpp
void TcuUnit::tick() {
  uint32_t active = 0;
  for (uint32_t b = 0; b < Q; ++b)
    if (!input_[b].empty() && is_wgmma(input_[b].peek()))
      active |= (1u << b);

  if (active != 0) {
    uint32_t ready = 0;
    for (uint32_t b = 0; b < Q; ++b)
      if ((active >> b) & 1u
          && cores_[b]->tbufa_ready()           // per-block A row resident
          && shared_b_.row_ready(cores_[b]->kn()))  // shared B row resident
        ready |= (1u << b);
    if (ready != active) { ++perf_.tbuf_stalls; return; }
  }

  // WMMA (no Q-coupling) advances per-block.
  for (uint32_t b = 0; b < Q; ++b) cores_[b]->tick();
}
```

`tbufa_ready()` returns true when the block's `TcuTbufA` has the current
uop's `A_w[m, k]` row resident. `shared_b_.row_ready(kn)` returns true
when `TcuSharedB::b_row_curr_` matches the current `(k, n)`. Bitmask
AND, O(Q), no descriptor compare per cycle.

### 5.5 What gets deleted

- `Impl::tbuf_state_` (unit-global) — replaced by per-block `TcuTbufA`
  + one `TcuSharedB`.
- `Impl::ready_cycle` and any synthetic-delay arithmetic — Rule 2.
- `load_lmem_word`'s abort path — replaced by gather from `LmemRsp::data`
  (Rule 1).
- The "shared B cache across warps" logic in
  [tensor_unit.cpp:959-964](../../sim/simx/tcu/tensor_unit.cpp#L959) —
  superseded by the structural `TcuSharedB`, which makes the sharing
  explicit at module level instead of bolted onto a per-warp struct.

---

## 6. Implementation plan

Each phase independently buildable; CSV trace diff is the gate.

### Phase A — Iteration-order fix (1 day)

Swap k/n in [tensor_unit.cpp:1394-1399](../../sim/simx/tcu/tensor_unit.cpp#L1394)
to k-outer / n-middle / m-inner. Update the comment to give the SimX
rationale (§4.3).

### Phase B — Per-block `TcuTbufA` + shared `TcuSharedB` SimObjects (1 week)

| # | Action |
|---|--------|
| B.1 | Create `sim/simx/tcu/tcu_tbuf_a.{h,cpp}`, `tcu_shared_b.{h,cpp}`, `tcu_core.{h,cpp}`. Per-block `TcuTbufA`; one `TcuSharedB` per `TcuUnit`. |
| B.2 | Delete the unit-global `Impl::tbuf_state_`. Replace with the new module instances. |
| B.3 | Implement A-side FSM (alloc on k-stripe entry → issue `M_STEPS` LmemReqs → drain → READY) and B-side FSM (alloc on `(k,n)` change → issue 1 LmemReq → drain → READY, +1 in-flight slot for ping-pong). |
| B.4 | Wire `lmem_req_out` / `lmem_rsp_in` channels to LMEM (no cache layer assumed; LMEM is banked SMEM with one-bank-row-per-cycle port arbitration). |

**Validation:** dense `sgemm_tcu_wg` (RS, the Makefile default) and sparse
`sgemm_tcu_wg_sp` functional results match RTLsim; abort paths gone; no
`core_->mem_read` calls anywhere in the WGMMA path. Re-run with
`-DWGMMA_RS` removed from the test Makefile to cover SS mode.

### Phase C — Functional + timing on the channels (3-4 days)

| # | Action |
|---|--------|
| C.1 | `TcuTbufA::tick()` and `TcuSharedB::tick()` consume `LmemRsp::data` — gather operands from response bytes (Rule 2). |
| C.2 | Delete any residual `ready_cycle` / `synth_fetch_delay` arithmetic. Stalls come from `full()`/`empty()` only. |
| C.3 | `tbuf_cache_hits` reports descriptor-cache match in `TcuTbufA` / `TcuSharedB`; LMEM port-cycle counters report through the existing LMEM SimObject. |

**Validation:** `sgemm_tcu_wg` cycles within ~10 % of RTLsim; CSV
functional trace byte-identical.

### Phase D — Q-warp lock-step probe (2 days)

| # | Action |
|---|--------|
| D.1 | Implement `TcuUnit::tick()` per §5.4 (probes both per-block `TcuTbufA::tbufa_ready()` and `TcuSharedB::row_ready()`). WMMA bypasses the gate. |
| D.2 | Regression: WGMMA with NUM_WARPS = 2 × ISSUE_WIDTH must hang (timeout-asserted), matching RTLsim. |

### Phase E — Sparse SS streaming (follow-up)

Wire FETCH_META as a third request stream alongside A and B. Same shape;
defer until dense is solid.

---

## 7. Validation

| Signal | Pass criterion |
|--------|----------------|
| `sgemm_tcu_wg` functional trace vs RTLsim | byte-identical |
| `sgemm_tcu_wg_sp` functional trace vs RTLsim | byte-identical (sparse path) |
| `sgemm_tcu_wg` cycles vs RTLsim | within ~10 % |
| Persistent TCU bytes per active TB | ≤ ~1.2 KB (= `Q × M_STEPS × 128 B + 256 B`, §4.10.1) |
| `core_->mem_read` calls in TCU path | zero (grep) |
| Hang under NUM_WARPS > ISSUE_WIDTH | reproduces |

Both `sgemm_tcu_wg` and `sgemm_tcu_wg_sp` default to `-DWGMMA_RS`
(RS mode: A from registers, B from SMEM). SS-mode coverage requires
editing the test Makefile to drop `-DWGMMA_RS` and rebuilding the kernel.

Reproduce per phase from `build_test32/`:

```bash
# Dense WGMMA, RS mode (default)
ci/blackbox.sh --driver=rtlsim --app=sgemm_tcu_wg --perf=1 --debug=3 --log=run_rtlsim.log
ci/blackbox.sh --driver=simx   --app=sgemm_tcu_wg --perf=1 --debug=3 --log=run_simx.log

# Sparse WGMMA
ci/blackbox.sh --driver=simx   --app=sgemm_tcu_wg_sp --perf=1 --debug=3 --log=run_simx_sp.log
```

---

## 8. Risks

- **R1. LMEM `MemRsp` must carry data before Phase B lands.** Sequence
  Phase B after simx_v3 §5.4–§5.5 (`Memory` carries line data through the
  LMEM channels). No transitional backdoor — Rule 1 admits no exception.
- **R2. LMEM-port serialization.** All LmemReqs serialize through one
  bank-row/cycle port; cold-start under realistic serialization is
  `(Q + 1) + miss_lat` cycles, not `miss_lat`. Document in §4.9
  configuration; do not understate cold latency in cycle-fidelity claims.
- **R3. Channel ordering.** Multiple in-flight `LmemReq`s must complete
  in FIFO order to drive the consumer FSMs correctly. Verify with a
  directed test before C.3.
- **R4. N-direction warp tiling unsupported in this version.** The
  shared B buffer assumes M-tiling. If a workload uses N-tiling (warps
  share A), the shared B buffer thrashes uselessly — A-side fetches still
  duplicate. §4.8 documents; out of scope.
- **R5. Deadlock-equivalence under new structure.** §4.4 open question —
  verify reproduction with the new shared-B-buffer mechanism in Phase D.
- **R6. Lock-step gate granularity** (§4.8 open question). Adopt per-uop;
  revisit if RTL-side dictates arrival-time gating.

---

## 9. Out of scope

- Cross-warp Q-coalescing (§F2).
- Double-buffering / multi-slot tile buffers (§F4).
- Backdoor `core_->mem_read` (Rule 1).
- Synthetic timing alongside any functional path (Rule 2).
- Cycle-accurate RTL parity beyond ~10 %.

---

## 10. References

- [hw/rtl/tcu/VX_tcu_unit.sv](../../hw/rtl/tcu/VX_tcu_unit.sv) — RTL Q-block dispatch.
- [hw/rtl/tcu/VX_tcu_tbuf_fetch.sv](../../hw/rtl/tcu/VX_tcu_tbuf_fetch.sv) — RTL streaming FSM (model for §5.2 FSM, not its storage).
- [hw/rtl/tcu/VX_tcu_uops.sv:128](../../hw/rtl/tcu/VX_tcu_uops.sv#L128) — k-outer iteration.
- [sim/simx/tcu/tensor_unit.cpp](../../sim/simx/tcu/tensor_unit.cpp) — current implementation being replaced.
- [docs/proposals/simx_v3_proposal.md](simx_v3_proposal.md) — host architecture refactor.
- [docs/proposals/simx_rtl_perf_divergence.md](simx_rtl_perf_divergence.md) — cycle-count validation methodology.
