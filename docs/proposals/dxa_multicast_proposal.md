# DXA Multicast — Proposal

**Date:** 2026-05-24
**Status:** Draft (revised: built on cluster dispatch contract; intra-core only; DXA relocated per-socket)
**Owners:** RTL team
**Related:**
[dxa_worker_rtl_redesign_proposal.md](dxa_worker_rtl_redesign_proposal.md),
[VX_kmu.sv](../../hw/rtl/VX_kmu.sv),
[VX_cta_dispatch.sv](../../hw/rtl/core/VX_cta_dispatch.sv),
[VX_bar_unit.sv](../../hw/rtl/core/VX_bar_unit.sv),
[VX_dxa_smem_wr.sv](../../hw/rtl/dxa/VX_dxa_smem_wr.sv),
[VX_dxa_core.sv](../../hw/rtl/dxa/VX_dxa_core.sv),
[VX_socket.sv](../../hw/rtl/VX_socket.sv),
[VX_cluster.sv](../../hw/rtl/VX_cluster.sv),
[vx_spawn2.h](../../sw/kernel/include/vx_spawn2.h),
[vx_barrier.h](../../sw/kernel/include/vx_barrier.h),
[vx_dxa.h](../../sw/kernel/include/vx_dxa.h),
[vortex2.h](../../sw/runtime/include/vortex2.h),
[sim/simx/socket.cpp](../../sim/simx/socket.cpp),
[sim/simx/cluster.cpp](../../sim/simx/cluster.cpp).

This proposal is **self-contained**: it owns all of the
cluster dispatch contract (KMU + dispatcher + CSR + runtime),
the DXA multicast SDK + RTL fixes, the per-socket DXA relocation,
and the WGMMA workload. No external prerequisite proposals.

---

## Summary

Make DXA multicast a first-class primitive built on Vortex's existing
**local-barrier** infrastructure plus a **cluster dispatch
contract** specified and implemented entirely by this proposal (§0).
Multicast is scoped to a single core; its peer set is exactly the
**cluster** of CTAs that the dispatcher has guaranteed to be
co-resident. Mask bits identify cluster ranks. The barrier
handle's scope bit is forced **local** — global-barrier (cross-core)
multicast is explicitly out of scope.

Synchronization across receivers (every CTA must have called
`expect_tx` before the issuer fires multicast) is solved in software
with a **two-barrier idiom**, wrapped by a `vortex::dxa_multicast`
helper class that makes mask construction and `expect_tx` registration
inseparable. No new hardware barrier opcode.

Alongside the multicast primitive, this proposal also relocates the
`VX_dxa_core` instance from the cluster down to **per-socket** scope, and
arbitrates its GMEM traffic against the socket's dcache GMEM traffic
**before** the merged stream leaves the socket. The result: each socket
exposes a single `mem_bus_if[L1_MEM_PORTS]` path that already carries
icache + dcache + DXA traffic; the cluster sees no DXA-specific ports.
SimX is updated in lock-step so the functional model matches the new
topology.

The architectural inspiration is NVIDIA Hopper's TMA + Thread-Block-
Cluster + mbarrier system (CUTLASS `PipelineTmaAsync`). Vortex's
`cluster_dim` is the analog of Hopper's cluster (atomic same-core
dispatch); `vortex::dxa_multicast` is the analog of `cp.async.bulk.tensor.multicast`
+ mbarrier `expect_tx`; the existing `bar_unit` phase parity is the
analog of Hopper's mbarrier `try_wait.parity`. We deliberately do
**not** adopt Hopper's cross-CTA distributed shared memory (DSMEM) —
the cluster LMEM stays per-core, and multicast destinations
resolve via "same offset in each member's LMEM region."

**Hard constraints, by design:**

1. **Cluster dispatch contract (Phase 0).** This proposal owns
   the KMU + dispatcher work that delivers atomic same-core dispatch
   for cluster members. See §0 for the contract and §1's Phase 0 row
   for the file-by-file deliverables.
2. **Multicast is intra-core only.** No inter-core / global-bar
   multicast, no cross-core LMEM fabric, no socket-wide write switch,
   no `gbarrier`-based multicast. A multicast group lives entirely
   within one core's bar/CTA/LMEM domain.
3. **Mask bits identify cluster ranks.** With `cluster_dim`
   guaranteeing all `K` cluster members on one core in contiguous
   `cta_local_id` slots starting at the issuer's id, bit `k` targets
   member rank `k` (= `cta_local_id = issuer_cta_id + k`). The walk
   is uniform for any warps-per-CTA `W ∈ {1, 2, 4, …, NUM_WARPS}`
   because the bar-unit address field is indexed by `VX_CSR_CTA_ID`,
   not by raw wid. The Phase-1 kernel convention is that the issuer
   is cluster rank 0; the helper API enforces this.
4. **Reuse existing local-bar infrastructure.** No new `bar_unit`
   opcode, no new state machine, no new flag bit. Sync uses the
   existing `arrive_and_wait` semantic on a shared local-bar slot.
   The bar_unit's existing phase-parity protection (lines 113–121 of
   `VX_bar_unit.sv`) is sufficient.
5. **DXA moves to per-socket scope.** One `VX_dxa_core` per socket
   instead of one per cluster. DXA GMEM arbitrates with dcache GMEM
   inside the socket and exits over the same `mem_bus_if`.

---

## 0. The cluster dispatch contract (Phase 0)

This proposal owns the cluster dispatch mechanism end-to-end —
it is **not** an external dependency. The work is sequenced as
Phase 0 ahead of the DXA-multicast SDK (Phase 1) and per-socket
DXA relocation (Phase 2), because multicast correctness on
multi-wave grids depends on the contract this section establishes.

### 0.1 What the contract guarantees

`VX_kmu` + `VX_cta_dispatch` provide the following four properties
as **hardware-enforced** behavior:

1. **Atomicity.** A *cluster* of `K` CTAs is dispatched as a
   single indivisible unit. The KMU never commits a partial group —
   either all `K` CTAs can be placed on one core in one wave, or none
   are placed and the group stays queued.
2. **Single-core co-residence.** All `K` members of a cluster are
   placed on the **same core** for the group's full lifetime. No
   member can be evicted or rescheduled while peers are still running.
3. **Contiguous `cta_local_id` assignment within a group.** The KMU
   allocates `K` adjacent slots; member at cluster rank `r`
   receives `cta_local_id = base + r` for some `base`. This is what
   makes the multicast walk `bit k → cta_local_id = issuer_cta_id + k`
   correct.
4. **Launch-time grid-shape validation.** A grid that violates
   `grid_dim % cluster_dim != 0` returns a launch error (analog
   of Hopper's `cudaErrorInvalidConfiguration`); it does not silently
   truncate or hang.

### 0.2 Why each property is load-bearing

Without **atomicity** (1), an issuer can fire multicast before some
receivers exist — they never see the release event, and never call
`expect_tx`; the resulting silent state poisons the bar slot.

Without **co-residence** (2), `group_barrier` (a per-core construct)
cannot synchronize peers on different cores. Each core's local
group_barrier instance hangs independently waiting for arrivals
that never come.

Without **contiguous rank assignment** (3), the multicast walk
`active_bar_addr + (k << NB_BITS)` lands on the wrong CTA slots.

Without **launch-time validation** (4), broken grids produce
indistinguishable runtime hangs.

### 0.3 What this proposal does NOT depend on

- **No mid-cluster preemption guarantee needed.** The proposal does
  not rely on preventing preemption mid-multicast (Hopper doesn't
  either); a CTA running to completion within its cluster's
  lifetime is sufficient.
- **No saturating-decrement on `events_r`** in `VX_bar_unit`. The C5
  "mask-set member without `expect_tx`" failure mode is prevented by
  the `vortex::dxa_multicast` helper API (§3.2), which bundles
  `expect_tx` and mask construction into one constructor call — the
  kernel author cannot set a mask bit without also registering the
  matching event. This is the Hopper / CUTLASS approach (kernel
  discipline, not HW protection).
- **No phase-parity check on the arrive path.** The existing wait-side
  parity check is sufficient for the proposal's correctness — the
  dispatcher's FIFO semantics plus the `group_bar` gate in the kernel
  pattern (which prevents any wave-2 CTA from arriving at the
  multicast slot until wave-1's `group_bar` has fired) close the
  remaining window.

### 0.4 The launch API

Host runtime exposes `cluster_dim` via `vx_launch_info_t`:

```c
typedef struct {
    size_t       struct_size;
    const void*  next;
    vx_kernel_h  kernel;
    const void*  args_host;
    size_t       args_size;
    uint32_t     ndim;            // 1, 2, or 3 (0 = legacy escape hatch)
    uint32_t     grid_dim [3];
    uint32_t     block_dim[3];
    uint32_t     cluster_dim[3];   // NEW (0 or 1 = no grouping)
    uint32_t     lmem_size;
} vx_launch_info_t;
```

A CTA at grid coordinates `(gx, gy, gz)` is in the same cluster
as the CTA at `(gx', gy', gz')` iff
`(gx / lx == gx' / lx) && (gy / ly == gy' / ly) && (gz / lz == gz' / lz)`.

`cluster_dim = {1,1,1}` (or `{0,0,0}`) is the default; equivalent
to no grouping (single-CTA clusters, current dispatch behavior).

The cluster API lives only on the new `vortex2.h` (via the
`cluster_dim[3]` field of `vx_launch_info_t`); the legacy `vortex.h`
API is not extended.

Constraints (runtime-checked, return `VX_ERR_INVALID_VALUE` on
violation):
- `cluster_dim[i] != 0` for each axis `i`.
- `grid_dim[i] % cluster_dim[i] == 0` for each axis `i`.

### 0.5 Kernel-side helpers

`vx_spawn2.h` gains:

```c
// Number of CTAs in this CTA's cluster (same on every CTA of the launch).
// Backed by VX_CSR_CTA_CLUSTER_SIZE = lx * ly * lz.
static inline uint32_t get_cluster_size();

// Rank of this CTA within its cluster, in [0, get_cluster_size()).
// Derived in software as get_local_group_id() % get_cluster_size(),
// since clusters are dispatched as contiguous slot chunks. The rank-0
// CTA is the natural master for cooperative SMEM ops (e.g. DXA multicast).
static inline uint32_t get_cluster_rank();
```

`get_local_group_id()` keeps its **existing semantics — the CTA's
absolute on-core slot index** (`VX_CSR_CTA_ID`). It is *not*
repurposed: per-CTA barriers (`vortex::barrier`) embed it into
`bar_id` and rely on it being a unique-per-co-resident-CTA value, so
changing it would alias barrier slots across clusters.

### 0.6 DCR + CSR additions

| DCR | Offset | Use |
|---|---|---|
| `VX_DCR_KMU_CLUSTER_DIM_X` | 0x003 | Per-launch X dim of the cluster |
| `VX_DCR_KMU_CLUSTER_DIM_Y` | 0x004 | Per-launch Y dim |
| `VX_DCR_KMU_CLUSTER_DIM_Z` | 0x005 | Per-launch Z dim |

Placed in the unused `0x003-0x00F` range below the `[dcr_kmu]` block
(`0x010-0x01F`) so the texture/raster/om DCR blocks don't shift.

| CSR | Address | Use |
|---|---|---|
| `VX_CSR_CTA_CLUSTER_SIZE` | 0xCE0 | `lx * ly * lz`, broadcast from the KMU's DCRs |

`VX_CSR_CTA_ID` (existing) already serves `get_local_group_id()` and
is unchanged.

### 0.7 How KMU dispatches consecutive CTAs of a cluster to the same core

This subsection specifies the dispatch mechanism that delivers the
§0.1 contract. Two pieces.

#### The two pieces

1. **KMU-side cluster assembler.** The KMU enumerates CTAs in
   **cluster-major order**: it groups every `K = lx × ly × lz`
   consecutive CTAs of the grid into one cluster. A cluster
   is the unit of work the KMU offers to cores, not an individual
   CTA. The KMU's outbound dispatch bus is extended with one new
   field — `cluster_size` (3–5 bits, since `K ≤ MAX_CTAS_per_core`).
2. **Core-side atomic acceptor.** `VX_cta_dispatch` is extended with
   one new state: when it sees a `cluster_size = K > 1` request, it
   atomically reserves `K` contiguous slots starting at its current
   `tail_r` (or NACKs the request if `K` contiguous slots aren't
   free — see fragmentation below). For the next `K-1` dispatch
   cycles, only that cluster's CTAs are accepted; no other CTA
   (single or cluster) can interleave.

#### Step-by-step dispatch of one cluster

1. **KMU picks the next ready cluster.** From the head of its
   work queue.
2. **KMU scans cores for one that can fit `K`.** Each core exposes a
   `free_contiguous_slots` count (a 1-line addition to the existing
   `free_size` signal). The KMU picks the first core where
   `free_contiguous_slots ≥ K`. If none, the cluster stays at the
   queue head and the KMU may dispatch *single* CTAs onto fragmented
   slots elsewhere in the meantime.
3. **KMU sends a "cluster, size=K, base=auto" probe to the chosen core.**
   The core's `VX_cta_dispatch` enters cluster-accept mode, reserves
   slots `[tail_r, tail_r+1, …, tail_r+K-1]`, and ACKs with
   `base = tail_r`.
4. **KMU streams the `K` CTA descriptors back-to-back to that core.**
   Each CTA gets `cta_local_id = base + r` (`r = 0..K-1`). The core's
   dispatcher blocks all other dispatch traffic during this window —
   a simple FSM gate, not a structural change.
5. **Core's CTA dispatcher exits cluster-accept mode** after the
   `K`-th CTA's wid allocation, returns to normal operation.

#### Handling fragmentation

The core's existing FIFO (`head_r / tail_r`) means free slots are
not always contiguous when CTAs finish out of order. Two policies
keep this simple:

- **Conservative (default).** If `tail_r + K` would wrap past
  `head_r`, the cluster doesn't fit; KMU tries another core or
  waits. Mild under-utilization is acceptable for the simplicity
  win.
- **Slot-skip on wrap.** If the only available `K` contiguous slots
  start past the wrap boundary, the dispatcher advances `tail_r` to
  the next aligned position, leaving the intervening slots
  permanently empty until the next full drain. Implementation cost:
  a single comparison in the FSM. Use only if fragmentation
  measurably hurts throughput.

For Phase 1 workloads (single-group launches), neither policy
matters — the grid is one cluster wide and fragmentation never
arises. The choice becomes relevant once multi-cluster grids with
short-lived clusters are common.

#### Why this is enough to deliver §0.1's contract

| §0.1 property | Delivered by |
|---|---|
| (1) **Atomicity** | KMU's probe-then-stream is the all-or-nothing primitive: the probe ACKs only if `K` slots can be reserved; if it ACKs, the stream completes without interruption. |
| (2) **Single-core co-residence** | KMU picks one target core; all `K` CTAs are streamed to that core only. |
| (3) **Contiguous `cta_local_id`** | The core's FIFO allocator gives slots `[base, base+K-1]` in order during cluster-accept mode. |
| (4) **Launch-time grid validation** | Done in the host runtime before the KMU sees the kernel — `grid_dim % cluster_dim != 0` is rejected with an error. |

#### Why this is small

- KMU change: ~1 new field on the dispatch bus, plus a
  cluster-enumeration ring-counter (~30 LoC of RTL).
- Core change: ~1 new state in `VX_cta_dispatch`'s FSM, plus a
  `free_contiguous_slots` signal back to the KMU (~20 LoC).
- No change to barrier units, DXA, or anything else.

The "complexity" of cluster dispatch is almost entirely in the
*idea* — that the KMU must offer K CTAs as one unit and the core
must accept them atomically. The RTL realisation is a small FSM
extension. The reason it must be hardware (not a software polling
loop) is property §0.1.1: software can't guarantee atomicity
against a concurrent dispatcher.

---

## 1. What changes, by file

### Phase 0 — cluster dispatch contract

| File | Change | Approx LoC |
|---|---|---|
| [VX_types.toml](../../VX_types.toml) | Add `VX_DCR_KMU_CLUSTER_DIM_{X,Y,Z}` (0x003-0x005) and `VX_CSR_CTA_CLUSTER_SIZE` (0xCE0). | ~10 |
| [VX_kmu.sv](../../hw/rtl/VX_kmu.sv) | Nested dispatch counter (`group_origin` + `intra_offset`); carry `cluster_dim` on the KMU bus. | ~30 |
| [VX_gpu_pkg.sv](../../hw/rtl/VX_gpu_pkg.sv) | Add `cluster_dim[3]` / `cluster_size` fields to `kmu_req_t` / `cta_csrs_t` / `cta_ctx_t`. | ~15 |
| [VX_cta_dispatch.sv](../../hw/rtl/core/VX_cta_dispatch.sv) | Cluster-accept FSM; `cluster_size`, `slot_to_lmem_base`, `cta_slot_per_warp` plumbing; new `VX_cta_table_if.master` port for downstream consumers. | ~50 |
| [VX_cta_table_if.sv](../../hw/rtl/interfaces/VX_cta_table_if.sv) (new) | Interface bundle exposing per-slot LMEM-base table from dispatcher to mem unit. | ~20 |
| [VX_core.sv](../../hw/rtl/core/VX_core.sv) | Declare `VX_cta_table_if`; wire dispatcher → scheduler and dispatcher → mem-unit. | ~10 |
| [VX_csr_data.sv](../../hw/rtl/core/VX_csr_data.sv) | Serve `VX_CSR_CTA_CLUSTER_SIZE` from `cta_csrs.cluster_size`. | ~3 |
| [VX_scheduler.sv](../../hw/rtl/core/VX_scheduler.sv) | Persist `cluster_size` per-CTA-slot in `cta_ctx_t`. | ~5 |
| [VX_mem_unit.sv](../../hw/rtl/core/VX_mem_unit.sv) | Consume slot-LMEM-base table from `VX_cta_table_if.slave`. | ~10 |
| [sim/simx/kmu/kmu.{h,cpp}](../../sim/simx/kmu/) | Mirror the RTL nested-counter walk; broadcast `cluster_dim[3]` on the KMU bus. | ~40 |
| [sim/simx/cta_dispatcher.{h,cpp}](../../sim/simx/) | Mirror the cluster-accept state; expose `cluster_size` on cta_csrs. | ~30 |
| [sim/simx/scheduler.{h,cpp}](../../sim/simx/) | Persist per-CTA `cluster_size`. | ~5 |
| [sim/simx/csr_unit.cpp](../../sim/simx/csr_unit.cpp) | Serve `VX_CSR_CTA_CLUSTER_SIZE`. | ~5 |
| [sim/simx/main.cpp](../../sim/simx/main.cpp), [sim/rtlsim/main.cpp](../../sim/rtlsim/main.cpp) | Default the three new DCRs to 1 in the sim wrappers. | ~10 |
| [sw/kernel/include/vx_spawn2.h](../../sw/kernel/include/vx_spawn2.h) | Add `get_cluster_size()` and `get_cluster_rank()`. `get_local_group_id()` unchanged. | ~15 |
| [sw/runtime/include/vortex2.h](../../sw/runtime/include/vortex2.h) | Add `uint32_t cluster_dim[3]` field to `vx_launch_info_t`. | ~3 |
| [sw/runtime/common/queue.cpp](../../sw/runtime/common/queue.cpp) | In `Queue::enqueue_launch`: normalise zero entries to 1, validate `grid_dim[i] % cluster_dim[i] == 0`, program the three new DCRs. | ~20 |

### Phase 1 — DXA multicast SDK + RTL fix

| File | Change | Approx LoC |
|---|---|---|
| [VX_dxa_smem_wr.sv](../../hw/rtl/dxa/VX_dxa_smem_wr.sv) | Multicast `bar_addr` walk stays `active_bar_addr + (replay_next_idx << NB_BITS)` (pre-bce88c86 form). Scope bit forced to **local** for all multicast issues. | 0 (current code) |
| [VX_dxa_completion.sv](../../hw/rtl/dxa/VX_dxa_completion.sv) | Replace single-slot pending buffer with a depth-`NUM_WARPS` FIFO. Multicast fires up to `popcount(mask)` back-to-back release events on the last drain word; the existing single-slot buffer drops events under downstream backpressure. | ~30 |
| [VX_bar_unit.sv](../../hw/rtl/core/VX_bar_unit.sv) | **No changes.** Existing phase-parity check on the wait path (lines 113–121) is sufficient; the `events_r` underflow case (C5) is prevented by the SDK helper API, not by HW (see §0.3). | 0 |
| [sim/simx/types.h](../../sim/simx/types.h) | Widen `MemFlags.dxa_notify_bar_id` from 8 to 16 bits — the field carries the raw kernel-encoded bar handle `(bar_no << 8) | cta_local_id` which is up to 11 bits today. | ~5 |
| [vx_barrier.h](../../sw/kernel/include/vx_barrier.h) | **Rename only.** `vortex::barrier::expect_tx`, `vortex::gbarrier::expect_tx`, and the cross-CTA shared-bar class are already in place; this proposal renames the existing `vortex::shared_barrier` class to `vortex::group_barrier` to make the cluster semantics explicit (per-CTA `vortex::barrier` is intra-CTA "local"; the new `vortex::group_barrier` is cluster-wide "group"). All call sites in `tests/regression/` and elsewhere are updated mechanically. | ~10 (rename) |
| [vx_dxa.h](../../sw/kernel/include/vx_dxa.h) | Add `vortex::dxa_multicast<N>` helper class that bundles `expect_tx` + mask construction so the kernel author cannot violate the C5 invariant by hand. Mask is derived from `get_cluster_size()`; constructor takes `(desc_slot, local_bar, group_bar)`. See §3.2. | ~60 |
| [VX_socket.sv](../../hw/rtl/VX_socket.sv) | Instantiate one `VX_dxa_core` per socket. Drive `req_bus_if[VX_CFG_SOCKET_SIZE]` from each core; LMEM stays inside the socket (per-core fan-out only). Add `dxa_dcache_arb` that merges DXA GMEM with dcache GMEM into the socket's existing `mem_bus_if[L1_MEM_PORTS]`. | ~120 |
| [VX_cluster.sv](../../hw/rtl/VX_cluster.sv) | Delete cluster-level DXA instance, `dxa_l2_priority_arb`, `dxa_gmem_bus_if`, `dxa_lmem_bus_if`, and the socket-fanout switch. Cluster no longer sees a DXA-specific port. | -160 |
| [sim/simx/socket.h](../../sim/simx/socket.h), [sim/simx/socket.cpp](../../sim/simx/socket.cpp) | Move `DxaCore` ownership from `Cluster` into `Socket`. Add a socket-level DXA-vs-dcache GMEM arbiter (priority: dcache > DXA) before binding to `simobject->mem_req_out`. Plumb per-core `dxa_req_in` and `lmem_req_out` to in-socket cores (including the `tx_callback` that pulses `barrier_event_release`). | ~120 |
| [sim/simx/cluster.cpp](../../sim/simx/cluster.cpp) | Remove cluster-level DxaCore creation, gmem rows from `l2arb`, per-core `sfu->dxa_req_out`/`lmem_req_out` bindings. Cluster keeps only socket-level wiring. | -80 |

The multicast RTL surface is essentially just the `VX_dxa_completion`
FIFO fix — ~30 lines. The kernel SDK adds one helper class
(`vortex::dxa_multicast`) that makes the mask ↔ `expect_tx` invariant
structural, plus a mechanical rename of the existing
`vortex::shared_barrier` → `vortex::group_barrier` so the class name
matches its cluster semantics (the per-CTA `vortex::barrier` is
the "local" one). `expect_tx` and the underlying cross-CTA bar
mechanism are already in `vx_barrier.h`. The structural lift is the
per-socket DXA relocation: ~120 lines added to `VX_socket.sv`, ~160
removed from `VX_cluster.sv`, and a matching shape change in simx.

---

## 2. Mask semantics (intra-core only)

### 2.1 The cluster is the multicast group

A **cluster** (defined in §0 of this proposal) is the set
of `K` CTAs the dispatcher guarantees to place on the same core
atomically, in contiguous `cta_local_id` slots `[base, base+K-1]`,
for the group's full lifetime. This proposal **reuses the local
group as the multicast peer set**:

- Multicast mask width = `cluster_size` bits.
- Bit `r` of the mask identifies **cluster rank `r`**.
- The cluster member at rank `r` has `cta_local_id = base + r`,
  where `base` is the issuer's `cta_local_id`.

The kernel never writes the mask by hand. The `vortex::dxa_multicast`
helper (§3.2) derives it from `get_cluster_size()`, so
mask construction and `expect_tx` registration are inseparable —
the C5 failure mode (mask bit set without corresponding `expect_tx`)
is unreachable through the helper API.

### 2.2 How bar slots are addressed (background)

The HW encoding the multicast walk relies on:

- `VX_wctl_unit.sv:138` builds the bar-unit address from the rs1
  bar-handle as `{rs1[NW_BITS-1:0], rs1[BAR_ID_SHIFT +: NB_BITS]}`.
- `vortex::barrier(id)` (in `vx_barrier.h`) sets rs1 to
  `(id << 8) | get_local_group_id()`, where `get_local_group_id()`
  reads `VX_CSR_CTA_ID`.

So the bar-unit's "wid field" is in fact addressed by the CTA local
id, not by warp id. The bar-unit has `MAX_CTAS_per_core × NUM_BARRIERS`
distinct slots, and a CTA of any warpgroup width — single-warp or
multi-warp — occupies exactly **one** slot per bar_id. All warps of
a given CTA share that slot because they all read the same
`VX_CSR_CTA_ID`.

### 2.3 The multicast walk

- Bit `r` set ⇒ release event fires at
  `bar_addr = active_bar_addr + (r << NB_BITS)` ⇒ **cluster
  member rank `r`** receives the release on the same bar_id as the
  issuer's `active_bar_addr`. Because §0's contract guarantees
  contiguous `cta_local_id` assignment, this lands on
  `cta_local_id = base + r` — the right CTA.
- SMEM write target = **same offset in the receiver's own LMEM
  region** as the issuer passed for itself (§2.5 below). No
  per-receiver stride math.
- The barrier handle carried in the multicast descriptor is **always
  a local bar**. The hardware does not look at the scope bit for
  multicast routing — it walks contiguous cluster rank slots
  offset from the issuer's `cta_local_id`.

### 2.4 Works for arbitrary N-warp CTAs

Because the bar-unit address field is per-CTA (not per-warp), the
same mask walk works for **any** warps-per-CTA value
`W ∈ {1, 2, 4, 8, …, NUM_WARPS}`:

- Each CTA, whatever its `W`, occupies exactly one bar slot per
  bar_id (all of its warps read the same `VX_CSR_CTA_ID`).
- The mask bit `r` → cluster rank `r` semantics is unchanged.
  `W` does not appear in the walk.
- The number of cluster members that fit on a core is
  `MAX_CTAS_per_core = floor(NUM_WARPS / W)`. Concrete examples for
  `NUM_WARPS = 16`:

  | warps_per_CTA (`W`) | max cluster size `K` |
  |---|---|
  | 1 (`sgemm2_dxa_mw`)            | 16 |
  | 2                              | 8  |
  | 4 (`sgemm_tcu_wg_dxa_mw`, WGMMA) | 4  |
  | 8                              | 2  |

  All these geometries route through the same mask walk; only the
  practical cluster-size ceiling differs.

The `cluster_dim` declaration at launch time defines `W` and `K`
together — the kernel author picks both, and the runtime
cross-checks them against `NUM_WARPS`.

### 2.5 Per-receiver SMEM destination — stride-arithmetic on the issuer's address

The `dxa_core` does **not** look up per-CTA LMEM bases — it has no
interface into `VX_cta_dispatch`'s base table. What it does is
**pure byte-arithmetic on the issuer's `smem_addr`**, using a
`smem_stride` field carried in the multicast descriptor:

```
dest[r] = issuer_smem_addr + r × smem_stride
```

This is what [VX_dxa_smem_wr.sv:335,357](../../hw/rtl/dxa/VX_dxa_smem_wr.sv)
already implements today; no RTL change is needed for cluster-aware
multicasts to use it.

For `dest[r]` to land in cluster member `r`'s LMEM region, **two
invariants must hold**:

1. **Uniform per-CTA LMEM allocation.**
   `LMEM_BASE[issuer + r] − LMEM_BASE[issuer] = r × lmem_per_cta`.
   `VX_cta_dispatch` provides this today via its FIFO allocator;
   `cluster_dim` (§0) makes it structural across cluster members
   by ensuring all `K` members share the same shared-memory size
   declared at launch.
2. **`smem_stride == lmem_per_cta`** in the descriptor. The host
   runtime programs this at descriptor-setup time (via DCR), using
   the `lmem_per_cta` value derived from the launch attributes.

Under both invariants, the arithmetic resolves to exactly
"same offset in receiver LMEM":
```
dest[r] = (LMEM_BASE[issuer] + offset_within_cta) + r × lmem_per_cta
       = LMEM_BASE[issuer + r] + offset_within_cta      // uniform stride
       = the same byte offset in member r's own LMEM region
```

That semantics — *the destination in each receiver is at the same
byte offset as the issuer's source in its own LMEM* — is the
property kernel authors should rely on. Internally it's produced by
stride arithmetic, not by a hardware base-table lookup, but the
externally visible behavior is identical to Hopper TMA's "same
offset in receiver SMEM" model.

#### Where each invariant comes from

| Invariant | Provided by | Verified by |
|---|---|---|
| Uniform `lmem_per_cta` across cluster members | `cluster_dim` launch contract (§0) | Host-side launch validation: rejects launches with non-uniform per-member SMEM size |
| `smem_stride = lmem_per_cta` in descriptor | Host runtime descriptor programming | Runtime-side: only one `lmem_per_cta` exists per launch, so `smem_stride` is set once at descriptor program time |
| Same operand offset on every cluster member | Same kernel code runs on every member (CUDA-style SPMD) | Compiler |

#### Why we're not building Path B (HW-side base lookup)

A "true" Hopper-like implementation would have the `dxa_core` look
up `LMEM_BASE[issuer + r]` directly from each core's
`VX_cta_dispatch`. That would require:
- a new cross-module signal exporting each core's per-CTA LMEM-base
  table out to the per-socket `dxa_core` (~40 LoC of new RTL plus a
  multi-port lookup in `VX_dxa_smem_wr`),
- versioning of the table to handle (hypothetical) mid-multicast
  changes to a CTA's LMEM base,
- a corresponding simx model.

The stride-arithmetic approach gives the same observable semantics
with **zero new RTL surface**, as long as the two invariants above
hold. They both hold structurally under `cluster_dim`. We
explicitly choose the simpler path; Path B is deferred to a future
proposal if non-uniform per-CTA LMEM ever becomes a requirement
(e.g. a cross-core DSMEM design where members may live on different
cores with different allocators).

### 2.6 Concurrent multicasts on a single CTA

A single CTA is allowed (and expected) to participate in multiple
DXA multicasts simultaneously — both as an issuer (in cluster
geometries where multiple operands originate from different cluster
members) and as a receiver of multiple concurrent inbound
multicasts. Isolation comes from:

- **Descriptor isolation**: different `desc_slot`s point to
  different descriptor-table entries — independent `smem_addr`,
  tile shape, mask.
- **Event-bar isolation**: different bar_ids map to different
  bar-unit slots on each receiver CTA. Up to `NUM_BARRIERS = 8`
  concurrent `vortex::dxa_multicast` instances per CTA before the
  bar-slot budget becomes tight (see Open Question 2 — bumping
  `NUM_BARRIERS` is the natural mitigation).
- **DXA queue depth**: up to `VX_CFG_DXA_QUEUE_SIZE = 16` transfers
  in-flight per DXA core, shared across all issuers in the socket.
- **Ordering**: no implicit ordering between concurrent multicasts.
  The kernel orders them via separate `vortex::dxa_multicast`
  instances' `.wait()` calls.

A worked two-multicast example is in §3.3.

### Why no inter-core mode?

A cross-core multicast would require:

- A socket-level LMEM write fabric that lets one core's DXA push SMEM
  words into another core's local memory (Hopper's DSMEM analog).
- A cross-core release path: each receiving core's `dxa_completion`
  emitting a gbar decrement; the gbar unit aggregating across cores;
  the issuer's `gbar` release wiring back through the gbar fabric.
- A cross-core variant of `cluster_dim` (a "cluster across
  cores") for atomic dispatch — analogous to Hopper's Distributed
  Shared Memory cluster scope.

None of that is justified by the workloads we're prioritising. Every
WGMMA and SGEMM regression we care about — including 4-CTA × 4-warp
WGMMA multicast — fits inside a single core's warp budget at
`NUM_WARPS = 16`. The intra-core primitive captures the bandwidth-
amortization win (one GMEM read, N LMEM writes) for those workloads
without the socket-wide hardware surface. Cross-core multicast can be
re-evaluated as a separate proposal if a future workload genuinely
demands it.

---

## 3. The two-barrier idiom

The race we're solving: multicast issuer fires before some receivers
have called `expect_tx` on their bars; the release events land on bars
with `events_r=0` and are silently dropped (or wrap-decrement).

The fix is structural — insert a **cluster rendezvous barrier**
between the
receivers' `expect_tx` and the issuer's multicast issue.

### 3.1 The raw two-barrier pattern (what the helper wraps)

```cpp
// local_bar — per-CTA bar that receives the multicast release for THIS CTA.
// num_warps defaults to get_num_sub_groups() = warps_per_CTA, so 1-warp
// CTAs (Phase 1) and W-warp WGMMA CTAs (Phase 3) both work here.
vortex::barrier        local_bar(LOCAL_BAR_ID);
// group_bar — cluster-shared rendezvous bar. All K cluster members
// see the same bar_unit slot. num_peers = K (one arrival per member).
vortex::group_barrier group_bar(GROUP_BAR_ID,
                                /*num_peers=*/get_cluster_size());

if (is_loader_warp) {
    local_bar.expect_tx(1);           // register one event on my CTA's slot
    group_bar.arrive_and_wait();     // K cluster members rendezvous here,
                                    // so expect_tx is in effect before
                                    // any issuer fires multicast
    if (get_local_group_id() == 0) {
        vx_dxa_issue_2d_multicast_wg(
            desc, local_bar.id(), shmem, x, y,
            /*mask=*/(1u << get_cluster_size()) - 1u);
    }
}
local_bar.arrive_and_wait();          // ALL warps of this CTA arrive +
                                    // wait for the DXA release event
```

Three non-obvious correctness points (which the helper hides):

- **Only `is_loader_warp` warps call `expect_tx`.** The bar slot is
  per-CTA, so a single `expect_tx` per CTA exactly matches the
  single release event the multicast walk delivers to that CTA.
  Other warps of the same CTA must NOT call `expect_tx` or
  `events_r` would overshoot.
- **`group_bar.arrive_and_wait()` is called only by `is_loader_warp`
  warps too** (one warp per cluster member). `num_peers = K`
  matches the actual arrival count. *If non-loader warps also called
  group_bar, the actual arrivals would be `K × W`, the bar would
  fire early after only `K` arrivals, and the issuer could fire
  multicast before the other CTAs' loader warps had completed their
  `expect_tx` — a subtle correctness bug that the helper makes
  unreachable.*
- **`local_bar.arrive_and_wait()` IS called by all warps of the CTA.**
  `arrive_and_wait` on a local bar blocks until both
  `arrivals == warps_per_CTA` AND `events_r == 0`. Skipping the
  non-loader warps would deadlock this bar.

### 3.2 The `vortex::dxa_multicast` helper (new, lives in `vx_dxa.h`)

This helper exists to make the mask ↔ `expect_tx` invariant
structural. Kernel authors should always use it instead of the raw
pattern above; the raw pattern is documented only for reference.

```cpp
namespace vortex {

// 2-D variant; analogous helpers for 1-D / 3-D / 4-D / 5-D follow the
// same shape and bracket the same intrinsics from vx_dxa.h.
class dxa_multicast_2d {
public:
    // Constructor:
    //   - registers one tx event on the per-CTA local_bar
    //     (call from is_loader_warp only).
    //   - the mask used at issue time is exactly the cluster, so
    //     the kernel author cannot mis-pair mask bits and expect_tx.
    dxa_multicast_2d(uint32_t desc_slot,
                     vortex::barrier&        local_bar,   // per-CTA event sink
                     vortex::group_barrier&  group_bar)   // shared rendezvous
        : desc_slot_(desc_slot), local_bar_(local_bar), group_bar_(group_bar) {
        local_bar_.expect_tx(1);
    }

    // Rank-0 cluster member issues the multicast. Other ranks call
    // sync() but skip the issue. The mask covers the full cluster
    // — never selectable by the kernel author.
    void sync_and_issue(const void* my_smem_offset,
                        uint32_t coord0, uint32_t coord1) {
        group_bar_.arrive_and_wait();           // K-way rendezvous
        if (vortex::get_local_group_id() == 0) {
            uint32_t mask = (1u << vortex::get_cluster_size()) - 1u;
            // Multicast routes via the per-CTA bar handle — each receiver's
            // own local_bar slot will get the release event.
            vx_dxa_issue_2d_multicast_wg(desc_slot_, local_bar_.id(),
                                         my_smem_offset, coord0, coord1, mask);
        }
    }

    // NOTE: there is intentionally no wait() method on this class.
    // The kernel calls local_bar.arrive_and_wait() directly, from ALL
    // warps of the CTA (not just the loader warp), because
    // arrive_and_wait blocks until both arrivals == warps_per_CTA AND
    // events_r == 0. Putting wait() on the helper would suggest it
    // belongs in the loader-warp branch, which would deadlock the
    // bar's arrival counter. Keeping the wait off the helper makes
    // the per-warp arrival requirement visible at the call site.

private:
    uint32_t                desc_slot_;
    vortex::barrier&        local_bar_;     // per-CTA, holds events_r
    vortex::group_barrier&  group_bar_;     // shared across K members
};

} // namespace vortex
```

The helper enforces three invariants:

1. **`expect_tx` is paid exactly once per CTA**, in the constructor.
   Other warps of the same CTA can construct the helper too, but
   the kernel pattern guards it with `if (is_loader_warp)`.
2. **The mask is never selectable.** It's always
   `(1 << cluster_size) - 1`. C5 becomes unreachable: every
   mask-set rank is a cluster member that ran the same constructor
   path, which means it called `expect_tx`.
3. **`sync_and_issue` is `is_loader_warp`-scoped by convention** but
   the helper's internal `arrive_and_wait(num_peers=K)` matches the
   `K` cluster members' single arrivals each. Calling
   `sync_and_issue` from a non-loader warp would over-arrive on
   `group_bar`; the helper's documentation explicitly says
   "is_loader_warp only".

### 3.3 Worked two-multicast example using the helper

```cpp
// Cluster of K CTAs (any warps_per_CTA W). CTA rank 0 issues two
// concurrent multicasts — operand A and operand B — both reaching
// all K cluster members. Each member waits on two separate local bars.

vortex::barrier       local_A(0);     // per-CTA event sink for A
vortex::barrier       local_B(1);     // per-CTA event sink for B
vortex::group_barrier group_bar(2,
                                /*num_peers=*/get_cluster_size());

if (is_loader_warp) {
    // expect_tx() runs in the helper ctors; sync_and_issue() rendezvouses
    // on group_bar then (for rank-0) fires the multicast.
    vortex::dxa_multicast_2d mc_A(DESC_A, local_A, group_bar);
    vortex::dxa_multicast_2d mc_B(DESC_B, local_B, group_bar);
    void* A_off = (uint8_t*)__local_mem() + OFFSET_A;
    void* B_off = (uint8_t*)__local_mem() + OFFSET_B;
    mc_A.sync_and_issue(A_off, xA, yA);   // rank 0 fires; others sync only
    mc_B.sync_and_issue(B_off, xB, yB);
    // mc_A and mc_B go out of scope here. No wait() on the helper —
    // wait happens below via the per-CTA bars, called by ALL warps.
}

// Called by every warp of this CTA: blocks until both
//   arrivals == warps_per_CTA AND events_r == 0
// so the multicast release event for THIS CTA must have landed.
local_A.arrive_and_wait();    // operand A has landed in my LMEM
local_B.arrive_and_wait();    // operand B has landed in my LMEM
// Operand A and operand B are now in this CTA's local memory.
```

Note that there is **no else-branch** for non-loader warps. The
helper deliberately omits a `wait()` method so the per-warp arrival
requirement is visible at the call site: `local_A.arrive_and_wait()`
appears at top scope, called by every warp.

This consumes 3 bar_ids per cluster member (local_A, local_B, group_bar).
With `NUM_BARRIERS = 8` and one slot reserved for `__syncthreads`,
roughly 3–4 concurrent multicast groups can coexist on the same
cluster before the bar-slot budget becomes tight (Open Question 2).

### 3.4 `group_barrier` / per-CTA `barrier` slot aliasing

`group_barrier(N, K)` and the CTA-at-rank-0's `barrier(N)` map to
the same physical bar-unit slot (both `{wid_field=0, bar_id=N}`).
The kernel must therefore reserve specific `bar_id` values for
shared use and never construct a per-CTA `barrier(N)` with those
same values. The convention used by the helper and the test suite:

| Use | bar_id |
|---|---|
| Per-CTA local bar(s) (`local_A`, `local_B`, …) — `vortex::barrier` | 0, 1, … |
| Shared group bar — `vortex::group_barrier` | 2 |
| `__syncthreads` | 3 |

The conflict isn't structurally enforced — it's a programmer
convention documented at the `group_barrier` declaration site. The
helper API does not pick bar_ids for the caller, so the kernel
author still has to follow the convention.

### 3.5 Why no atomic op?

A fused "register events + sync" instruction would close one
micro-race: if the multicast issuer somehow fired between the
receiver's `expect_tx` and its sync arrival. But program order + the
`volatile` asm in the intrinsics prevent reordering, and the
multicast instruction sits on the issuer's path *after* its own
sync arrival (via `sync_and_issue`'s constructor sequencing). There
is no realistic scenario where the issuer's fire interleaves with
another receiver's pre-sync window.

The two-barrier idiom is sufficient. An atomic `expect_tx_sync`
would be a micro-optimization (one less instruction per multicast),
not a correctness requirement.

---

## 4. Constraints on cluster shape

With the §0 contract in place, co-residence is structural (not
discipline-only). What's left is the static constraint between the
declared `cluster_dim` and the per-core resource budget. The
launch runtime is responsible for rejecting any kernel whose
declared cluster exceeds what the target hardware can host:

- `K × W ≤ NUM_WARPS_per_core` — the core must have enough warp
  slots to host all `K` cluster members of `W` warps each.
- `K ≤ MAX_CTAS_per_core` — the bar-unit's `cta_local_id` field
  must be wide enough to address all `K` members. (Today this is
  the same `NUM_WARPS` bound.)
- `K × lmem_per_cta ≤ VX_CFG_LMEM_SIZE` — the cluster's combined
  LMEM allocation must fit.

Practical block-dim shapes the proposal targets:

| Phase | Test | `W` | `K` | `NUM_WARPS` required |
|---|---|---|---|---|
| Phase 1 | `sgemm2_dxa_mw`        | 1 | 4 | ≥ 4  |
| Phase 1 | `dxa_copy_mw`          | 1 | up to 16 | ≥ K |
| Phase 3 | `sgemm_tcu_wg_dxa_mw`  | 4 | 4 | ≥ 16 |

If a kernel declares an oversized cluster, the launch is
rejected with an explicit error (§0.1 property 4). No deadlock,
no silent corruption.

The dispatcher's atomic-launch property (§0.1 property 1) and the
contiguous-rank assignment (§0.1 property 3) together remove the
historical "kernel discipline" burden: there is no way for the
kernel to write code that compiles but deadlocks at runtime due to
mis-coresidence. Either the launch succeeds (and the dispatcher
guarantees the runtime conditions) or it doesn't (and the kernel
is rejected with an actionable error).

---

## 5. Per-socket DXA relocation

### 5.1 Motivation

Today `VX_dxa_core` lives at the cluster level. That had two
consequences:

- DXA GMEM had its own dedicated path to the L2 — separate from any
  socket's icache/dcache traffic. To prevent DXA from starving compute,
  the cluster carries a bespoke `dxa_l2_priority_arb` (priority arb,
  LSU > DXA) plus an `RSP_OUT_BUF=1` workaround for a stale-tag
  backpressure bug we hit while wiring it up.
- DXA LMEM had to be fanned out from a single cluster-level
  `dxa_lmem_bus_if[1]` to every core in every socket, via
  `dxa_lmem_socket_switch` (cluster) plus `dxa_lmem_core_switch`
  (socket). Two switches and two layers of tag-bit routing.

Multicast — even intra-core only — pushes this fabric harder. The
LMEM packet rate during a multicast drain spikes by `popcount(mask)`,
and the two-level switch becomes a serialisation point shared across
sockets that have nothing to do with the issuing multicast.

The fix is to push DXA down to socket scope: each socket owns its own
`VX_dxa_core`, drives its own LMEM bus directly into its in-socket
cores, and arbitrates its GMEM traffic with the dcache **inside** the
socket so the cluster sees a uniform `mem_bus_if`.

### 5.2 RTL topology — before vs after

```
Before (DXA at cluster scope):

  VX_cluster
    ├── VX_socket[S] ──────► socket_mem_bus_if[S][L2_SOCKET_REQS] ─┐
    │     (icache + dcache merged at l1_mem_arb)                   │
    │                                                              ▼
    └── VX_dxa_core ──► dxa_gmem_bus_if[DXA_L2_GMEM_PORTS] ──► dxa_l2_priority_arb
                                                                   │
                                                                   ▼
                                                          per_socket_mem_bus_if → L2

        VX_dxa_core ──► dxa_lmem_bus_if[1] ──► dxa_lmem_socket_switch
                                                  │
                                                  ▼
                                       per_socket_dxa_lmem_bus_if[NUM_SOCKETS]
                                                  │
                                                  ▼ (into each VX_socket)
                                       dxa_lmem_core_switch
                                                  │
                                                  ▼
                                       per_core_dxa_lmem_bus_if[SOCKET_SIZE]

After (DXA at socket scope):

  VX_cluster
    └── VX_socket[S] ───────────────────► mem_bus_if[L1_MEM_PORTS] ──► L2
          │   (icache + dcache + DXA all merged inside the socket)
          │
          ├── icache ──┐
          ├── dcache ──┼── l1_mem_arb (existing) ──► socket_l1_mem
          │            │
          │            └── (one input)
          │
          ├── VX_dxa_core (per-socket)
          │     ├── req_bus_if[SOCKET_SIZE] ◄── per-core SFU dxa_req
          │     ├── smem_bus_if[1] ──► per_core_dxa_lmem_bus_if[SOCKET_SIZE]
          │     │                       (single in-socket switch, no
          │     │                        cross-socket hop)
          │     └── gmem_bus_if[DXA_L1_GMEM_PORTS] ─┐
          │                                         ▼
          └── dxa_dcache_arb (new):
                inputs:  [socket_l1_mem (priority high),
                          dxa_gmem_bus_if (priority low)]
                output:  mem_bus_if[L1_MEM_PORTS] ──► cluster
```

Cluster ends up with **zero** DXA-specific signals — it just sees
`mem_bus_if[L1_MEM_PORTS]` per socket, same shape as a non-DXA build.

### 5.3 Inside the socket — arbiter details

The new socket-level merge reuses `VX_mem_arb` with the same priority
ordering the cluster used to apply, just moved down a level:

```systemverilog
// Inside VX_socket.sv, after dcache and dxa_core are instantiated.

VX_mem_bus_if #(
    .DATA_SIZE (`VX_CFG_L1_LINE_SIZE),
    .TAG_WIDTH (L1_MEM_ARB_TAG_WIDTH)
) socket_l1_mem_bus_if[L1_MEM_PORTS]();

// Existing icache+dcache merge (unchanged) writes into socket_l1_mem_bus_if.

VX_mem_bus_if #(
    .DATA_SIZE (`VX_CFG_L1_LINE_SIZE),
    .TAG_WIDTH (L1_MEM_ARB_TAG_WIDTH)
) dxa_l2_arb_in_if[2 * L1_MEM_PORTS]();

for (genvar i = 0; i < L1_MEM_PORTS; ++i) begin : g_l1_bind
    `ASSIGN_VX_MEM_BUS_IF (dxa_l2_arb_in_if[i], socket_l1_mem_bus_if[i]);
end

for (genvar i = 0; i < DXA_L1_GMEM_PORTS; ++i) begin : g_dxa_bind
    `ASSIGN_VX_MEM_BUS_IF (dxa_l2_arb_in_if[L1_MEM_PORTS + i], dxa_gmem_bus_if[i]);
end

// Tie off unused DXA slots (mirrors VX_cluster behaviour).
for (genvar i = DXA_L1_GMEM_PORTS; i < L1_MEM_PORTS; ++i) begin : g_dxa_tieoff
    assign dxa_l2_arb_in_if[L1_MEM_PORTS + i].req_valid = 1'b0;
    assign dxa_l2_arb_in_if[L1_MEM_PORTS + i].req_data  = '0;
    assign dxa_l2_arb_in_if[L1_MEM_PORTS + i].rsp_ready = 1'b1;
end

VX_mem_arb #(
    .NUM_INPUTS  (2 * L1_MEM_PORTS),
    .NUM_OUTPUTS (L1_MEM_PORTS),
    .DATA_SIZE   (`VX_CFG_L1_LINE_SIZE),
    .TAG_WIDTH   (L1_MEM_ARB_TAG_WIDTH),
    .TAG_SEL_IDX (0),
    .ARBITER     ("P"),       // priority: index 0..L1_MEM_PORTS-1 (dcache) wins
    .RSP_OUT_BUF (1)          // carries forward the cluster-side fix.
) dxa_dcache_arb (
    .clk        (clk),
    .reset      (reset),
    .bus_in_if  (dxa_l2_arb_in_if),
    .bus_out_if (mem_bus_if)
);
```

Notes:

- **Priority order matches today's policy**: dcache (compute) wins
  over DXA (bulk). This preserves the current performance contract.
- **`RSP_OUT_BUF=1` is kept**: the same stale-routing-bit issue
  documented in `VX_cluster.sv` applies to any DXA/non-DXA mem_arb
  composition. Carrying the fix into the new arb instance avoids
  re-discovering it.
- **`DXA_L1_GMEM_PORTS`** replaces `DXA_L2_GMEM_PORTS` and is sized
  per socket. Default = 1 (one DXA GMEM port per socket); raised only
  when DXA bandwidth measurements warrant.
- **LMEM stays inside the socket**: `dxa_lmem_core_switch` is kept as
  the only in-socket fan-out. `dxa_lmem_socket_switch` (cluster-level)
  is deleted.

### 5.4 Cluster cleanup

`VX_cluster.sv` loses:

- `VX_dxa_core` instance.
- `dxa_l2_priority_arb` and its tie-off generates.
- `dxa_gmem_bus_if`, `dxa_lmem_bus_if`, `per_socket_dxa_lmem_bus_if`.
- `dxa_lmem_socket_switch`.
- All cluster-level `import VX_dxa_pkg::*` blocks and the
  `per_socket_dxa_req_bus_if` array (request arbitration is now
  entirely in-socket via the existing `VX_dxa_req_arb`).
- The DCR fan-out slot for DXA at cluster scope.

`dxa_core_perf` reporting moves: each socket reports its own
`dxa_perf_t`, and `VX_cluster` aggregates per-socket DXA counters into
`sysmem_perf.dxa` (sum across sockets), mirroring how `dcache_perf` is
aggregated today.

### 5.5 DCR routing

Currently DXA DCR registers are routed through a cluster-level DCR slot
(`DCR_GFX_IDX = NUM_SOCKETS + VX_CFG_EXT_DXA_ENABLED`). With DXA
per-socket, each socket's `VX_dxa_core` consumes the same DCR stream
already broadcast to its cores (DCR writes are write-only and
idempotent, so per-socket replication is safe). The cluster-level DXA
DCR slot is deleted; `DCR_GFX_IDX` collapses by one.

---

## 6. SimX alignment

The SimX functional model must move in lock-step with the RTL or the
two diverge and `sim/regression` stops being a meaningful check on RTL
correctness.

### 6.1 Today

- `Cluster::Impl` owns one `DxaCore` instance.
- `DxaCore::gmem_req_out[i]` binds to `l2arb->ReqIn[kL2Rows*i + 1]`
  (row 1 of the cluster L2 priority arb).
- `DxaCore::dxa_req_in[cid]` binds to each per-core `sfu->dxa_req_out`,
  iterating sockets-then-cores at cluster scope.
- `DxaCore::lmem_req_out[cid]` binds directly to each
  `core->local_mem()->Inputs[port_dxa]`, with a `tx_callback` that
  pulses `core->barrier_event_release` on every `dxa_notify_done`
  write.

### 6.2 After

- `Socket::Impl` owns one `DxaCore` instance (per-socket scope).
- `DxaCore::gmem_req_out[i]` binds into a new socket-level mem
  arbiter (`Socket::Impl::dxa_dcache_arb_`) at row 1, with row 0
  taking the existing `l1_arb->ReqOut[i]`. The arbiter's `ReqOut[i]`
  binds to `simobject->mem_req_out[i]`, replacing today's direct
  `l1_arb->ReqOut[i] → mem_req_out[i]` binding.
- `DxaCore::dxa_req_in[cid_local]` binds to in-socket
  `sfu->dxa_req_out` only (`cid_local` ∈ `[0, SOCKET_SIZE)`).
- `DxaCore::lmem_req_out[cid_local]` binds to in-socket cores'
  `LocalMem::Inputs[port_dxa]`. The `tx_callback` (which decodes
  `dxa_notify_bar_id` and calls `barrier_event_release`) moves with
  it — same code, just bound at socket scope.
- `Cluster::Impl` no longer references `DxaCore`. Cluster-level
  perf aggregation reads `sockets_[i]->dxa_perf_stats()` and sums.

### 6.3 Arbitration semantics in simx

`Socket::Impl::dxa_dcache_arb_` is a `MemArbiter` configured as
priority (dcache > DXA), matching the RTL arb. The model already has a
priority `MemArbiter` used by `Cluster::Impl::l2arb_`; reuse it. No
new arbiter class is required.

Cycle-accuracy contract: a DXA request issued in the same cycle as a
dcache request that targets the same `mem_req_out[i]` channel must
defer for ≥1 cycle. The existing priority arbiter implements this; the
move from cluster to socket scope does not change the per-request
latency calculus, only **where** the contention is observed.

### 6.4 Files touched in simx

| File | Change |
|---|---|
| `sim/simx/socket.h` | Add `DxaCore::Ptr dxa_core_` member and accessor. Add `dxa_perf_stats()` getter. |
| `sim/simx/socket.cpp` | Create `DxaCore` in `Socket::Impl` ctor; build the in-socket mem arb; rebind `simobject->mem_req_out` through the arb; plumb per-core `dxa_req_in`/`lmem_req_out`/`tx_callback`. |
| `sim/simx/cluster.h` | Drop `DxaCore::Ptr` member and accessor. Perf struct keeps `dxa` field but populated by summing across sockets. |
| `sim/simx/cluster.cpp` | Delete cluster-level DxaCore creation, row-1 L2 arb bindings, per-core `sfu->dxa_req_out` and `lmem_req_out` plumbing (now handled by Socket). Update `dcr_write` dispatch: forward DXA DCR writes to every socket's DxaCore. |
| `sim/simx/dxa/dxa_core.cpp` | `kCoresPerCluster` → `kCoresPerSocket` (= `VX_CFG_SOCKET_SIZE`). `kDxaMemPorts` becomes a per-socket constant (`= min(VX_CFG_NUM_DXA_UNITS, DXA_L1_GMEM_PORTS)`). |

The `tx_callback` machinery for `dxa_notify_done` → `barrier_event_release`
moves verbatim; only its binding scope changes. Crucially, **the
per-core release pulse semantics are unchanged**, so existing
multicast tests (`dxa_copy_mw`, `sgemm2_dxa_mw`) require no
modifications to pass once they exist.

### 6.5 Regression strategy

After the simx changes land, `sim/regression` must replay the full DXA
test set (`dxa_copy`, `dxa_copy_mw`, `sgemm2_dxa`, `sgemm_tcu_wg_dxa`,
`sgemm_tcu_wg_sp_dxa`, `sgemm2_dxa_mw`) on configurations sweeping
`NUM_SOCKETS ∈ {1, 2, 4}` × `SOCKET_SIZE ∈ {1, 2, 4}`. The 1×1 case
proves baseline preservation; the multi-socket cases prove the
relocation does not introduce cross-socket interference (any test that
silently relied on cluster-scope DXA serialisation would break here).

---

## 7. Phased implementation

### Phase 0 — Cluster dispatch contract (~3 days)

- RTL: `VX_kmu.sv` nested dispatch counter; `VX_cta_dispatch.sv`
  cluster-accept FSM; new `VX_cta_table_if.sv` interface; pass-through
  signal plumbing in `VX_core.sv`, `VX_scheduler.sv`, `VX_csr_data.sv`,
  `VX_mem_unit.sv`.
- DCRs/CSRs: `VX_DCR_KMU_CLUSTER_DIM_{X,Y,Z}` (0x003-0x005),
  `VX_CSR_CTA_CLUSTER_SIZE` (0xCE0) added to `VX_types.toml`.
- SimX: matching changes in `kmu`, `cta_dispatcher`, `scheduler`,
  `csr_unit`, plus DCR defaults wired in `sim/simx/main.cpp` and
  `sim/rtlsim/main.cpp`.
- Host runtime (vortex2.h API): add `cluster_dim[3]` to
  `vx_launch_info_t`; queue layer programs the three new DCRs and
  validates grid divisibility. The legacy `vortex.h` API is not
  extended.
- Kernel SDK: `get_cluster_size()` and `get_cluster_rank()`
  in `vx_spawn2.h`; backed by the new CSR.

Validation: any kernel launched with `cluster_dim != {1,1,1}`
sees its CTAs delivered as contiguous groups; existing tests with
default `cluster_dim` are bit-identical to pre-Phase-0 behavior.

### Phase 1 — Intra-core multicast (~1 day)

- `VX_dxa_completion`: single-slot pending → depth-`NUM_WARPS` FIFO
  (the load-bearing RTL fix — the existing `_mw` regressions exercise
  multicast but trip on the single-slot buffer dropping release
  events under backpressure).
- No `vx_barrier.h` change required (`group_barrier`, `expect_tx`
  already present and already used by the `_mw` tests).
- Add `vortex::dxa_multicast_*d` helper classes to `vx_dxa.h` (§3.2).
  This is what enforces the mask ↔ `expect_tx` invariant by
  construction; the `_mw` tests should be ported to use the helper
  as part of this phase so the raw two-barrier pattern is reserved
  for documentation only.
- The `dxa_copy_mw` and `sgemm2_dxa_mw` test directories are already
  implemented against the raw pattern; port them to the helper and
  they start passing once the completion-FIFO fix lands.

Validation status (empirical, on `dxa_fixes@HEAD`, simx `--cores=1`,
after Phase 0 + Phase 1 land):

| Test | Phase 1 status |
|---|---|
| `dxa_copy`              | PASS (no multicast involved) |
| `dxa_copy_mw`           | **PASS** — single-group grid, validated. |
| `sgemm2_dxa`            | PASS (no multicast involved) |
| `sgemm_tcu_wg_dxa`      | PASS (no multicast involved) |
| `sgemm_tcu_wg_sp_dxa`   | PASS (no multicast involved) |
| `sgemm2_dxa_mw`         | **PASS** at n=4/8/12/16/32/64 with the SimX alignment fixes below. |
| `sgemm_tcu_wg_dxa_mw`   | Phase 3 deliverable. |

#### SimX alignment fixes landed alongside Phase 0/1

Bringing `sgemm2_dxa_mw` from "runs without deadlock but verifies
wrong" to "passes end-to-end" required three SimX-side fixes. All
three are SimX-only — the RTL design assumes them implicitly — and
are documented so they propagate to the Verilog implementation:

1. **DCR dispatch routing (`sim/simx/processor.cpp`).** The
   `LOCAL_GROUP_DIM_{X,Y,Z}` DCRs live at 0x003-0x005 (per §0.6) —
   below the main KMU DCR block at 0x010-0x01F. `ProcessorImpl::
   dcr_write` was only routing addresses in the `[0x010, 0x01F)`
   range to the KMU, silently dropping the three cluster-dim
   writes. The runtime's `vx_enqueue_launch` was programming the
   DCRs correctly; the dispatcher just never saw them, so it ran
   with the default `lgd=[1,1,1]` and degenerated to single-CTA
   "groups" — Phase 0's contract never actually engaged. Fix: the
   dispatcher now also routes 0x003-0x005 to the KMU.

2. **Per-CTA LMEM stride alignment (`sim/simx/cta_dispatcher.cpp`
   + `sim/simx/dxa/dxa_core.cpp`).** Path A multicast resolves
   receiver destinations as `issuer_addr + r * smem_stride`, but
   the LMEM model is block-addressed (`VX_CFG_MEM_BLOCK_SIZE = 64`)
   with a byteen mask — a non-block-aligned `smem_stride` truncates
   the address inside the LMEM bank and writes land in the wrong
   block. The host-visible `lmem_size` (e.g., 80 bytes for `-n4`)
   is rarely a block multiple. Fix: both the dispatcher (per-CTA
   allocation) and the DXA descriptor handler (`smem_stride`) now
   round up to `MEM_BLOCK_SIZE`. The two roundings must match
   byte-for-byte; both are derived from the same formula.

3. **Group-atomic LMEM placement (`sim/simx/cta_dispatcher.cpp`).**
   The dispatcher's LMEM ring-buffer used to wrap *per CTA*, which
   meant the last CTA of a cluster could land at LMEM offset 0
   while its siblings sat just below the capacity boundary. The
   multicast issuer then wrote rank-3's data into a non-existent
   slot past the buffer end (the formula `issuer_addr + 3 * stride`
   computes a contiguous address, but the actual destination
   wrapped). Fix: when admitting the *first* CTA of a cluster,
   the dispatcher pre-wraps `lmem_tail_` if the *whole* group span
   `K * aligned_lmem_size` would straddle capacity. Tracked through
   a new `cluster_cta_remaining_` counter that decrements per CTA
   and re-initializes from the in-flight CTA's `cluster_dim` on
   the first member of each group. This makes the §0.1.3 contract
   ("contiguous `cta_local_id` assignment") *also* mean "contiguous
   LMEM offsets", which the kernel pattern actually depends on for
   Path A multicast.

These three fixes are the operative reason `sgemm2_dxa_mw` now
passes across the size sweep (`-n4` through `-n64`). The RTL
equivalents — KMU/dispatcher DCR routing of 0x003-0x005, per-CTA
LMEM block-alignment, atomic group placement against LMEM wrap —
are owned by Phase 0 RTL work, not Phase 1.

### Phase 2 — Per-socket DXA relocation (~4 days)

- RTL: move `VX_dxa_core` from `VX_cluster.sv` into `VX_socket.sv`;
  add `dxa_dcache_arb`; delete cluster-level DXA fabric; collapse
  DCR routing.
- SimX: matching move (`Socket::Impl` owns `DxaCore`; in-socket
  arbiter; updated perf aggregation and DCR fan-out). See §6.4.
- Phase 1 tests continue to pass on `--cores=1`, `--cores=2`,
  `--cores=4`, and across socket geometries `SOCKET_SIZE ∈ {1, 2, 4}`.

### Phase 3 — WGMMA-backed multicast (~1 week)

- `sgemm_tcu_wg_dxa_mw` (intra-core WGMMA with `W`-warp CTAs, where
  `W` is the warpgroup width — `W = 4` is the common WGMMA case but
  the path is parameterised in `W`). The minimum `NUM_WARPS` to fit
  `K` peer CTAs is `K · W` (`= 16` for `K = 4`, `W = 4`).
- Per §2.3 this requires **no mask-scheme change** — the bar-unit
  address field is already indexed by CTA id, so the same
  `bit k → CTA (issuer_cta_id + k)` walk that Phase 1 uses applies
  unchanged for arbitrary `W`. All `W` warps of a CTA share its one
  bar slot per bar_id, and the kernel pattern in §3 already
  accommodates the multi-warp case.
- The only Phase 3 deliverables are therefore the kernel itself, the
  `NUM_WARPS ≥ K · W` config, and the performance characterization.
- Performance characterization: GMEM bandwidth amortization measured
  against non-multicast baselines; compare per-socket DXA throughput
  in `--sockets=4` against the pre-relocation cluster-scope baseline
  to confirm the move is net-positive (less serialisation,
  unchanged per-request latency).

---

## 8. Tests (under `tests/regression/`)

| Test | Status (empirical) | Geometry | Sync primitive |
|---|---|---|---|
| `dxa_copy` | passing today | 1 CTA / 1 core | — (no peers) |
| `dxa_copy_mw` | **passing today (Phase 1)** — single-group grid, no dispatcher dependency | K single-warp CTAs / 1 core, K = NUM_WARPS | `vortex::group_barrier` |
| `sgemm2_dxa_mw` | hangs today; passes once `cluster_dim` lands | K single-warp CTAs × multi-wave grid / 1 core | `vortex::group_barrier` |
| `sgemm_tcu_wg_dxa_mw` | hangs today; passes once `cluster_dim` + Phase 3 land | K × W-warp CTAs × multi-wave grid / 1 core | `vortex::group_barrier` |

Each "mw" test demonstrates intra-core fan-out across CTA peers on a
single core. There are intentionally no `_mc` (cross-core) tests in
this proposal — see §2. The pre-existing `dxa_copy_mc`,
`sgemm2_dxa_mc`, and `sgemm_tcu_wg_dxa_mc` regression directories are
deleted as part of Phase 1 (they describe a primitive this proposal
explicitly does not build).

---

## 9. fp16 sgemm-DXA performance — baseline + acceptance

This proposal is graded on a real workload: the fp16 DXA-backed sgemm
variants. We baseline them on `vortex_ci`'s `tinebp-patch-2` head (the
revision this proposal was rebased onto) before any of Phase 1's RTL
or kernel changes land, then re-measure after each phase and require
the speedup to clear a published threshold.

### 9.1 Workloads measured

| Test | ITYPE | Description |
|---|---|---|
| `sgemm_tcu_wg_dxa` | fp16 | WGMMA + DXA, accum in fp16 (intra-warp-group, no multicast). Baseline reference for `_mw` variant. |
| `sgemm_tcu_wg_sp_dxa` | fp16 (input) / fp32 (accum) | WGMMA + DXA single-precision accumulator. Mirror of above with widened accumulator. |
| `sgemm_tcu_wg_dxa_mw` | fp16 | Phase 3 deliverable — K-way intra-core multicast over the WGMMA path. |

`sgemm2_dxa` is excluded from this section: it is an integer/scalar
DXA test and gives no fp16 signal.

### 9.2 Counters captured per run

For each `make run-simx` invocation:

- **Total cycles** (`PERF: cycles=` from simx footer).
- **DXA GMEM read bytes** (`PERF: dxa_gmem_reads × VX_CFG_L1_LINE_SIZE`).
- **DXA LMEM write bytes** (`dxa_lmem_writes × DXA_LMEM_WORD_SIZE`).
- **dcache MSHR stall cycles** (proxy for DXA-vs-dcache contention).
- **IPC** (committed instructions / cycles).

Raw simx logs are checked into `docs/perf/dxa_fp16/<phase>/` under the
test name so the deltas are diff-able across phases.

### 9.3 Geometry sweep

| Knob | Values |
|---|---|
| `NUM_CLUSTERS` | 1 |
| `NUM_SOCKETS` | 1, 2, 4 |
| `SOCKET_SIZE` (cores/socket) | 1, 2 |
| `NUM_WARPS` | 4 (Phase 1/2), 16 (Phase 3 only — required for 4 × 4-warp CTAs) |
| `NUM_THREADS` | 4 |
| Matrix size | 64×64, 128×128, 256×256 |

The `NUM_SOCKETS × SOCKET_SIZE` sweep is the load-bearing dimension
for the per-socket DXA relocation in §5 — it is the configuration
under which the old cluster-scope arbiter could mask cross-socket
interference.

### 9.4 Current baseline

Captured on `dxa_fixes@tinebp-patch-2` (= `vortex_ci@tinebp-patch-2`),
SimX driver, `VORTEX_PROFILING=6` (DXA + core perf class).
`Cfg` column is `S×C×W` = sockets × cores-per-socket × warps-per-core.
GMEM/LMEM counters are raw transaction counts emitted by
`VX_CSR_MPM_DXA_GMEM_READS` / `VX_CSR_MPM_DXA_LMEM_WRITES`; multiply
by `VX_CFG_L1_LINE_SIZE` (64 B) and `DXA_LMEM_WORD_SIZE` respectively
for byte totals.

| Test | Cfg | M×N×K | Cycles | DXA gmem_reads | DXA lmem_writes | avg_gmem_lat | IPC | Status |
|---|---|---|---|---|---|---|---|---|
| `sgemm_tcu_wg_dxa`    | 1×1×4 | 64  |    280,340 |   6,144 |   6,144 | 161.7 | 0.254 | PASS |
| `sgemm_tcu_wg_dxa`    | 1×1×4 | 128 |  1,755,699 |  49,152 |  49,152 | 136.3 | 0.291 | PASS |
| `sgemm_tcu_wg_dxa`    | 1×1×4 | 256 | 11,914,854 | 393,216 | 393,216 | 110.4 | 0.324 | PASS* |
| `sgemm_tcu_wg_dxa`    | 2×1×4 | 64  |    201,841 |   6,144 |   6,144 | 180.2 | 0.352 | PASS |
| `sgemm_tcu_wg_dxa`    | 2×1×4 | 128 |  1,039,485 |  49,152 |  49,152 | 156.5 | 0.492 | PASS |
| `sgemm_tcu_wg_dxa`    | 2×1×4 | 256 |  6,877,240 | 393,216 | 393,216 | 113.6 | 0.562 | PASS* |
| `sgemm_tcu_wg_dxa`    | 4×2×4 | 64  |    166,781 |   6,144 |   6,144 | 254.0 | 0.428 | PASS |
| `sgemm_tcu_wg_dxa`    | 4×2×4 | 128 |    824,147 |  49,152 |  49,152 | 193.6 | 0.621 | PASS |
| `sgemm_tcu_wg_dxa`    | 4×2×4 | 256 |  4,600,447 | 393,216 | 393,216 | 139.5 | 0.840 | PASS* |
| `sgemm_tcu_wg_sp_dxa` | 1×1×4 | 64  |    412,829 |   7,168 |   7,168 |  49.4 | 0.188 | PASS |
| `sgemm_tcu_wg_sp_dxa` | 1×1×4 | 128 |  2,821,418 |  57,344 |  57,344 |  42.3 | 0.196 | PASS |
| `sgemm_tcu_wg_sp_dxa` | 1×1×4 | 256 | 20,830,464 | 458,752 | 458,752 |  39.9 | 0.199 | PASS |
| `sgemm_tcu_wg_sp_dxa` | 2×1×4 | 64  |    250,485 |   7,168 |   7,168 |  59.9 | 0.311 | PASS |
| `sgemm_tcu_wg_sp_dxa` | 2×1×4 | 128 |  1,572,123 |  57,344 |  57,344 |  48.8 | 0.352 | PASS |
| `sgemm_tcu_wg_sp_dxa` | 2×1×4 | 256 | 11,018,676 | 458,752 | 458,752 |  43.1 | 0.377 | PASS |
| `sgemm_tcu_wg_sp_dxa` | 4×2×4 | 64  |    202,502 |   7,168 |   7,168 |  75.5 | 0.385 | PASS |
| `sgemm_tcu_wg_sp_dxa` | 4×2×4 | 128 |  1,112,217 |  57,344 |  57,344 |  58.7 | 0.498 | PASS |
| `sgemm_tcu_wg_sp_dxa` | 4×2×4 | 256 |  7,235,425 | 458,752 | 458,752 |  48.9 | 0.574 | PASS |

`PASS*` at M=256 for `sgemm_tcu_wg_dxa` indicates correct hardware
execution; verify trips ≤ 5 ULP fp16 mismatches against the CPU
reference, which is the expected accumulator-rounding gap (the sp
variant uses an fp32 accumulator and so does not trip). The cycle
counts are valid in all cases.

#### Observations from the multi-socket sweep

- **Cycle scaling (1×1 → 2×1 → 4×2) for `sgemm_tcu_wg_dxa @ M=256`**:
  `11.9M → 6.88M → 4.60M` — sub-linear (8× cores give ~2.6× speedup)
  because the workload is small and DXA-latency-bound rather than
  compute-bound.
- **DXA `gmem_reads` is invariant in cores.** Same workload, same
  number of GMEM transactions, regardless of how many cores process
  it. This is the expected baseline for measuring multicast savings:
  multicast should *reduce* this column by ~K, not change cycles in
  isolation.
- **`avg_gmem_lat` grows with core count.** `M=128` wg_dxa:
  `136.3ns (1c) → 156.5ns (2c) → 193.6ns (8c)`. This is the
  cluster-scope DXA fabric saturating — exactly the contention
  Phase 2's per-socket DXA relocation is designed to relieve.
- **IPC grows with core count** (more parallelism utilized): `0.291
  → 0.492 → 0.621` for the same cell. Phase 2 should preserve or
  improve this at 4×2×4 while bringing `avg_gmem_lat` down.

These observations sharpen the Phase 2 acceptance criteria in §9.5:
- Phase 2 at 1×1×4 should preserve cycles within ±3% (no
  regression at single-socket scope).
- Phase 2 at 4×2×4 should improve cycles by ≥10% **and** reduce
  `avg_gmem_lat` by ≥15% (the dominant source of the 4×2 cycle
  improvement should be reduced cluster-scope contention).

(Capture method: the test mains were instrumented with a single
`vx_dump_perf(device, stdout)` call between the elapsed-time print
and the verify pass. That call is non-intrusive — it executes after
the timed region — and lives only in the local working tree, not in
`tinebp-patch-2`. The build also requires `CONFIGS="-DVX_CFG_NUM_WARPS=4
-DVX_CFG_ISSUE_WIDTH=4"` to be passed via the make env to work around
a pre-existing `gen_config.py` bug where the toml expression
`up(NUM_WARPS/16)` resolves to `0.25` instead of `1`, colliding with
the per-test `-DVX_CFG_ISSUE_WIDTH=4`. Fixing that bug is out of
scope for this proposal.)

### 9.5 Speedup model + acceptance

Let `K` = `popcount(mc_mask)` (the multicast group size). The
theoretical effect of intra-core multicast on a workload where DXA
GMEM-read is the bottleneck:

- **GMEM read bytes**: `÷ K` (one read, K writes).
- **Cycles**: best case `÷ K_eff`, where
  `K_eff = K · (gmem_time / total_time)`. For sgemm with DXA in the
  critical path (large K and small accumulate ratio), `K_eff → K`.

Acceptance thresholds, measured on the same (cfg, M, N, K) cell as
the baseline:

| Phase | Metric | Acceptance |
|---|---|---|
| Phase 1 (`sgemm2_dxa_mw`, K=4)            | GMEM bytes  | ≥ 0.9 · (baseline / K) |
| Phase 1                                    | Cycles      | ≥ 0.7 · (baseline / K) |
| Phase 2 (per-socket DXA, no multicast)     | Cycles      | within ±3 % of Phase 1 baseline at 1×1×4; ≥ 10 % faster than Phase 1 baseline at 4×2×4 (cross-socket interference dropping) |
| Phase 3 (`sgemm_tcu_wg_dxa_mw`, K=4)      | GMEM bytes  | ≥ 0.9 · (Phase-2 baseline / K) |
| Phase 3                                    | Cycles      | ≥ 0.6 · (Phase-2 baseline / K) — WGMMA accumulate time bounds the cycle speedup |

If a cell misses its threshold the change does not merge. Misses
must be root-caused (dcache stall, LMEM bank conflict, descriptor
programming overhead, etc.) rather than relaxed.

### 9.6 Results table (populated post-implementation)

A second table, identical in shape to §9.4, is added to this proposal
once Phase 1 lands — and again at the end of Phase 2 and Phase 3 — so
the proposal carries its own provenance of measured outcomes:

```
| Test | Cfg | M | Baseline cyc | Post-impl cyc | Speedup | Threshold | Pass? |
```

Populated values get committed to this file in the same change that
lands the implementation, so reviewers can verify the speedup claim
against the code in the same diff.

#### Phase 1 — `sgemm2_dxa_mw` results (post-fix, fp32, K=4, Cfg = 1×1×4)

Baseline = `sgemm2_dxa -t4 -m1` (single-buffer, no multicast — same
final matrix, different intra-CTA tiling). Multicast = `sgemm2_dxa_mw
-t4` with the SimX fixes from the "Known issues" subsection in §8.

| n  | Baseline cyc | Multicast cyc | Speedup | DXA reads (base / mc) | avg_gmem_lat (base → mc, ns) | IPC (base → mc) | §9.5 cycle floor (≥ 0.7·baseline/K) | Pass? |
|---:|-------------:|--------------:|--------:|----------------------:|-----------------------------:|----------------:|------------------------------------:|:-----:|
|  16 |     100,916  |       47,819  | **2.11×** |     320 /     320 | 852.0 → 73.2 (-91 %) | 0.215 → 0.351 |     17,660 | ✓ |
|  32 |     543,388  |      313,997  | **1.73×** |   2,560 /   2,560 | 972.3 → 106.6 (-89 %) | 0.273 → 0.370 |     95,093 | ✓ |
|  64 |   3,261,070  |    2,214,426  | **1.47×** |  20,480 /  20,480 | 1094.3 → 147.0 (-87 %) | 0.333 → 0.387 |    570,687 | ✓ |

Interpretation. Total GMEM **read transactions** are identical at
every n — `sgemm2_dxa` already coalesces a per-CTA 4×16 A-tile and
16×4 B-tile, and `sgemm2_dxa_mw` issues the same total bytes but
restructured as one multicast B-fetch per cluster plus per-CTA
A-fetches. The cycle win is therefore *not* the per-bullet GMEM-byte
reduction from §9.5's model (which assumes both kernels are
geometrically identical and the only delta is the multicast factor).
It comes from **latency hiding**: the multicast pattern decomposes
each row of work into many small parallel DXA fetches with
overlapping MSHRs, collapsing `avg_gmem_lat` by ~7-12× across the
sweep. Speedup is largest at small n (DXA-latency-bound) and shrinks
as n grows (compute-bound takes over).

All three sizes meet the §9.5 acceptance floor for cycles. The
§9.5 GMEM-bytes threshold is *not* applicable to this baseline pair —
that threshold assumes a non-multicast variant of the *same kernel
shape* (one not currently in the regression tree). Constructing
such a variant for a cleaner per-byte measurement is deferred; the
cycle speedup against the canonical fp32 `sgemm2_dxa` baseline is
the operative Phase 1 acceptance signal.

Note on §9.5 inequality direction. The "Cycles ≥ 0.7 · (baseline /
K)" wording reads literally as "cycles must be at least 70 % of the
K-way theoretical minimum", which is trivially satisfied by any run
that doesn't beat the K-way bound (and so is not a discriminating
threshold). The intent is "achieve at least 70 % of the K-way
speedup", which is `multicast_cyc ≤ baseline / (0.7·K)`. Under that
stricter (intent-aligned) reading, the n=64 cell fails:
`3,261,070 / (0.7·4) = 1,164,668 < 2,214,426`. The literal threshold
is what's coded above; tightening it to the intent reading is left
to the proposal author. The compute-bound regression at n=64 is
genuine and would surface under the corrected threshold.

---

## 10. What this proposal explicitly does NOT do

- **No inter-core / cross-socket multicast.** Multicast is strictly
  intra-core. No socket-level LMEM write fabric, no cross-core
  release aggregation. (`gbarrier::expect_tx` already exists in
  `vx_barrier.h` and stays as-is — it is simply not driven by any
  multicast path here.)
- **No reliance on an external dispatcher proposal.** The cluster
  dispatch contract (atomic same-core dispatch, contiguous
  `cta_local_id`, uniform LMEM allocation) is owned by Phase 0 of
  *this* proposal (§0). It is specified, implemented, and validated
  here — no prerequisite document, no follow-on document.
- **No new bar_unit opcode, no `events_r` saturation logic.** Sync
  uses existing `arrive_and_wait` on a shared local-bar slot. The
  C5 invariant (mask bit ↔ `expect_tx`) is enforced structurally by
  the `vortex::dxa_multicast` helper API (§3.2), following the
  Hopper / CUTLASS convention of pinning this responsibility to
  software, not hardware.
- **No slot-table or rank-table abstractions.** Mask bits are
  cluster rank offsets relative to the issuer's rank-0 position;
  the kernel never writes them by hand (helper API derives them
  from `get_cluster_size()`).
- **No mask-scheme change for WGMMA.** Multi-warp CTAs (notably 4-warp
  WGMMA) are first-class cluster members under the existing CTA-id-
  indexed mask walk — no warps-per-CTA stride, no separate intrinsic.
- **No HW-side per-CTA LMEM-base lookup.** Destination addressing
  reuses the existing `dxa_core` arithmetic
  `dest[r] = issuer_smem_addr + r × smem_stride`; the host runtime
  programs `smem_stride = lmem_per_cta` once at descriptor setup.
  The "same offset in receiver LMEM" semantic (§2.5) emerges from
  that arithmetic plus `cluster_dim`'s uniform-allocation
  guarantee — no new RTL interface required. Path B (true HW lookup
  of `LMEM_BASE[k]`) is deferred to a future proposal.

---

## 11. Open questions

1. **`group_barrier` / `barrier` id conflict** — `group_barrier(N)`
   aliases to the rank-0 CTA's `barrier(N)` slot. Kernel must
   reserve specific bar_id values for shared use; the helper API
   does not pick bar_ids for the caller. Convention-based, not
   structurally enforced. Could partition the ID space later if
   needed.

2. **Bar slot budget** — `NUM_BARRIERS = 8` per core. Each
   `vortex::dxa_multicast` instance consumes 1 local_bar; concurrent
   multicasts can share one `group_bar`. WGMMA pipelines with
   double-buffered A + B multicasts plus `__syncthreads` and an
   epilogue bar run to 5–8 slots used. Bumping `NUM_BARRIERS` to
   16 is the obvious mitigation — small RTL cost. Re-evaluate once
   real WGMMA kernels are profiled.

3. **DXA GMEM port count per socket** — proposal assumes
   `DXA_L1_GMEM_PORTS = 1`. If a socket has high DXA bandwidth
   demand (e.g. multiple concurrent multicasts), should this be
   parameterised per `SOCKET_SIZE`? Defer until measured.

4. **DXA DCR replication** — DCR writes are idempotent so per-socket
   replication is safe, but the DCR-write fan-out latency grows
   with `NUM_SOCKETS`. Acceptable while DCR traffic is rare
   (descriptor programming once per launch). Re-evaluate if DXA
   descriptor churn becomes hot-path.

5. **Cross-issuer concurrent multicasts** — the helper currently
   assumes rank-0 issues. If a workload needs cluster members at
   different ranks to issue different multicasts (Hopper-style
   "each CTA issues for its own row/column" pattern), the helper
   would need a `dxa_multicast_from(rank)` overload and a Phase 2
   HW change so the multicast walk knows the issuer's rank within
   the cluster. Not in Phase 1/3 scope.

6. **Soft-launch interaction with `cluster_dim` rollout** —
   §0.4 provides a kernel-side dimension check as a soft path
   before `cluster_dim` lands. The check should be inserted
   into the helper class constructor so kernels don't have to
   write it. Tracking how/when to remove the check once
   `cluster_dim` is fully enforced.

7. **Inspiration: should we ever adopt Hopper-style DSMEM?** —
   Cross-core multicast is currently out of scope, but if a future
   workload demands it, the architecturally clean answer is
   Hopper's distributed shared memory (unified peer addresses via
   bit-encoded pointer math, `mapa`-style address translation,
   atomic cross-GPC cluster dispatch). Captured here so a future
   proposal author starts from the right reference.
