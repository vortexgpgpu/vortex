# DXA Multicast — Proposal

**Date:** 2026-05-12
**Status:** Draft
**Owners:** RTL team
**Related:**
[dxa_worker_rtl_redesign_proposal.md](dxa_worker_rtl_redesign_proposal.md),
[VX_bar_unit.sv](../../hw/rtl/core/VX_bar_unit.sv),
[VX_gbar_unit.sv](../../hw/rtl/mem/VX_gbar_unit.sv),
[VX_dxa_smem_wr.sv](../../hw/rtl/dxa/VX_dxa_smem_wr.sv),
[vx_barrier.h](../../sw/kernel/include/vx_barrier.h).

---

## Summary

Make DXA multicast a first-class primitive built on **Vortex's existing local
and global barrier infrastructure**, with the minimum possible RTL surface
area. The barrier handle's scope bit selects the mode:

- **Local bar → intra-core multicast.** Mask bits index warp positions
  relative to the issuer's wid. All addressed CTAs must be co-resident on
  the issuer's core.
- **Global bar → inter-core multicast.** Mask bits index core IDs within
  the socket. Each addressed core must have one CTA waiting on the gbar.

Synchronization across receivers (every CTA must have called `expect_tx`
before the issuer fires multicast) is solved in software with a
**two-barrier idiom** — no new hardware barrier opcode.

**Hard constraints, by design:**

1. **Zero changes to `VX_cta_dispatch`.** No slot tables, no rank metadata,
   no admission gates. Co-residence is the kernel's responsibility, surfaced
   as a deterministic deadlock at the sync barrier when the kernel asks for
   an impossible configuration.
2. **Mask bits encode warps (intra) or cores (inter).** Bit `k` targets
   `(issuer_wid + k)` for local mode and core `k` for global mode. Multi-warp
   CTAs are not supported as multicast peers in Phase 1 (the dispatcher
   allocates contiguous wids per CTA, so the next bar-holder is not at
   `+1`); single-warp CTAs are required for intra-core multicast.
3. **Reuse all existing barrier infrastructure.** No new `bar_unit` opcode,
   no new state machine, no new flag bit. Sync uses the existing
   `arrive_and_wait` semantic on a shared bar slot.

---

## 1. What changes, by file

| File | Change | Approx LoC |
|---|---|---|
| [VX_dxa_smem_wr.sv](../../hw/rtl/dxa/VX_dxa_smem_wr.sv) | Multicast `bar_addr` walk stays `active_bar_addr + (replay_next_idx << NB_BITS)` (pre-bce88c86 form). The walk handles both local-bar and gbar scope: the issuer's `active_bar_addr` `is_global` bit selects the release routing path inherited from `VX_bar_unit`. | 0 (current code) |
| [VX_dxa_completion.sv](../../hw/rtl/dxa/VX_dxa_completion.sv) | Replace single-slot pending buffer with a depth-`NUM_WARPS` FIFO. Multicast fires up to `popcount(mask)` back-to-back release events on the last drain word; the existing single-slot buffer drops events under downstream backpressure. | ~30 |
| [vx_barrier.h](../../sw/kernel/include/vx_barrier.h) | Add `vortex::shared_barrier` class (local bar with shared `bar_addr` across CTAs on a core) + `gbarrier::expect_tx` method. | ~40 |
| **New (Phase 2)** — `hw/rtl/mem/VX_socket_lmem_bus_if.sv`, `hw/rtl/mem/VX_socket_lmem_arb.sv` | Socket-level cross-core LMEM write bus for inter-core multicast SMEM fan-out. **The principal hardware lift; everything else above is small.** | ~200 |

The Phase 1 (intra-core only) RTL surface is essentially just the
`VX_dxa_completion` FIFO fix — about 30 lines. The rest of the work is in
software. Phase 2 (inter-core) adds the cross-core LMEM fabric.

---

## 2. Mask semantics

### Intra-core (local bar)

- Mask is `NUM_WARPS` bits wide.
- Bit `k` set ⇒ release event fires at `bar_addr = active_bar_addr + (k << NB_BITS)` ⇒ wid `(issuer_wid + k)` receives.
- SMEM write target = `issuer_smem_addr + k * smem_stride_words` (kernel-supplied stride via `vx_dxa_program_desc_multicast`).
- **Requires**: single-warp CTAs (so wid `(issuer_wid + k)` is CTA `(issuer_cta + k)`'s bar holder).
- **Co-residence**: all `popcount(mask)` CTAs must be dispatched on the issuer's core before the issuer fires the multicast. The kernel sync barrier (see §3) enforces this — issuer cannot proceed until all peers arrive.

### Inter-core (global bar)

- Mask is `NUM_CORES` bits wide.
- Bit `k` set ⇒ core `k` participates. SMEM write target = `(core_id=k, lmem_offset)` via the socket-level cross-core LMEM bus.
- Each receiving core's `dxa_completion` generates a local `txbar_bus` event with `bar_addr = {is_global=1, gbar_id}` → core's `bar_unit` decrements that core's gbar `events_r` → when `events_r==0`, the core notifies `VX_gbar_unit` → cross-core release aggregation completes.
- **Requires**: one CTA per participating core, all waiting on the same gbar id.
- **Co-residence**: gbar aliases naturally across cores; the cross-core sync gbar (see §3) blocks until every participating core's CTA has called `arrive_and_wait`.

---

## 3. The two-barrier idiom

The race we're solving: multicast issuer fires before some receivers have
called `expect_tx` on their bars; the release events land on bars with
`events_r=0` and are silently dropped (or wrap-decrement).

The fix is structural — insert a **sync barrier** between the receivers'
`expect_tx` and the issuer's multicast issue.

### Intra-core kernel pattern

```cpp
// Per-CTA event bar — receives the multicast release on this CTA's slot.
vortex::barrier        evt_bar(0);
// Shared sync bar — all peer CTAs see the same bar_unit slot.
vortex::shared_barrier sync_bar(1, num_peers);

if (is_dxa_warp) {
    evt_bar.expect_tx(1);           // register event on my private slot
    sync_bar.arrive_and_wait();     // wait for all peers to also prime
    if (cta_id == 0) {
        vx_dxa_issue_2d_multicast_wg(desc, evt_bar.id(), shmem, x, y, mc_mask);
    }
}
evt_bar.arrive_and_wait();          // wait for my DXA release event
```

### Inter-core kernel pattern

Swap `shared_barrier` for `gbarrier` (gbar already aliases across cores by
construction):

```cpp
vortex::gbarrier evt_gbar (0);  // per-core event slot
vortex::gbarrier sync_gbar(1);  // cross-core sync

if (is_dxa_warp) {
    evt_gbar.expect_tx(1);
    sync_gbar.arrive_and_wait();
    if (my_core == 0) {
        vx_dxa_issue_2d_multicast_wg(desc, evt_gbar.id(), shmem, x, y, core_mask);
    }
}
evt_gbar.arrive_and_wait();
```

### The `vortex::shared_barrier` helper

The existing `vortex::barrier(N)` encodes `bar_addr = {wid=cta_id, bar_id=N}`
— each CTA gets a different bar slot, so two CTAs constructing `barrier(N)`
cannot rendezvous through it.

`shared_barrier(N, num_peers)` instead encodes `bar_addr = {wid=0, bar_id=N}`
— a fixed slot from every CTA's perspective. All peers calling
`arrive_and_wait` on this slot hit the same `bar_unit` state and sync
correctly.

### Conflict rule (kernel discipline)

`shared_barrier(N)` and CTA 0's `barrier(N)` map to the same physical slot
(both `{wid=0, bar_id=N}`). The kernel must therefore reserve specific
`bar_id` values for shared use and never construct a per-CTA `barrier(N)`
with those same values. The convention in our tests:

| Use | bar_id |
|---|---|
| Per-CTA event bar | 0 (A) and 1 (B in WGMMA variants) |
| Shared sync bar | 2 |

The conflict isn't structurally enforced; it's a programmer convention
documented at the `shared_barrier` declaration site.

### Why no atomic op?

A fused "register events + sync" instruction would close one micro-race: if
the multicast issuer somehow fired between the receiver's `expect_tx` and
its sync arrival. But program order + the `volatile` asm in the intrinsics
prevent reordering, and the multicast instruction sits on the issuer's path
*after* its own sync arrival. There is no realistic scenario where the
issuer's fire interleaves with another receiver's pre-sync window.

The two-barrier idiom is sufficient. An atomic `expect_tx_sync` would be a
micro-optimization (one less instruction per multicast), not a correctness
requirement.

---

## 4. Co-residence as a SW constraint

This proposal explicitly chooses **deterministic deadlock at the sync
barrier** over silent corruption when the kernel asks for an impossible
multicast layout.

### Intra-core constraints

For a multicast group of K CTAs co-resident on one core:

- `K × warps_per_CTA ≤ NUM_WARPS_per_core`
- Single-warp CTAs are required for Phase 1 (so wid `(issuer_wid + k)` is CTA `(issuer_cta + k)`'s bar holder).
- Practical block_dim for sgemm-style tests: `(NUM_THREADS, 1)` — one warp per CTA.

If violated: `sync_bar.arrive_and_wait()` blocks indefinitely because peers
cannot be dispatched. The kernel deadlocks visibly (scheduler timeout). This
is the correct failure mode — surfaces the SW design error.

### Inter-core constraints

- `mc_group_size ≤ NUM_CORES`
- KMU dispatches CTAs round-robin across cores; the kernel's grid layout determines which `mc_group_size` CTAs land on participating cores.

---

## 5. Phased implementation

### Phase 1 — Intra-core multicast (~1 day)

- `VX_dxa_completion`: single-slot pending → depth-`NUM_WARPS` FIFO.
- `vortex::shared_barrier` + `vortex::gbarrier::expect_tx` added to `vx_barrier.h`.
- Rename `dxa_multicast` → `dxa_copy_mw`; rewrite as intra-core multicast copy.
- Add `sgemm2_dxa_mw` regression.

Validation: `dxa_copy`, `dxa_copy_mw`, `sgemm2_dxa`, `sgemm_tcu_wg_dxa`,
`sgemm_tcu_wg_sp_dxa`, `sgemm2_dxa_mw` all pass on `--cores=1`.

### Phase 2 — Inter-core multicast fabric (~2 weeks)

The principal hardware lift. Adds:

- `VX_socket_lmem_bus_if` — socket-level cross-core LMEM write bus.
- `VX_socket_lmem_arb` — demux by `target_core` to each core's LMEM bus.
- Each core's `VX_dxa_completion` recognizes inter-core write attrs and emits gbar release events.
- Tests already drafted (`dxa_copy_mc`, `sgemm2_dxa_mc`) using `vortex::gbarrier` for both event and sync.

Validation: `dxa_copy_mc`, `sgemm2_dxa_mc` pass on `--cores=4`. All Phase 1 tests continue.

### Phase 3 — WGMMA-backed multicast (~1 week)

- `sgemm_tcu_wg_dxa_mw` (intra-core WGMMA, requires `NUM_WARPS=16` to fit `4 CTAs × 4 warps`).
- `sgemm_tcu_wg_dxa_mc` (inter-core WGMMA, requires `NUM_CORES=4`).
- Performance characterization: GMEM bandwidth amortization measured against non-multicast baselines.

---

## 6. Tests (under `tests/regression/`)

| Test | Scope | Geometry | Sync primitive |
|---|---|---|---|
| `dxa_copy` | regression | 1 CTA / 1 core | — (no peers) |
| `dxa_copy_mw` | new (Phase 1) | K single-warp CTAs / 1 core | `vortex::shared_barrier` |
| `dxa_copy_mc` | new (Phase 2) | 1 CTA / K cores | `vortex::gbarrier` |
| `sgemm2_dxa_mw` | new (Phase 1) | K single-warp CTAs / 1 core | `vortex::shared_barrier` |
| `sgemm2_dxa_mc` | new (Phase 2) | 1 CTA / K cores | `vortex::gbarrier` |
| `sgemm_tcu_wg_dxa_mw` | new (Phase 3) | K × 4-warp CTAs / 1 core | `vortex::shared_barrier` |
| `sgemm_tcu_wg_dxa_mc` | new (Phase 3) | 1 × 4-warp CTA / K cores | `vortex::gbarrier` |

Each "mw" test demonstrates the intra-core fan-out across CTA peers on a
single core. Each "mc" test demonstrates the inter-core fan-out across
cores within a socket. The WGMMA variants demonstrate the bandwidth-
amortization value in a real GEMM workload.

---

## 7. What this proposal explicitly does NOT do

- **No dispatcher changes.** Co-residence is the kernel's responsibility,
  enforced via deterministic deadlock at the sync barrier.
- **No new bar_unit opcode.** Sync uses existing `arrive_and_wait` on a
  shared bar slot.
- **No slot-table or rank-table abstractions.** Mask bits are raw wid/core
  offsets relative to the issuer.
- **No multi-warp CTAs in intra-core multicast** (Phase 1). Single-warp CTA
  constraint is documented and surfaced via test geometry.
- **No cross-socket multicast.** Inter-core is socket-scoped, matching
  NVIDIA's GPC-scoped cluster model. Cross-socket is out of scope.

---

## 8. Open questions

1. **`shared_barrier` / `barrier` id conflict** — `shared_barrier(N)` aliases
   to CTA 0's `barrier(N)` slot. Kernel must reserve specific bar_id values
   for shared use. Convention-based, not structurally enforced. Could
   partition the ID space later if needed.

2. **Bar slot budget** — `NUM_BARRIERS = 8` per core. Each multicast group
   needs at least 2 bar IDs (event + sync). For complex WGMMA kernels with
   multiple multicast groups + double-buffering, budget gets tight. May need
   to bump `NUM_BARRIERS` if kernels grow.

3. **Failure detection for unmet co-residence** — currently surfaces as
   scheduler timeout. Could add a SW-side `__assert_multicast_coherence`
   that the kernel can call to validate `popcount(mask) × warps_per_CTA ≤
   NUM_WARPS` at runtime. Nice-to-have, not load-bearing.
