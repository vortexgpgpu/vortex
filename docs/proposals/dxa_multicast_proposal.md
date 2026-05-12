# DXA Multicast — Proposal

**Date:** 2026-05-12
**Status:** Draft
**Owners:** RTL team
**Related:**
[dxa_worker_rtl_redesign_proposal.md](dxa_worker_rtl_redesign_proposal.md),
[VX_bar_unit.sv](../../hw/rtl/core/VX_bar_unit.sv),
[VX_gbar_unit.sv](../../hw/rtl/mem/VX_gbar_unit.sv),
[VX_cta_dispatch.sv](../../hw/rtl/core/VX_cta_dispatch.sv),
[VX_dxa_smem_wr.sv](../../hw/rtl/dxa/VX_dxa_smem_wr.sv).

---

## Summary

Make DXA multicast a first-class primitive that reuses Vortex's existing
**local** and **global** barrier infrastructure as its synchronization fabric.
The barrier handle's scope bit selects the multicast mode: a **local bar**
multicasts to co-resident CTAs within one core (intra-core), and a **global
bar** multicasts to peer CTAs across cores in a socket (inter-core). The
multicast mask becomes a **CTA-position mask** interpreted in the bar's scope,
replacing today's raw-wid mask.

Three observable changes:

- **Mask semantics**: `dxa_multicast_mask` bits index CTA dispatcher slots
  (intra-core) or core IDs (inter-core); no more raw-wid encoding.
- **Routing tables**: `VX_cta_dispatch` exports `slot → (wid_base, lmem_base)`
  so the DXA worker can fan SMEM writes and barrier releases to the correct
  per-CTA destinations without the kernel computing strides.
- **Cross-core LMEM bus**: a new socket-level write fabric carries inter-core
  SMEM writes to peer cores' LMEM, alongside the existing gbar-bus path for
  release events.

The receiver-side `expect_tx` primitive ([sw/kernel/include/vx_barrier.h:49](../../sw/kernel/include/vx_barrier.h#L49))
is reused unchanged. Co-residence is enforced by the kernel's barrier-wait
pattern in the intra-core case and by gbar's existing cross-core aggregation
in the inter-core case — no new admission gate.

---

## 1. Constraints (load-bearing)

1. **Reuse the existing barrier fabric.** No new sync primitive. Local barriers
   stay in [VX_bar_unit.sv:75-122](../../hw/rtl/core/VX_bar_unit.sv#L75-L122);
   global barriers stay on `gbar_bus_if` /
   [VX_gbar_unit.sv](../../hw/rtl/mem/VX_gbar_unit.sv).
2. **One DXA multicast intrinsic** for both modes — the bar handle determines
   scope. The kernel does not pick a "mode" separately.
3. **Kernel does not compute physical strides.** No `smem_stride` or
   `bar_stride` in the descriptor. Per-receiver LMEM bases come from the
   dispatcher's slot table; per-receiver wid bases come from the same table.
4. **`libs/` is off-limits.** All edits land in `hw/rtl/dxa/`,
   `hw/rtl/core/VX_cta_dispatch.sv`, `hw/rtl/core/VX_bar_unit.sv`, and
   the new socket-level LMEM bus interface under `hw/rtl/mem/`.
5. **No functional regression.** Existing tests (`dxa_copy`, `sgemm2_dxa`,
   `sgemm_tcu_wg_dxa`, `sgemm_tcu_wg_sp_dxa`) must continue passing on every
   phase. `dxa_multicast` is migrated to the new mask semantics.
6. **Receiver `expect_tx` is mandatory** for both modes — every receiver CTA
   must pre-register the pending event before the issuer can fire. The
   issuer is itself a receiver and must also call `expect_tx` if it is in
   the mask.

---

## 2. Architecture

### 2.1 The bar-scope dichotomy

`vx_barrier_arrive`/`expect_tx`'s `rs1[31]` bit (already plumbed through
[VX_wctl_unit.sv:153](../../hw/rtl/core/VX_wctl_unit.sv#L153) as `bar.is_global`)
is the multicast mode select. The DXA worker receives the bar handle from the
issue instruction; the same bit chooses how `dxa_multicast_mask` is
interpreted and how SMEM writes and release events are routed.

| Aspect | Intra-core (local bar) | Inter-core (global bar) |
|---|---|---|
| `bar_id[31]` | 0 | 1 |
| Mask bit `k` means | "CTA dispatcher slot `k` on this core receives" | "Core `k` (the CTA on that core that's waiting on this gbar id) receives" |
| Co-residence requirement | All addressed CTAs already dispatched on issuer's core | All addressed cores have an active CTA waiting on the gbar |
| SMEM write destination per `k` | `slot_to_lmem_base[k] + smem_offset`, local LMEM | `(core_id=k, LMEM_BASE + smem_offset)`, via cross-core LMEM bus |
| Release event per `k` | One local-bar event packet at `{slot_to_wid_base[k], bar_id_lo}` | One arrival packet on `gbar_bus_if` from each receiving core; aggregated by [VX_gbar_unit.sv:50](../../hw/rtl/mem/VX_gbar_unit.sv#L50) |
| Event count `expect_tx(N)` | `N = 1` per receiver | `N = 1` per receiver |
| Existing infra reused | Local bar slot path in `VX_bar_unit` | `gbar_bus_if` + `VX_gbar_unit` |

### 2.2 Mask interpretation

The mask is **`NUM_CTAS_PER_CORE`-bit wide in intra-core mode** (≤ NUM_WARPS,
the dispatcher's slot count is bounded by NUM_WARPS) and **`NUM_CORES`-bit
wide in inter-core mode**. Today's NUM_WARPS-bit `cta_mask` field on the DXA
req-bus carries either shape — semantics depend on `bar_is_global`.

**Self-inclusion**: the mask **must include the issuer's own slot/core** for
both modes. The DXA worker writes to every bit in the mask, including the
issuer. This keeps the implementation uniform and matches CUDA's
`multicast::cluster` semantics — the issuer also receives the data into its
own SMEM.

### 2.3 Routing tables (the new state)

`VX_cta_dispatch` already tracks `cta_slot_per_warp_r` (wid → slot,
[line 112](../../hw/rtl/core/VX_cta_dispatch.sv#L112)). For multicast routing
we need the *inverse plus LMEM base*, exported to the DXA worker:

```sv
// In VX_cta_dispatch.sv (new state)
reg [NUM_WARPS-1:0][NW_WIDTH-1:0]       slot_to_wid_base_r;   // (slot → wid)
reg [NUM_WARPS-1:0][`LMEM_LOG_SIZE-1:0] slot_to_lmem_base_r;  // (slot → LMEM byte base)
reg [NUM_WARPS-1:0]                     slot_valid_r;         // already exists
```

- Written at CTA admission (the `cur_slot_r` / `cur_lmem_base_r` registration
  already captured at [line 383-391](../../hw/rtl/core/VX_cta_dispatch.sv#L383-L391)).
- For a multi-warp CTA, `slot_to_wid_base_r[slot]` is the *first* warp dispatched
  to that slot — the kernel's "lead warp" by convention (rank 0).
- Read combinationally by the DXA worker's `smem_wr` stage via a new
  `VX_cta_table_if` interface.

In inter-core mode no per-core table is needed — the mask bit position **is**
the core id. The DXA worker emits writes to `(core_id_k, LMEM_BASE +
smem_offset)`, where `LMEM_BASE` is the well-known per-core LMEM region.
The receiving core's CTA layout in LMEM is the kernel's responsibility (typically:
every receiver kernel uses `__local_mem()` which is the start of the CTA's
allocated LMEM region — equivalent across all receivers).

### 2.4 SMEM write fan-out

The replay walker in `VX_dxa_smem_wr` already exists
([line 343-372](../../hw/rtl/dxa/VX_dxa_smem_wr.sv#L343-L372)) — it lifts to
take a routing-table lookup rather than the hardcoded `<< NB_BITS` arithmetic
in [line 403-405](../../hw/rtl/dxa/VX_dxa_smem_wr.sv#L403-L405):

```sv
// Today (after the bce88c86 restoration):
wire [BAR_ADDR_W-1:0] smem_wr_attr_bar = is_multicast
    ? BAR_ADDR_W'(active_bar_addr + (BAR_ADDR_W'(replay_next_idx) << NB_BITS))
    : active_bar_addr;

// After (intra-core):
wire [NW_WIDTH-1:0] tgt_wid = cta_table_if.slot_to_wid_base[replay_next_idx];
wire [BAR_ADDR_W-1:0] smem_wr_attr_bar = is_multicast_local
    ? {tgt_wid, active_bar_addr[BAR_ID_W-1:0]}
    : active_bar_addr;

// SMEM target byte address (intra-core):
wire [LMEM_ADDR_W-1:0] tgt_lmem_base = cta_table_if.slot_to_lmem_base[replay_next_idx];
wire [LMEM_ADDR_W-1:0] tgt_lmem_addr = tgt_lmem_base + smem_offset;
```

For inter-core, the SMEM write rides a new `VX_socket_lmem_bus_if` instead of
the per-core LMEM arbiter — see §2.6.

### 2.5 Barrier release fan-out

**Intra-core**: per receiver `k`, emit one local-bar event packet to
`{slot_to_wid_base[k], bar_id_lo}` on the existing `dxa_txbar_bus_if` →
`VX_wctl_unit` → `VX_bar_unit` path. The `is_event + phase=0` decrement
branch ([VX_bar_unit.sv:83](../../hw/rtl/core/VX_bar_unit.sv#L83)) drives the
pending-event counter at each receiver's bar slot to zero, unblocking each
CTA's `arrive_and_wait`.

**Inter-core**: a single release packet is emitted per receiver core via that
core's `dxa_txbar_bus_if`. Each receiving core's `VX_bar_unit` decrements its
local copy of the gbar's pending-event counter. When that counter and the
local arrival count both hit zero, the core sends `gbar_req_valid` to
`VX_gbar_unit` ([VX_bar_unit.sv:137](../../hw/rtl/core/VX_bar_unit.sv#L137)).
`VX_gbar_unit` aggregates across cores, and when all participating cores have
reported, broadcasts `rsp_valid` back to release every waiter.

The DXA worker in inter-core mode therefore needs to **emit release packets
to remote cores**. This needs a socket-level fanout for `dxa_txbar_bus_if`:
one issuer-side `txbar_bus_if` per remote `core_id` in the mask. Concretely:
the issuer's DXA worker drives a per-target-core release packet onto the
existing inter-core bar/release infrastructure described in §2.6.

### 2.6 Socket-level cross-core LMEM bus (the principal lift)

The only piece of fabric that does not exist today. Add a new socket-scoped
bus:

```sv
// hw/rtl/mem/VX_socket_lmem_bus_if.sv
interface VX_socket_lmem_bus_if;
    typedef struct packed {
        logic [NC_WIDTH-1:0]            target_core;
        logic [`LMEM_LOG_SIZE-1:0]      lmem_addr;     // byte address into target's LMEM
        logic [DXA_LMEM_WORD_SIZE*8-1:0] data;
        logic [DXA_LMEM_WORD_SIZE-1:0]   byteen;
        logic [BAR_ADDR_W-1:0]           release_bar;  // for the bundled release event
        logic                            release_last;  // last word for this transfer
    } req_data_t;

    logic        req_valid;
    logic        req_ready;
    req_data_t   req_data;
endinterface
```

Wiring:
- **Producer**: the DXA worker's `smem_wr` in inter-core mode drives this bus
  instead of the local LMEM bus. One write per (mask bit) per fb word.
- **Consumer**: a new socket-level arbiter `VX_socket_lmem_arb` demuxes by
  `target_core` and presents the write to that core's local LMEM arbiter as
  if it were an LSU write to LMEM.
- **Release packet**: piggybacks on the same bus. The receiving core's
  LMEM-completion logic (already in place via `notify_smem_done`) generates
  the local `dxa_txbar_bus_if` arrival packet, which feeds the per-core
  bar_unit's gbar-event decrement path.

`VX_socket.sv` instantiates the arbiter and wires it to each core's LMEM bus.
For `NUM_SOCKETS > 1`, this bus is socket-local — multicast across sockets is
out of scope (matches NVIDIA: clusters are GPC-scoped).

The width is modest: `NC_WIDTH + LMEM_LOG_SIZE + DXA_LMEM_WORD_SIZE*9 +
BAR_ADDR_W + 1` ≈ `~300 b` for typical params. One bus per socket.

### 2.7 DXA descriptor changes

The DXA descriptor stays the same except:

- **Remove** `smem_stride` and `bar_stride` from the issue path
  ([naive c80a85c8 fields](https://github.com/.../VX_dxa_pkg.sv)). The
  dispatcher table provides per-receiver LMEM base; the bar position is
  derived from `slot_to_wid_base` (intra) or the mask bit position (inter).
- **Keep** `is_multicast` and `cta_mask` (reinterpreted per §2.2).
- **Add** `bar_is_global` to the worker's launch params, sourced from the
  issuing instruction's bar handle (the same `rs1[31]` bit `VX_wctl_unit`
  already extracts).

---

## 3. Software interface

### 3.1 The single intrinsic

```c
// sw/kernel/include/vx_dxa.h
// Multicast 2D copy. `bar_id` selects scope: local bar → intra-core
// multicast (mask = CTA slot bits); global bar → inter-core multicast
// (mask = core id bits).
inline void vx_dxa_issue_2d_multicast_wg(
    uint32_t desc_slot,
    uint32_t bar_id,
    void*    smem_offset,
    uint32_t coord_x,
    uint32_t coord_y,
    uint32_t mask);
```

The intrinsic remains a single instruction; no SW-side mode switch. The bar
handle's scope bit drives everything downstream.

### 3.2 Receiver pre-registration

`bar.expect_tx(1)` (formerly `arrive_tx`,
[vx_barrier.h:49](../../sw/kernel/include/vx_barrier.h#L49)) must be called
by **every** CTA in the multicast mask before `bar.arrive_and_wait()`. This
is unchanged from today and applies to both modes — for global bars,
`gbarrier::expect_tx` should be added as a mirror of `barrier::expect_tx`.

### 3.3 Kernel patterns

**Intra-core (within one core, K co-resident CTAs share a tile):**
```cpp
vortex::barrier bar(0);                       // local bar
const bool is_dxa_warp = (get_sub_group_id() == 0);
const uint32_t my_slot = csr_read(VX_CSR_CTA_ID);

if (is_dxa_warp) bar.expect_tx(1);

if (is_dxa_warp && my_slot == 0) {
    uint32_t mask = (1u << num_active_ctas) - 1;
    vx_dxa_issue_2d_multicast_wg(kDescB, bar.id(), shB,
                                  col_base, row_base, mask);
}
bar.arrive_and_wait();
// shB now contains the multicast'd tile B for every co-resident CTA.
```

**Inter-core (one CTA per core, K cores in the socket share a tile):**
```cpp
vortex::gbarrier gbar(0);                      // global bar
const bool is_dxa_warp = (get_sub_group_id() == 0);
const uint32_t my_core = vx_core_id();

if (is_dxa_warp) gbar.expect_tx(1);

if (is_dxa_warp && my_core == 0) {
    uint32_t core_mask = (1u << num_active_cores) - 1;
    vx_dxa_issue_2d_multicast_wg(kDescB, gbar.id(), shB,
                                  col_base, row_base, core_mask);
}
gbar.arrive_and_wait();
```

The kernel body is structurally identical between the two modes — only the
barrier construction differs.

---

## 4. Hardware changes (file-by-file)

| File | Change |
|---|---|
| [VX_cta_dispatch.sv](../../hw/rtl/core/VX_cta_dispatch.sv) | Add `slot_to_wid_base_r[NUM_WARPS][NW_WIDTH]` and `slot_to_lmem_base_r[NUM_WARPS][`LMEM_LOG_SIZE]` reg arrays. Write at admission ([line 388-391](../../hw/rtl/core/VX_cta_dispatch.sv#L388-L391)). Export via new `VX_cta_table_if`. |
| **New**: `hw/rtl/core/VX_cta_table_if.sv` | Slim read-only interface: `slot_to_wid_base[NUM_WARPS]`, `slot_to_lmem_base[NUM_WARPS]`, `slot_valid[NUM_WARPS]`. |
| [VX_dxa_worker.sv](../../hw/rtl/dxa/VX_dxa_worker.sv) | Accept `cta_table_if.slave` port. Pipe `bar_is_global` from descriptor to smem_wr. Drop `smem_stride` / `bar_stride` plumbing. |
| [VX_dxa_smem_wr.sv](../../hw/rtl/dxa/VX_dxa_smem_wr.sv) | Replace `<< NB_BITS` walker ([line 403](../../hw/rtl/dxa/VX_dxa_smem_wr.sv#L403)) with `slot_to_wid_base` lookup (intra-core) and core-id walker (inter-core). For inter-core, output onto the new socket bus instead of local LMEM. |
| **New**: `hw/rtl/mem/VX_socket_lmem_bus_if.sv` | Socket-level cross-core LMEM write bus (§2.6). |
| **New**: `hw/rtl/mem/VX_socket_lmem_arb.sv` | Demux by `target_core` to each core's LMEM bus. |
| [VX_socket.sv](../../hw/rtl/VX_socket.sv) | Instantiate `VX_socket_lmem_arb`, route DXA inter-core writes from each core's worker to peers' LMEM. |
| [VX_bar_unit.sv](../../hw/rtl/core/VX_bar_unit.sv) | No structural change — existing `is_global + is_event + phase=0` decrement path is the receive side of inter-core release. Verify gbar timing under DXA-driven release. |
| [vx_dxa.h](../../sw/kernel/include/vx_dxa.h) | Document mask reinterpretation; no signature change. |
| [vx_barrier.h](../../sw/kernel/include/vx_barrier.h) | Add `gbarrier::expect_tx(uint32_t count = 1)` mirror of the local version. |

---

## 5. Co-residence guarantees

This is the part where the design replaces the current race-prone luck with
explicit invariants. There is no new hardware admission gate — the existing
barrier semantics carry the burden, as long as the kernel follows the
discipline below.

### 5.1 Intra-core (local bar)

The receivers must already be dispatched on the issuer's core when the
multicast fires. Two sub-cases:

- **All-receivers-already-running**: every multicast receiver is an admitted,
  in-flight CTA at the moment the issuer's DXA worker fires its first SMEM
  write. The kernel-side invariant enforcing this is:
  - Each receiving CTA calls `bar.expect_tx(1)` *before* `bar.arrive_and_wait`.
  - The issuer's `arrive_and_wait` follows its DXA issue.
  - Because the local bar is per-CTA-slot, a not-yet-dispatched receiver's
    bar slot does not exist — its release event would land on an empty slot
    and be silently dropped (the exact race we hit today in `dxa_multicast`
    with `num_cores=1`, 4 warps/CTA).
  - The kernel must therefore size CTAs so all multicast targets fit
    simultaneously: `popcount(mask) × warps_per_CTA ≤ NUM_WARPS`.
- **Detection**: a `DBG_ASSERT_DXA_MULTICAST_COHERENCE` macro can flag the
  case where the DXA worker walks a mask bit `k` whose `slot_valid_r[k]` is
  0 — this is a kernel bug, not a hardware race.

### 5.2 Inter-core (global bar)

The receivers are remote cores. The kernel sizes the grid so each
participating core hosts one CTA waiting on the same gbar id. The gbar
mechanism's existing aggregation
([VX_gbar_unit.sv:50](../../hw/rtl/mem/VX_gbar_unit.sv#L50)) handles the
cross-core synchronization:

- Each receiving core's CTA calls `gbar.expect_tx(1)` before
  `gbar.arrive_and_wait`.
- The DXA worker emits one release packet per masked core; each remote core's
  `VX_bar_unit` decrements its local gbar pending-event count.
- When the local count reaches zero and the core's CTA has arrived, that
  core fires `gbar_req_valid`; `VX_gbar_unit` aggregates across all
  participating cores and broadcasts the response.

The same `slot_valid` discipline applies at each receiving core — but
because each core hosts only one CTA per gbar id in this usage, it reduces
to "the receiver CTA has been admitted by the receiver's `VX_cta_dispatch`."
A small admission-deadlock risk exists if the KMU dispatches CTAs in an
order that places the issuer before the receivers; this is mitigated by the
KMU dispatching CTAs round-robin across cores (already the default policy),
so all participating cores have at least one CTA admitted before the issuer
starts executing.

---

## 6. Phased plan

Each phase is independently shippable with regression gates.

### Phase 1 — Intra-core multicast (~1 week)

- Add `slot_to_wid_base_r` / `slot_to_lmem_base_r` to `VX_cta_dispatch` + new
  `VX_cta_table_if`.
- Modify `VX_dxa_smem_wr` to use slot-table lookup in intra-core mode.
- Wire `bar_is_global` through DXA descriptor / worker.
- Migrate `dxa_multicast` test to new mask semantics:
  - Configure to `--cores=1 --warps=8` (or similar) so `popcount(mask) ×
    warps_per_CTA ≤ NUM_WARPS`.
  - Verify `expect_tx(1)` is called per receiver.
- Update `dxa_copy_multicast` if still present.
- Pass: `dxa_copy`, `dxa_multicast` (new), `sgemm2_dxa`, `sgemm_tcu_wg_dxa`,
  `sgemm_tcu_wg_sp_dxa` on `num_cores=1`.

### Phase 2 — Inter-core multicast fabric (~2 weeks)

- New `VX_socket_lmem_bus_if` + `VX_socket_lmem_arb`.
- DXA worker emits onto socket bus in inter-core mode.
- DXA worker emits per-target-core release packets through the
  `dxa_txbar_bus_if` fanout at the socket level.
- Add `gbarrier::expect_tx`.
- New test `dxa_multicast_g` (§7.2).
- Pass: `dxa_multicast_g` on `num_cores=4 --warps=4`; all Phase 1 tests
  continue to pass.

### Phase 3 — sgemm_tcu_wg_dxa optimization (~3 days)

- Restructure kernel to use multiple smaller CTAs sharing tile B.
- Add `USE_DXA_MULTICAST` variant.
- Measure GMEM bandwidth and steady-state cycles.
- Document the geometry trade-off (small CTAs → multicast benefit vs.
  large CTAs → register-pressure benefit).

### Phase 4 — Cleanup (~2 days)

- Remove the legacy `<< NB_BITS` walker; remove dead `smem_stride` /
  `bar_stride` fields from `VX_dxa_pkg`.
- Add `DBG_ASSERT_DXA_MULTICAST_COHERENCE` for the slot-not-resident case.
- Update [docs/microarchitecture.md](../microarchitecture.md) with the
  multicast section.

---

## 7. Testing

### 7.1 Intra-core: `tests/regression/dxa_multicast`

**Configuration**: `--cores=1`, NUM_WARPS large enough for K co-resident
CTAs of `warps_per_CTA` warps each.

**Geometry**: small tile (e.g. 4×4 elements per CTA), K=4 receiver CTAs,
each CTA uses 1 or 2 warps so `K × warps_per_CTA ≤ NUM_WARPS`.

**Kernel pattern**:
```cpp
vortex::barrier bar(0);
if (is_dxa_warp) bar.expect_tx(1);
if (is_dxa_warp && cta_id == 0) {
    vx_dxa_issue_2d_multicast_wg(kDescSrc, bar.id(), __local_mem(),
                                  col_base, row_base,
                                  (1u << arg->num_ctas) - 1);
}
bar.arrive_and_wait();
verify_tile(__local_mem());      // every CTA verifies independently
```

**Verification**:
- Each CTA's LMEM contains the identical tile after the barrier.
- Store-back path writes the LMEM tile to GMEM at distinct CTA-owned
  output regions; host compares each region against the source tile.
- Run under both rtlsim and simx; report cycles for documentation.

**Sweep**:
- K ∈ {2, 4, 8} (provided NUM_WARPS allows).
- `warps_per_CTA` ∈ {1, 2}.
- Issuer ∈ {first CTA, last CTA in mask}.

### 7.2 Inter-core: `tests/regression/dxa_multicast_g` (new)

**Configuration**: `--cores=K`, NUM_WARPS small (e.g. 4) so each core hosts
one CTA.

**Geometry**: K=NUM_CORES receiver cores, each running one CTA.

**Kernel pattern**:
```cpp
vortex::gbarrier gbar(0);
if (is_dxa_warp) gbar.expect_tx(1);
if (is_dxa_warp && vx_core_id() == 0) {
    vx_dxa_issue_2d_multicast_wg(kDescSrc, gbar.id(), __local_mem(),
                                  col_base, row_base,
                                  (1u << arg->num_cores) - 1);
}
gbar.arrive_and_wait();
verify_tile(__local_mem());
```

**Verification**:
- Each core's LMEM contains the identical tile after the gbar release.
- Store-back from each core to a distinct GMEM region.
- Run under rtlsim with `--cores=4 --warps=4`.

**Sweep**:
- `num_cores` ∈ {2, 4, 8}.
- Issuer ∈ {core 0, core (K-1)}.
- Partial mask (only some cores participate) — verifies non-participating
  cores are unaffected.

### 7.3 Integration: `tests/regression/sgemm_tcu_wg_dxa` (optimized variant)

Add a `USE_DXA_MULTICAST` compile-time switch:
- Without: existing one-CTA-per-core layout (baseline).
- With: K CTAs per core, all sharing tile B via intra-core multicast.

**Metrics to capture**:
- Steady-state cycles per output tile.
- GMEM read traffic (via perf counters): should drop to ~1/K of baseline
  for B.
- Total runtime on the standard sgemm test sizes.

### 7.4 Unit-level testbench: `hw/unittest/dxa_core`

Extend with multicast-specific scenarios:
- Mask with self-only (degenerate, single SMEM write per word).
- Mask with all bits set.
- Sparse mask (bits at non-contiguous positions).
- Verify release packet count == popcount(mask) — no extra, no missing.

### 7.5 Regression CI gates

Each phase adds to the existing regression list:
- Phase 1: `dxa_multicast` (intra-core) must pass on `num_cores=1`.
- Phase 2: `dxa_multicast_g` must pass on `num_cores=4`. The intra-core test
  also runs with `num_cores=4` to verify it still works with idle cores.
- Phase 3: `sgemm_tcu_wg_dxa` runtime ≤ baseline; B GMEM traffic ≤ 1/K
  baseline ± tolerance.

---

## 8. Open questions

1. **Mask self-inclusion under partial replay.** If the issuer is also a
   receiver (and §2.2 says yes), should the issuer's own SMEM write go
   through the same socket bus (for inter-core mode) or short-circuit to
   the local LMEM arbiter? Short-circuit is cleaner and avoids needless
   socket-bus traffic for the local copy.
2. **`VX_cta_table_if` placement.** Define under `hw/rtl/core/` (with
   the dispatcher) or `hw/rtl/dxa/` (with its consumer)? Recommend `core/`
   since `VX_cta_dispatch` owns the data.
3. **Socket bus arbitration.** Multiple cores' DXA workers might issue
   inter-core multicasts concurrently. Use round-robin or fixed-priority?
   Round-robin matches existing socket-level conventions.
4. **gbar pre-existing `size_m1`.** The existing local-bar arrive aggregator
   sets `gbar_req_size_m1` from `count_r` on a normal `is_arrive` arrival
   ([VX_bar_unit.sv:139](../../hw/rtl/core/VX_bar_unit.sv#L139)). For
   multicast we want the receiver-side `count_r` to be `num_warps_per_CTA -
   1` (the standard local-arrival count for that CTA), and the
   pending-event decrement to land via the DXA-driven release. This is
   compatible with no changes — confirm with simulation.

---

## 9. What this proposal does *not* do

- **Does not introduce a new "cluster" abstraction at the dispatcher.** The
  local/global bar dichotomy already provides the scoping mechanism.
- **Does not add a hardware admission gate.** Kernel discipline (every
  receiver calls `expect_tx` before issuer fires) carries the synchronization.
  Misuse manifests as deadlock or a debug-asserted "slot not resident"
  message — diagnosable, not silently incorrect.
- **Does not span sockets.** Inter-core multicast is socket-local, matching
  NVIDIA's GPC-scoped cluster model. Cross-socket multicast is intentionally
  out of scope.
- **Does not change `vx_dxa_issue_2d_wg`** (the non-multicast variant) or
  any other DXA descriptor. Only multicast routing is touched.
