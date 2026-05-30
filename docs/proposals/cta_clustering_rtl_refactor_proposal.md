# CTA Clustering RTL Refactor — Proposal

**Date:** 2026-05-29
**Status:** Draft — design only; no code changes yet.
**Owners:** RTL team
**Related:**
  - [Launch-level CTA clustering CHANGELOG entry](../../CHANGELOG.md) (vortex2.h `cluster_dim[3]`),
  - SimX reference: [sim/simx/cta_dispatcher.cpp](../../sim/simx/cta_dispatcher.cpp), [sim/simx/dxa/dxa_core.cpp](../../sim/simx/dxa/dxa_core.cpp),
  - RTL surface: [hw/rtl/core/VX_cta_dispatch.sv](../../hw/rtl/core/VX_cta_dispatch.sv), [hw/rtl/core/VX_mem_unit.sv](../../hw/rtl/core/VX_mem_unit.sv), [hw/rtl/dxa/VX_dxa_unit.sv](../../hw/rtl/dxa/VX_dxa_unit.sv), [hw/rtl/dxa/VX_dxa_smem_wr.sv](../../hw/rtl/dxa/VX_dxa_smem_wr.sv), [hw/rtl/interfaces/VX_cta_table_if.sv](../../hw/rtl/interfaces/VX_cta_table_if.sv).

---

## Summary

Adopt **cluster-contiguous LMEM allocation** symmetrically in RTL and SimX. The dispatcher reserves K × per-CTA-LMEM as one span at the first CTA of every cluster, with a pre-wrap to LMEM offset 0 if the span would straddle the ring boundary. With contiguity guaranteed, the DXA multicast path collapses to stride arithmetic at the issuer (`smem_addr + r × stride`); the entire **receiver-side address translator**, the `cta_table_if` interface, and three of the four flop tables (`slot_to_lmem_base_r`, `slot_to_wid_base_r`, `wid_to_lmem_base_r`, `cta_slot_per_warp_r` — only the wid → slot map is retained for retirement) all drop out.

The first-of-cluster predicate is **emitted by the KMU** as a 1-bit field on `kmu_bus_if.data` (combinational `(intra_offset == 0)`), and **consumed by the dispatcher** to gate the K-span admission. No counter, no decrement, no width-sizing question — single source of truth shared by both RTL and SimX. The mirror change in SimX (1 line in `Kmu::step`, 1 field in `kmu_req_t`, delete `cluster_cta_remaining_` from `CtaDispatcher`) keeps the two implementations line-for-line equivalent on the admission path.

Net effect (sized at `VX_CFG_NUM_WARPS=16`, `VX_CFG_LMEM_LOG_SIZE=14`):
- **−672 flops** of cta_table state (3 × 16 × 14-bit tables removed).
- **−1 NUM_WARPS-input MUX cone** on the DXA receive path + the registered MUX + 1-cycle skid buffer wrapped around it for timing closure (Fix A / Fix B in [VX_mem_unit.sv](../../hw/rtl/core/VX_mem_unit.sv)).
- **+1 bit** on `kmu_bus_if.data` + **+~15 lines** of dispatcher logic (K-span admission gate, no flopped counter).
- **LMEM utilisation cost:** worst-case `(K−1) × lmem_size` bytes per LMEM-ring wrap event, ≤ a few percent of LMEM under realistic workloads.

---

## Motivation

### 1. The RTL today carries dead and over-defended state

The cta_table_if interface ([VX_cta_table_if.sv:36-56](../../hw/rtl/interfaces/VX_cta_table_if.sv#L36)) exposes three tables (`slot_to_lmem_base`, `cta_slot_per_warp`, `wid_to_lmem_base`) plus the fourth internal `slot_to_wid_base_r` flop array. A grep of consumers across the entire RTL tree:

| Field | Consumers | Notes |
|---|---|---|
| `slot_to_lmem_base` | VX_mem_unit (1) | DXA receive-side translator only |
| `wid_to_lmem_base`  | VX_dxa_unit  (1) | DXA issuer-side intra-offset computation only |
| `cta_slot_per_warp` | none (UNUSED_VAR everywhere) | Diagnostic only |
| `slot_to_wid_base_r` | none (UNUSED_VAR in cta_dispatch itself) | Dead store |

Every multicast WGMMA pays the cost of all four tables; every non-DXA build pays the cost of the three the DXA path uses. The pre-flattening between them (`wid_to_lmem_base_r ← cur_lmem_base_r` updated alongside `cta_slot_per_warp_r ← cur_slot_r` and `slot_to_lmem_base_r ← cur_lmem_base_r`) exists purely to amortise MUX-cascade depth — the dispatcher writes the same base into three different places per CTA to give the receiver-side translator a single registered MUX rather than two cascaded ones ([VX_cta_dispatch.sv:124-130](../../hw/rtl/core/VX_cta_dispatch.sv#L124)).

### 2. The DXA receive path is a known timing hotspot

[VX_mem_unit.sv:155-166](../../hw/rtl/core/VX_mem_unit.sv#L155) documents two stacked workarounds:
- **Fix A:** route through the `slot_to_lmem_base` field instead of cascading `cta_slot_per_warp → slot_to_lmem_base`, "Critical for U55C-class timing at NUM_WARPS ≥ 16 / 300 MHz where two 32:1 indexed MUXes in series + adder + arb cone would not close."
- **Fix B:** insert a 1-cycle skid buffer between the translator and the LMEM-DMA arbiter so the translator's combinational MUX + adder are the entire src-to-flop path.

Both workarounds exist because the receive-side runtime translation `dxa_recv_base = slot_to_lmem_base[recv_slot]` cannot leave the core (the table is per-core dispatcher state). Eliminating runtime translation eliminates the workarounds.

### 3. The contiguity SimX enforces is the natural assumption

SimX's [dxa_core.cpp:513](../../sim/simx/dxa/dxa_core.cpp#L513) implements multicast peer addressing as:

```cpp
req.addr += uint64_t(cta_warp_idx) * w.smem_stride;
```

Pure stride arithmetic. The dispatcher's K-span reservation ([cta_dispatcher.cpp:88-108](../../sim/simx/cta_dispatcher.cpp#L88)) makes that stride formula well-defined. Today's RTL maintains independence of CTA placement from cluster shape — paying the per-slot lookup cost — without using that flexibility anywhere (the K cluster members are issued sequentially by the KMU and there is no slot-reuse path).

### 4. The dispatcher itself is over-complicated for what it does

Two-cycle retirement forwarding (`rem_warps_write_r` + `rem_warps_write_rr` + `_rdata_fwd` mux at [VX_cta_dispatch.sv:241-244](../../hw/rtl/core/VX_cta_dispatch.sv#L241)) exists because `rem_warps_ram` is `RDW_MODE="R"` with `OUT_REG=1`. Switching to `RDW_MODE="W"` (write-first) removes one stage of forwarding (`_rr` window goes away). The pipeline depth justification in the comment is the historical R/OUT_REG choice, not a fundamental constraint.

---

## Proposed Design

### KMU (VX_kmu.sv) and KMU bus (VX_kmu_bus_if / `kmu_req_t`)

**Add** a 1-bit `is_first_of_cluster` field to the KMU bus payload (RTL) and to `kmu_req_t` (SimX). It is the *only* new signal in the entire proposal.

In RTL the producer is a combinational decode of state KMU already holds ([VX_kmu.sv:59](../../hw/rtl/VX_kmu.sv#L59)):

```sv
assign kmu_bus_if.data.is_first_of_cluster =
    (intra_offset[0] == '0) && (intra_offset[1] == '0) && (intra_offset[2] == '0);
```

In SimX the producer is the corresponding line in `Kmu::step()`, captured *before* the existing `intra_offset_` advance (the block_idx fill below it already runs in that ordering):

```cpp
req->is_first_of_cluster =
    (intra_offset_[0] == 0) &&
    (intra_offset_[1] == 0) &&
    (intra_offset_[2] == 0);
```

The signal is purely derived from existing state — no new flops on either side. The 1-bit width is exact (it is a boolean), not a sizing trade-off.

### Dispatcher (VX_cta_dispatch.sv + sim/simx/cta_dispatcher.{h,cpp})

**Admission check (replaces lines 256-273 in RTL; replaces the `cluster_cta_remaining_ == 0` branches in SimX):**

```sv
wire is_first_of_cluster = kmu_bus_if.data.is_first_of_cluster;
wire cluster_size_next   = kmu_bus_if.data.cluster_dim[0]
                         * kmu_bus_if.data.cluster_dim[1]
                         * kmu_bus_if.data.cluster_dim[2];

wire [LMEM_LOG+CLOG2(MAX_K)-1:0] eff_span = is_first_of_cluster
    ? (kmu_bus_if.data.lmem_size * cluster_size_next)
    : kmu_bus_if.data.lmem_size;

wire [LMEM_LOG:0] lmem_next_tail = lmem_tail_r + eff_span;
wire              lmem_alloc_wraps = is_first_of_cluster && lmem_next_tail[LMEM_LOG];
wire [LMEM_LOG:0] lmem_padding    = lmem_alloc_wraps ? (LMEM_SIZE - lmem_tail_r) : '0;
wire [LMEM_LOG:0] lmem_total_cost = kmu_bus_if.data.lmem_size + lmem_padding;
wire              lmem_ok         = (free_size_r >= lmem_total_cost);
```

Wraps only fire on first-of-cluster, so cluster members 2..K never wrap mid-cluster.

**State update on `kmu_bus_if_fire`** — the dispatcher carries no cluster-position state of its own:

```sv
cur_lmem_base_r <= lmem_alloc_wraps ? '0 : lmem_tail_r;
lmem_tail_r     <= lmem_alloc_wraps
    ? `VX_CFG_LMEM_LOG_SIZE'(kmu_bus_if.data.lmem_size)
    : lmem_next_tail[LMEM_LOG-1:0];
```

**Delete in RTL:** `slot_to_lmem_base_r`, `slot_to_wid_base_r`, `wid_to_lmem_base_r`. All three are produced for cta_table consumers that go away under this proposal.

**Delete in SimX:** `CtaDispatcher::cluster_cta_remaining_` (plus its reset, decrement, and the `== 0` reads — all replaced by `pending_cta_.is_first_of_cluster` / `cta_.is_first_of_cluster`).

**Retain in RTL:** `cta_slot_per_warp_r` — still used by retirement (`done_slot = cta_slot_per_warp_r[warp_done_wid]`). This stays internal to the dispatcher.

### Interface (VX_cta_table_if.sv)

**Delete** the interface. After the dispatcher changes there are zero consumers. The `cta_table_if` slave ports on VX_dxa_unit and VX_mem_unit go away.

### DXA issuer (VX_dxa_unit.sv)

Today the issuer subtracts the absolute SMEM address from its own LMEM base to produce a CTA-relative intra-offset; the receiver re-adds. Under contiguity the issuer emits the **absolute address** unchanged and lets the multicast replay add `r × smem_stride` per beat.

```sv
// Was:
//   issuer_lmem_base = cta_table_if.wid_to_lmem_base[issuer_wid];
//   intra_offset     = lane0_rs1 - issuer_lmem_base;
//   dxa_req.smem_addr = intra_offset;
// Becomes:
assign dxa_req_data_in.smem_addr = lane0_rs1;   // absolute LMEM byte addr
```

Removes one NUM_WARPS-input MUX (`wid_to_lmem_base[issuer_wid]`) and one subtractor on the issue path. The `cta_table_if` slave port disappears.

### DXA SMEM writer (VX_dxa_smem_wr.sv)

The multicast replay path currently emits the **same** `replay_addr = fb_word_addr_r` for every receiver beat ([line 365](../../hw/rtl/dxa/VX_dxa_smem_wr.sv#L365)) and encodes the receiver slot into `bar_addr`'s upper NW_WIDTH bits, deferring the per-receiver address resolution to the receiver core. Under contiguity, fold the resolution into the replay itself:

```sv
// Per-replay-beat receiver rank within the cluster (0..K-1):
//   the priority-encoder index over the cta_mask gives an arbitrary
//   ordering, but the dispatcher placed receivers at contiguous slots,
//   so PE index == cluster rank in placement order.
wire [MC_NW_BITS-1:0] r = replay_next_idx;

// Replay address = base + r × stride
// (one tiny mul; r is log2(K) bits, smem_stride is the desc's stride.)
wire [SMEM_ADDR_WIDTH-1:0] replay_addr =
    fb_word_addr_r + SMEM_ADDR_WIDTH'(r) * SMEM_ADDR_WIDTH'(smem_stride);
```

`smem_stride` was already plumbed in from the descriptor — line 339 currently does `\`UNUSED_VAR (smem_stride)` because contiguity isn't assumed. This change makes it load-bearing.

The `bar_addr` recv-slot field is no longer used for address translation (only for the barrier completion notification). Keep encoding it there for the completion path; the receiver-side path simply ignores it for translation.

**Multiplier sizing:** `r` is `LOG2UP(VX_CFG_NUM_WARPS)` bits — typically 3–5. `r × stride` is a constant-bounded multiplier (≤ 5 partial products) or a 5-deep mux tree. Synthesis folds it to a small shift-add tree; latency is ~2 levels of LUT.

### Receiver translator (VX_mem_unit.sv)

**Delete** the entire DXA receive-side translation block (lines 144-220 of [VX_mem_unit.sv](../../hw/rtl/core/VX_mem_unit.sv#L144), ~80 lines + comments). The DXA write bus carries the absolute LMEM word address now; wire it straight into the LMEM DMA arbiter input.

The "Fix B" elastic buffer (1-cycle skid) goes away with it — the translator that was the timing-critical path no longer exists, so the skid registration that protected its closure is unnecessary. If a skid is still wanted for arb-cone insulation, it stays optional, but it's not load-bearing.

**Saves:** the NUM_WARPS-input indexed MUX + its registered output flop + the post-MUX elastic buffer (DATAW ≈ 1+ADDR+DATA+BYTEEN+ATTR+TAG bits × 2 entries). In NUM_WARPS=16 / VX_CFG_LMEM_LOG_SIZE=14: ~14 flops for the translator MUX out + ~500-1000 flops for the skid buffer payload, depending on DATA_SIZE.

### VX_core wiring

Drop the `cta_table_if` instance and modport connections at [VX_core.sv:96-98](../../hw/rtl/core/VX_core.sv#L96). The four `UNUSED_VAR` markers and the slot table connection from the dispatcher both disappear cleanly.

### Dispatcher retirement-path simplification (independent of clustering)

Two-cycle write forwarding on `rem_warps_ram` is needed only because the RAM is `RDW_MODE="R"` (read-before-write). Switching to `RDW_MODE="W"` collapses the `_rr` shadow registers + the second forwarding compare. Sketch:

```sv
// Drop these:
//   rem_warps_write_rr, rem_warps_waddr_rr, rem_warps_wdata_rr
//   rem_warps_rdata_fwd's second tier
// rem_warps_rdata_fwd reduces to:
wire [NW_WIDTH:0] rem_warps_rdata_fwd =
    (rem_warps_write_r && (rem_warps_waddr_r == done_slot_r_dly))
        ? rem_warps_wdata_r
        : rem_warps_rdata;
```

This is an independent cleanup; bundle it or split it as preferred.

---

## Implementation Phases

| Phase | Scope | Files | Validation |
|---|---|---|---|
| **0** | This proposal | `docs/proposals/cta_clustering_rtl_refactor_proposal.md` | review |
| **1** | KMU emits `is_first_of_cluster`; dispatcher adds K-span admission gate. Symmetric in RTL and SimX. Existing cta_table flops stay populated (additive). | `VX_kmu.sv`, `VX_cta_dispatch.sv`, `sim/simx/kmu/kmu.{h,cpp}`, `sim/simx/cta_dispatcher.{h,cpp}` | rtlsim full matrix + `sgemm_tcu_wg_dxa_mcast` (SimX + rtlsim) |
| **2** | Issuer + writer: switch to absolute-addr + stride-replay | `VX_dxa_unit.sv`, `VX_dxa_smem_wr.sv`, `sim/simx/dxa/dxa_core.cpp` (already does this — confirm parity) | mcast test + SimX↔RTL trace diff |
| **3** | Receiver: delete translator block; wire bus directly to LMEM arbiter | `VX_mem_unit.sv` | full DXA matrix |
| **4** | Delete cta_table_if + the three tables in dispatcher | `VX_cta_table_if.sv`, `VX_cta_dispatch.sv`, `VX_core.sv`, `VX_mem_unit.sv`, `VX_dxa_unit.sv` | rebuild |
| **5** | Optional cleanup: `RDW_MODE="W"` + drop `_rr` forwarding tier | `VX_cta_dispatch.sv` | full matrix |
| **6** | PPA measurement: Yosys + OpenSTA before/after on U55C-class config | `hw/syn/yosys/` | report |

Phases 1–4 share the cluster-contiguity assumption end-to-end. Phase 1 is **additive in both layers** — the new flag on the bus is a new struct field (older code ignores it), and the K-span check only tightens admission. With the cta_table flops still populated, the existing DXA receive translator keeps working through phases 1 alone. The DXA path simplification in 2–4 is interlocking and must land together.

Non-DXA builds get phase 1 (KMU bit + dispatcher K-span) and phase 4 (table deletion, since cta_table has no other consumers); phases 2/3 only apply under `VX_CFG_EXT_DXA_ENABLE`.

---

## Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Phase 2/3 boundary inconsistency (issuer drops translation but receiver still expects intra-offset) | low | wrong addresses, silent data corruption | land 2 + 3 together in one commit; CI on `sgemm_tcu_wg_dxa_mcast` |
| Cluster-spans-wrap admission stall starves a long-running kernel | low | throughput regression | the worst-case stall is `K × lmem_size` of waiting room; same admission backpressure as a single large CTA — no new deadlock risk |
| `r × stride` multiplier hurts replay-path timing | low | fmax drop | for max practical K=8 this is a 3-input partial-product sum; well below `lmem_alloc_wraps` adder depth that timing already closes. Worst case insert a stride-mult pipeline reg between PE and write-bus — this is a 1-cycle add on multicast only. |
| Hidden consumer of `cta_table_if` outside the grep scope | low | compile error | phase 4 catches it at elaboration; revert is a one-file patch |
| LMEM waste under cluster_dim > 1 trips a near-OOM test | low | test failure | the waste is `(K-1) × lmem_size` per wrap, bounded by `K × lmem_size` total. Tests that demand full LMEM should already size for cluster_dim=1 |

---

## What This Does NOT Change

- **KMU walk** ([VX_kmu.sv](../../hw/rtl/VX_kmu.sv) + [sim/simx/kmu/kmu.cpp](../../sim/simx/kmu/kmu.cpp)): the iteration order, `intra_offset` / `group_origin` semantics, and the `block_idx = group_origin + intra_offset` formula are all unchanged. The only addition is a 1-bit combinational output (`is_first_of_cluster`) — no walk-state changes.
- **CTA dispatcher's slot ring + retirement table** other than the cta_table flops: untouched (except for the independent `RDW_MODE` cleanup in phase 5).
- **DXA descriptor / multicast bus protocol**: the bus already carries `smem_stride`; we just make it load-bearing rather than `UNUSED_VAR`. The `bar_addr` encoding stays the same for completion notification.
- **Host runtime / kernel ABI**: zero changes.

---

## Out of Scope

- Cross-core multicast (DXA Path B): the proposal assumes intra-core multicast, which is what the current RTL supports. If cross-core multicast lands later, it will need a different rendezvous path because receiver bases live on different cores; the contiguity assumption is local-per-core.
- Dynamic CTA migration: nothing in the current RTL does it, but if a future scheduler reuses freed mid-LMEM slots while a cluster is still placing, contiguity breaks. Out of scope; flag it in the design doc if such a scheduler is proposed.
- KMU-side group-aware backpressure: currently the dispatcher backpressures KMU via `kmu_bus_if.ready`. If a cluster's K-span doesn't fit, the first CTA stalls; KMU stops emitting until space frees. No new mechanism needed.

---

## Expected PPA Outcome (qualitative)

| Metric | Direction | Driver |
|---|---|---|
| Flop count | ↓ | 3 × NUM_WARPS × LMEM_LOG_SIZE = 672 bits @ NW=16 (cta_table tables) + skid buf flops. The KMU-side `is_first_of_cluster` adds **0 flops** (combinational decode of existing `intra_offset` state). |
| LUTs / cell area | ↓ | NUM_WARPS-input MUX cone (DXA receive) + retirement `_rr` forwarding |
| fmax | ↑ | DXA receive path was Fix-A/Fix-B'd to close; removing it removes the critical path |
| KMU bus width | ↑ by 1 bit | `is_first_of_cluster` payload field — negligible |
| LMEM peak utilisation | ↓ marginally | `(K-1) × lmem_size` worst-case padding per wrap |
| Kernel runtime (sgemm_tcu_wg_dxa_mcast) | flat or slightly ↑ | one MUX cone delay removed from every multicast receive |
| SimX dispatcher state | ↓ | `cluster_cta_remaining_` member + reset + decrement + reload all deleted |

Phase 6 should quantify this on the Yosys/OpenSTA + Synopsys flows the repo ships.
