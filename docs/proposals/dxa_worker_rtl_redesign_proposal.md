# DXA Worker RTL Redesign — Proposal

**Date:** 2026-05-11
**Status:** Draft
**Owners:** RTL team
**Related:**
[dxa_simx_v3_proposal.md](dxa_simx_v3_proposal.md) (companion SimX proposal),
[libs_feature_agnostic_redesign.md](libs_feature_agnostic_redesign.md),
[VX_dxa_worker.sv](../../hw/rtl/dxa/VX_dxa_worker.sv).

---

## Summary

Aggressively reduce DXA worker area and increase end-to-end block-transfer
throughput by collapsing the current 5-stage pipeline's redundant storage
and serial drain into a tight direct-drain dataflow:

- **Eliminate `VX_dxa_rsp_buf`** (the per-tag CL BRAM) and the per-CL
  metadata FIFO inside `VX_dxa_gmem_req`. Replace both with a single
  ~16-bit per-tag slot table (×8 = ~128 b vs. today's ~4.5 kb).
- **Pipeline `VX_dxa_smem_wr`** from a 3-state FSM (`IDLE→FETCH→DRAIN`)
  with no inter-CL overlap into a 2-stage continuous pipeline
  (Align → Drain) that sustains **1 LMEM-word write per cycle**.
- **Overlap setup with the previous transfer's tail** so back-to-back
  launches no longer pay the 6–12-cycle DSP-precompute bubble.
- **Compress `VX_dxa_addr_gen`** offset state from 4× MEM_ADDR_WIDTH +
  5-input adder to one rolling cursor + 2-input adder.
- **Hoist multicast replay** out of `smem_wr` into the LMEM arbiter so
  M-way multicast is 1 write/cycle instead of M writes/cycle.

Net (back-of-envelope, MAX_OUTSTANDING=8, CL=64 B, SMEM_WORD=32 B):

| Axis | Before | After |
|---|---|---|
| Per-tag storage | ~563 b × 8 ≈ 4.5 kb (incl. BRAM tile) | ~16 b × 8 = 128 b |
| DSPs (setup) | 3 | 2 |
| Steady-state drain | 1 CL / (3–4 cyc) | 1 SMEM-word / cycle continuous |
| Setup bubble for streamed transfers | 6–12 cyc | 0 |
| Multicast cost | M× drain | 1× |

---

## 1. Constraints (load-bearing)

1. **No functional regression.** The current regression suite —
   `dxa_copy`, `dxa_multicast`, `sgemm2_dxa`, `sgemm_tcu_wg_dxa`,
   `sgemm_tcu_wg_sp_dxa`, plus the `hw/unittest/dxa_core` Verilator
   testbench — must pass on every phase. Each phase is independently
   shippable; no phase straddles a multi-week regression-gap window.
2. **`VX_dxa_worker_req_if` and `VX_dxa_completion` semantics unchanged.**
   The DXA core ↔ worker interface and the LMEM-side
   `notify_smem_done` / `bar_addr` flag plumbing are externally
   observable and stable across this redesign. The completion-via-LMEM-
   flag contract documented in
   [VX_dxa_smem_wr.sv:298-313](../../hw/rtl/dxa/VX_dxa_smem_wr.sv#L298)
   is preserved.
3. **`libs/` is off-limits.** All edits land in `hw/rtl/dxa/`. The
   redesign keeps using existing `VX_pipe_register`, `VX_fifo_queue`,
   `VX_dp_ram`, `VX_allocator`, `VX_priority_encoder` library blocks;
   if a library piece becomes unused, it is simply not instantiated.
4. **Worker is purely structural.** [VX_dxa_worker.sv](../../hw/rtl/dxa/VX_dxa_worker.sv)
   stays a wiring file — submodule boundaries change but the top-level
   stays structural ([VX_dxa_worker.sv:14-17](../../hw/rtl/dxa/VX_dxa_worker.sv#L14-L17)).

---

## 2. Why the current worker is suboptimal

[VX_dxa_worker.sv:21-307](../../hw/rtl/dxa/VX_dxa_worker.sv#L21-L307) wires
five submodules:

```
setup ─▶ addr_gen ─▶ gmem_req ─▶ rsp_buf ─┐
                       │                  │
                       └─ inflight_fifo ──┴─▶ smem_wr ─▶ smem_bus
```

CL payload data lives in **two** large storage structures
(`rsp_buf` DP-RAM and `smem_wr.fb_data_r`) and per-CL metadata lives in
**three** ([`VX_allocator`, `VX_pending_size`, `VX_fifo_queue inflight_fifo`](../../hw/rtl/dxa/VX_dxa_gmem_req.sv#L90-L186)).

### 2.1 Area hotspots (ranked)

| # | Hotspot | Where | Approx cost |
|---|---|---|---|
| 1 | `rsp_buf.data_store` DP-RAM | [VX_dxa_rsp_buf.sv:52-88](../../hw/rtl/dxa/VX_dxa_rsp_buf.sv#L52-L88) | `GMEM_DATAW × MAX_OUTSTANDING` ≈ 4096 b @ 8 slots, **1 BRAM tile** forced via `LUTRAM=0`. `OUT_REG=1` adds 1 cycle of fetch latency. |
| 2 | `inflight_fifo` LUTRAM | [VX_dxa_gmem_req.sv:134-186](../../hw/rtl/dxa/VX_dxa_gmem_req.sv#L134-L186) | DATAW ≈ 50 b × 8 = ~400 b. ~32 b of it is `smem_byte_addr` — recomputable. |
| 3 | `smem_wr.fb_data_r` fill buffer | [VX_dxa_smem_wr.sv:128](../../hw/rtl/dxa/VX_dxa_smem_wr.sv#L128) | `(CL_SIZE+SMEM_WORD_SIZE)*8` = 768 b @ 64+32. Duplicates `rsp_buf` data after barrel-shift. |
| 4 | `setup` operand regs + 3 DSPs | [VX_dxa_setup.sv:137-280](../../hw/rtl/dxa/VX_dxa_setup.sv#L137-L280) | ~480 b operand regs + 3 DSPs idle 100% of ACTIVE. One DSP-phase computes `total_smem_writes` which `VX_dxa_worker` marks `UNUSED_VAR` ([VX_dxa_worker.sv:343](../../hw/rtl/dxa/VX_dxa_worker.sv#L343)) — dead silicon. |
| 5 | `addr_gen.dim_offset_r[0..3]` + 5-input MEM_ADDR adder | [VX_dxa_addr_gen.sv:74-76](../../hw/rtl/dxa/VX_dxa_addr_gen.sv#L74-L76) | 4× MEM_ADDR_WIDTH regs + a 5-input MEM_ADDR-wide adder on the per-cycle path. Likely Fmax-limiting. |
| 6 | 512-b barrel shifter | [VX_dxa_smem_wr.sv:163](../../hw/rtl/dxa/VX_dxa_smem_wr.sv#L163) | Kept (the data still needs alignment) — moved behind a register in §3. |
| 7 | Redundant alloc-full / pending-full / fifo-full | [VX_dxa_gmem_req.sv:168](../../hw/rtl/dxa/VX_dxa_gmem_req.sv#L168) | Three near-identical credit checks on identically-sized structures. |

### 2.2 Throughput bottlenecks (ranked)

| # | Bottleneck | Where | Effect |
|---|---|---|---|
| 1 | `smem_wr` `IDLE→FETCH→DRAIN` FSM with **no overlap across CLs** | [VX_dxa_smem_wr.sv:96-230](../../hw/rtl/dxa/VX_dxa_smem_wr.sv#L96-L230) | Caps drain at 1 CL / (3 + ⌈CL/SMEM_WORD⌉) cycles → ~3–4× slower than the GMEM-side issue rate. **Dominant bottleneck.** |
| 2 | Setup latency 6–12 cyc per transfer | [VX_dxa_setup.sv:284](../../hw/rtl/dxa/VX_dxa_setup.sv#L284) (`done_ctr`) | Downstream pipeline fully idle. |
| 3 | Multicast strictly serial through one priority encoder + one SMEM port | [VX_dxa_smem_wr.sv:250-307](../../hw/rtl/dxa/VX_dxa_smem_wr.sv#L250-L307) | `popcount(cta_mask)`× per-word drain time. |
| 4 | `OUT_REG=1` on rsp_buf DP-RAM | [VX_dxa_rsp_buf.sv:52](../../hw/rtl/dxa/VX_dxa_rsp_buf.sv#L52) | +1 cyc per CL in S_FETCH. |
| 5 | 5-input MEM_ADDR adder in addr_gen | [VX_dxa_addr_gen.sv:74-76](../../hw/rtl/dxa/VX_dxa_addr_gen.sv#L74-L76) | Fmax. |
| 6 | `pending_full` back-pressure when MAX_OUTSTANDING < GMEM_RTT | [VX_dxa_gmem_req.sv:168](../../hw/rtl/dxa/VX_dxa_gmem_req.sv#L168) | Because drain rate < issue rate today, the engine quickly fills its 8 in-flight slots and stalls GMEM issue. The drain bottleneck (#1) is what makes this matter. |

### 2.3 Redundant state

- `inflight_fifo` carries `tag` — recoverable from FIFO position if tags
  are allocated round-robin.
- `inflight_fifo` carries `smem_byte_addr` — recoverable in `smem_wr` by
  accumulating `valid_length` per consumed CL.
- `inflight_fifo` carries `byte_offset` / `valid_length` — derivable from
  a shadow odometer; cheaper to keep in a tiny per-tag table than to
  duplicate the odometer.
- `rsp_arrived` bitvector in `rsp_buf` duplicates the per-tag valid flag
  needed by `smem_wr`'s head-of-queue check.
- `setup.r_total_smem_writes` / `worker.ag_total_smem_writes` — UNUSED_VAR
  ([VX_dxa_worker.sv:343](../../hw/rtl/dxa/VX_dxa_worker.sv#L343)).

---

## 3. Target architecture

```
                ┌──── slot_table[t] (×MAX_OUTSTANDING)
                │        { arrived, oob, last, byte_offset, valid_length }
                │
setup ─▶ addr_gen ─▶ gmem_req ─▶ direct-drain skid ─┐
                                                    │
                       slot_table  ◀────────────────┘
                              │
                              ▼
                          smem_wr_pipe (Align │ Drain)  ──▶ smem_bus
                                                     │
                                            (optional cta_mask) ──▶ lmem_arbiter fan-out
```

Five RTL submodules collapse to four:

| Today | Redesign | Note |
|---|---|---|
| `VX_dxa_setup` | `VX_dxa_setup` (slimmer) | Drop dead `total_smem_writes` precompute; add overlap with previous transfer's drain tail. |
| `VX_dxa_addr_gen` | `VX_dxa_addr_gen` (rolling-cursor) | One MEM_ADDR cursor + per-wrap deltas precomputed in `setup`. |
| `VX_dxa_gmem_req` | `VX_dxa_gmem_req` (slim) | Drops the metadata FIFO; writes to `slot_table`. Single credit counter. |
| `VX_dxa_rsp_buf` | **deleted** | Direct-drain into smem_wr's Align stage; arrival bit lives in `slot_table`. |
| `VX_dxa_smem_wr` | `VX_dxa_smem_wr` (2-stage pipe) | `Align │ Drain`, no inter-CL bubble. |

### 3.1 Direct-drain — replace `rsp_buf` + metadata FIFO with one slot table

Per outstanding tag, store only what cannot be recomputed from drain
order:

```systemverilog
typedef struct packed {
    logic                       arrived;
    logic                       oob;
    logic                       last;
    logic [GMEM_OFF_BITS-1:0]   byte_offset;     // 6 b @ 64 B CL
    logic [GMEM_OFF_BITS:0]     valid_length;    // 7 b
} dxa_slot_t;                                    // ≈ 16 b

dxa_slot_t slots_r [MAX_OUTSTANDING];
```

- **CL data is not buffered.** GMEM responses go straight into a 1-deep
  skid register at the input of `smem_wr.Align`. The Align stage barrel-
  shifts the CL using `slots_r[head].byte_offset` and presents it to
  Drain.
- **Tag = `head_idx % MAX_OUTSTANDING`.** Replace `VX_allocator` with a
  pair of issue/drain pointers + a credit counter. Drain order = issue
  order (matches today's behavior — see [VX_dxa_smem_wr.sv:105](../../hw/rtl/dxa/VX_dxa_smem_wr.sv#L105)).
- **`smem_byte_addr` is reconstructed in `smem_wr`** by initializing to
  `setup_params.initial_smem_base` and incrementing by `valid_length` on
  each CL drained.

Two response-ordering policies are admissible; pick by inspecting the
GMEM bus contract:

- **In-order responses (preferred):** `slot_table[head].arrived` is the
  only bit ever queried; the skid back-pressures the GMEM port. Smallest
  area, smallest LUT mux.
- **Out-of-order responses:** `arrived` is set per-tag by response
  arrival; the head-of-queue stalls until `arrived[head]=1`. Identical
  to today's `rsp_arrived` bitvector, but without the BRAM. Adds one
  small priority compare on `arrived`.

OOB CLs synthesize `arrived=1, oob=1` immediately at `gmem_req` time —
no GMEM transaction, no skid traffic.

**Savings:** delete one BRAM tile, ~400 b LUTRAM (`inflight_fifo`),
~120 b of allocator/pending free-list state. ~35× reduction in tag state.

### 3.2 `smem_wr` 2-stage pipeline (drain throughput)

Replace the `IDLE → S_FETCH → S_DRAIN` FSM
([VX_dxa_smem_wr.sv:96-230](../../hw/rtl/dxa/VX_dxa_smem_wr.sv#L96-L230))
with a continuously-flowing 2-stage pipeline:

```
Stage A (Align):
  - When slots_r[head].arrived && Drain ready-for-new-CL:
      pop the skid (or oob-fill), barrel-shift by byte_offset,
      latch into fb_data_r, release the slot.
  - 1 cycle. Owns the 512-b barrel shifter (now behind a register).

Stage B (Drain):
  - 1 SMEM-word per cycle from fb_data_r.
  - On last beat of the CL, signal Align "consume next" so there is
    zero bubble between CLs.
  - For multicast (today's path; see §3.5 for the hoist), drive
    replay_remaining_r alongside fb_data_r; one bus beat per replay.
```

Steady-state: **1 LMEM-word/cycle continuously**, independent of CL
boundaries. For CL=64 B, SMEM_WORD=32 B that is 2 SMEM beats/CL with no
inter-CL gap, vs. today's 3 + 2 = 5 cycles/CL → **2.5× drain
throughput**. With the multicast hoist (§3.5) M-way multicast collapses
from M cycles/word to 1.

### 3.3 Setup overlap — eliminate the 6–12-cycle bubble

Today `VX_dxa_setup` gates `req_ready` on `IDLE` and emits
`pipeline_start` after a rank-dependent `done_ctr`
([VX_dxa_setup.sv:284](../../hw/rtl/dxa/VX_dxa_setup.sv#L284)).

Redesign:

- Accept the next `req_if` request while the previous transfer is still
  in ACTIVE — run the DSP precompute in parallel with the previous
  transfer's drain.
- Stage the result in a `staged_setup_params` register (~120 b).
- Fire `pipeline_start` the cycle the previous transfer's
  `transfer_done` is asserted.

Adds ~120 b of registers, deletes 6–12 cycles of pipeline bubble per
streamed transfer. Standalone (non-back-to-back) transfers see no
change.

Also delete the `r_total_smem_writes` plumbing
([VX_dxa_setup.sv:416-419](../../hw/rtl/dxa/VX_dxa_setup.sv#L416-L419)
through [VX_dxa_worker.sv:343](../../hw/rtl/dxa/VX_dxa_worker.sv#L343))
— one DSP slice, one FSM phase, ~32 b regs.

### 3.4 `addr_gen` offset compression — shorter Fmax path

Replace four MEM_ADDR-wide `dim_offset_r[d]` accumulators + the 5-input
adder ([VX_dxa_addr_gen.sv:74-76](../../hw/rtl/dxa/VX_dxa_addr_gen.sv#L74-L76))
with one rolling cursor and per-wrap deltas precomputed in `setup`:

```systemverilog
gmem_base_r += stride[0];                    // inner step
gmem_base_r += wrap_delta[d];                // on dim-d wrap

// in VX_dxa_setup (uses idle DSPs):
wrap_delta[d] = stride[d+1] - tiles[d] * stride[d];
```

- Per-cycle adder becomes 2-input MEM_ADDR-wide.
- Saves 3× MEM_ADDR-wide regs.
- The 4-level ripple odometer at
  [VX_dxa_addr_gen.sv:168-191](../../hw/rtl/dxa/VX_dxa_addr_gen.sv#L168-L191)
  is preserved (row-boundary only — not on the critical path).

### 3.5 Multicast hoist into LMEM arbiter

Today, multicast replay scans `replay_remaining_r` through a priority
encoder once per output beat
([VX_dxa_smem_wr.sv:257-264](../../hw/rtl/dxa/VX_dxa_smem_wr.sv#L257-L264))
and issues one SMEM bus beat per CTA per word.

Two implementation options (pick one in Phase 5):

- **(A) Hoist into the LMEM arbiter.** `smem_wr` emits one bus beat with
  `cta_mask` + `smem_stride` attached as side-band; the LMEM arbiter /
  bank crossbar fan-outs to all banks targeted by the mask in a single
  cycle. Best ROI; requires LMEM arbiter support and bus_if sideband
  fields.
- **(B) Bank-broadcast write primitive.** Add a single-cycle "write
  data to N strided offsets" SMEM operation. Smaller scope but
  requires SMEM macro support.

Either eliminates the priority encoder, `replay_remaining_r`, and the
per-replay `cta_idx × smem_stride` multiplier from `smem_wr`. M-way
multicast drops from M cycles/word to 1.

### 3.6 Credit counter simplification

Replace `VX_allocator` + `VX_pending_size` + `inflight_fifo.full`
([VX_dxa_gmem_req.sv:90-186](../../hw/rtl/dxa/VX_dxa_gmem_req.sv#L90-L186))
with a single up/down `credits_r` and circular `issue_idx_r` /
`drain_idx_r`. Each free `slot_table` entry = one credit. Saves the
free-list LUTRAM and collapses three near-identical full checks into one.

---

## 4. Phasing

Each phase is independently shippable; each gates on the regression
suite passing.

| Phase | Scope | Risk | ROI |
|---|---|---|---|
| 0 | Dead-code cull: drop `r_total_smem_writes` plumbing; collapse `alloc_full ∨ pending_full ∨ fifo_full` into one credit. No behavior change. | Very low | Area only, minor. |
| 1 | `smem_wr` 2-stage pipeline (§3.2). Single-module refactor; preserves bus contract. **Doubles drain throughput in isolation.** | Med — focused targeted tests on CL/SMEM-word ratios + multicast. | High. |
| 2 | Setup overlap with previous transfer (§3.3). Touches `VX_dxa_setup` + `worker_req_if` handshake; write a back-to-back launch test. | Med | Med (streamed-transfer perf only). |
| 3 | `addr_gen` rolling-cursor (§3.4). Single-module + wrap-delta precompute in `setup`. | Low | Fmax + small area. |
| 4 | Direct-drain rewrite: delete `VX_dxa_rsp_buf`, collapse `inflight_fifo` into `slot_table` (§3.1, §3.6). Biggest area cut. | High — verify in-order response policy on `gmem_bus_if`; fall back to per-tag `arrived` bit if needed. | Very high (area). |
| 5 | Multicast hoist into LMEM arbiter (§3.5). Cross-module. Coordinate with arbiter owner. | High — external interface change. | Very high (multicast perf). |

Phase 0 + Phase 1 alone should land the headline throughput win
(roughly 2.5× steady-state drain for non-multicast) with no interface
churn. Phase 4 lands the headline area win.

---

## 5. Verification

- **Unit:** `hw/unittest/dxa_core/VX_dxa_core_top.sv` exercises a real
  worker against a stub L1; extend it with cases that stress
  - back-to-back transfers (Phase 2),
  - CL-aligned vs. unaligned spans (Phase 1),
  - response-ordering corner cases (Phase 4),
  - multicast with `popcount(cta_mask) ∈ {1,2,4,8}` (Phase 5).
- **Regression:** `tests/regression/dxa_copy`, `dxa_multicast`,
  `sgemm2_dxa`, `sgemm_tcu_wg_dxa`, `sgemm_tcu_wg_sp_dxa` must pass on
  every phase.
- **Perf:** with `PERF_ENABLE`, compare `dxa_perf.gmem_latency` and
  `dxa_perf.lmem_writes` per-phase against baseline; expected steady-
  state drain on the `dxa_copy` kernel is ≥ 2× by Phase 1 end.

---

## 6. Out of scope

- DXA core (request arb, queue, descriptor table, dispatch) — only
  shape-preserving changes that follow from worker-interface tweaks.
- GMEM bus protocol changes beyond response-ordering policy selection
  (Phase 4 may need a one-line `VX_mem_bus_if` parameter assertion).
- Descriptor format / `VX_dcr_dxa` — unchanged.
- SimX — see [dxa_simx_v3_proposal.md](dxa_simx_v3_proposal.md). The
  SimX worker model is *already* aligned with the redesigned RTL on
  drain throughput and per-tag storage; only the multicast hoist
  (Phase 5) requires a SimX change.
