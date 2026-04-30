# WGMMA SimX v3 — SMEM layout addendum

**Date:** 2026-04-29
**Status:** Draft addendum
**Amends:** [wgmma_simx_v3_proposal.md §4.1, §4.2](wgmma_simx_v3_proposal.md#41-layering-decision-minimal-structured-tcu-buffer)
**Companion:** [wgmma_simx_v3_rtl_phase1_audit.md](wgmma_simx_v3_rtl_phase1_audit.md)

---

## What's missing in §4.1

§4.1 says the per-block A buffer holds "`M_STEPS` bank-rows per block
(the active k-stripe's A rows for that warp)". This is only true if each
A-block is **one contiguous bank-row in SMEM**. The proposal does not
state this; current kernels lay out A row-major:

```cpp
// tests/regression/sgemm_tcu_wg/kernel.cpp:44
A_smem[r * ctx::tileK + c] = pA[...]
```

Under row-major, one A-block at NT=8 NRC=8 spreads its 8 words across
60 SMEM words — never one contiguous bank-row. The §4.1 storage claim
(~5× SRAM reduction) and the §4.6 "first-uop latency lower" claim
cannot land without fixing this.

**Resolution:** make block-major SMEM the explicit, sole layout for
WGMMA tiles. No flag, no fallback — drop row-major end-to-end. This is
load-bearing for §4 to deliver as written.

---

## SMEM layout spec

### A (per-warp)

```
A_smem[(k * M_STEPS + m) * BLOCK_WORDS + i * TC_K + k_in]

  k        ∈ [0, K_STEPS)        — outer index, matches step_k
  m        ∈ [0, M_STEPS)        — inner index, matches step_m
  i, k_in  ∈ [0, TC_M) × [0, TC_K)
  BLOCK_WORDS = TC_M * TC_K       — typically equals NUM_BANKS in canonical configs
```

**k outermost** (not `m * K_STEPS + k`): one k-stripe is M_STEPS
adjacent bank-rows starting at `base + k * M_STEPS * BLOCK_BANK_ROWS`.
This makes §4.10.2 (burst-fetch the 2Q A rows at k-stripe entry)
free-by-construction rather than an opt-in optimization.

### B (TB-shared)

```
B_smem[(k * N_STEPS + n) * BLOCK_WORDS + j * (TC_K * i_ratio) + k_in_elem]

  BLOCK_WORDS = TC_K * TC_N              — in 32-bit words
  j           ∈ [0, TC_N)                — N column within block
  k_in_elem   ∈ [0, TC_K * i_ratio)      — K-element index, packed i_ratio per word
```

Within-block layout is **N outer, K inner**: each 32-bit word holds
`i_ratio` consecutive K-elements at one (j, k_word) cell. This matches
[VX_tcu_core.sv:251](../../hw/rtl/tcu/VX_tcu_core.sv#L251)'s
`b_col[k] = rs2_data[b_off + j*TC_K + k]` indexing and the existing
gather's K-pair-per-word fp16/fp8 packing at
[VX_tcu_tbuf_gather.sv:236-246](../../hw/rtl/tcu/VX_tcu_tbuf_gather.sv#L236-L246).

Same k-major shape across blocks. Within a k-stripe, B blocks for
n=0..N_STEPS-1 are adjacent bank-rows — natural support for §4.10.4
ping-pong.

### Sub-word packing (unchanged)

Within a 32-bit word: fp32 = 1 element, fp16 = 2 elements, fp8 = 4
elements. Format-aware lane extraction at the gather stage, exactly as
[VX_tcu_tbuf_gather.sv:226-250](../../hw/rtl/tcu/VX_tcu_tbuf_gather.sv#L226-L250)
does today.

---

## Descriptor format

The current descriptor encodes `{base, ldm}` where `ldm` is the
row-stride in bytes. Under block-major, row-stride is meaningless —
replaced by block-stride, which is uniform and elab-time-known:

| Field | Bits | Meaning |
|---|---|---|
| `base` | low | SMEM bank-row address of `block(k=0, m=0)` (A) or `block(k=0, n=0)` (B) |
| (rest) | high | unused / reserved |

`block_bank_rows` is implicit (compile-time `ceil(BLOCK_WORDS / NUM_BANKS)`,
typically 1). Stripe-stride is `M_STEPS × block_bank_rows` (A) or
`N_STEPS × block_bank_rows` (B), also compile-time.

**Net descriptor change**: `ldm` field removed. Just the base.

---

## Kernel diff (per kernel)

```cpp
// before  ─  row-major
A_smem[r * tileK + c] = pA[(tile_row + r) * K + (k + c)];

// after  ─  k-major / block-major
uint32_t k_blk = c / TC_K;          uint32_t k_in = c % TC_K;
uint32_t m_blk = r / TC_M;          uint32_t i_in = r % TC_M;
A_smem[(k_blk * M_STEPS + m_blk) * (TC_M * TC_K)
       + i_in * TC_K + k_in] = pA[(tile_row + r) * K + (k + c)];
```

Same shape for B. Descriptor construction loses the stride argument:

```cpp
auto desc_a = vt::vx_make_smem_desc(A_warp);   // base only
```

Kernels affected: every WGMMA-using kernel under `tests/regression/`
and `kernel/include/` — `sgemm_tcu_wg`, `sgemm_tcu_wg_dxa`,
`sgemm_tcu_wg_sp`, `sgemm_tcu_wg_sp_dxa`, plus any in-tree dependency.

---

## Effect on §4 (storage / refill / fetch)

With block-major in place, every storage and refill formula in §4 reads
exactly as written:

| §4 claim | Holds under block-major |
|---|---|
| Per-block A = `M_STEPS` bank-rows  | ✓ stripe is contiguous |
| Shared B = `1` bank-row per (k,n)  | ✓ block IS the bank-row |
| First-uop fetch = `Q + 1` rows     | ✓ no over-fetch |
| §4.10.2 burst-A free               | ✓ k-major makes it default |
| §4.10.4 ping-pong straightforward  | ✓ next-B at adjacent address |

Refill triggers (also per
[wgmma_simx_v3_rtl_phase1_audit.md §2](wgmma_simx_v3_rtl_phase1_audit.md#2-refill-triggers)):

```
abuf:  refill on {desc_a, step_k} change  → fetch M_STEPS adjacent bank-rows
bbuf:  refill on {desc_b, step_k * N_STEPS + step_n} change  → fetch 1 bank-row
```

No address-span computation, no bank-row straddle, no sub-bank-row
demuxing — the block is the bank-row.

---

## What's deleted (no row-major path remains)

- `ldm` field from WGMMA descriptor encoding
- All row-major SMEM-load loops in `tests/regression/sgemm_tcu_wg*/kernel.cpp`
- Stride-shift logic in
  [VX_tcu_tbuf_gather.sv:90-101](../../hw/rtl/tcu/VX_tcu_tbuf_gather.sv#L90-L101)
  (`b_stride_shift_*` becomes constant 0; the file gets deleted entirely
  per §4 / §6 of the original proposal anyway)
- Any WGMMA path that assumes `A_smem[r * tileK + c]` indexing

---

## Risks specific to this addendum

- **Cooperative SMEM-load bank-conflict pattern shifts.** The kernel
  load loop touches the same words, just in a different order. Measure
  pre/post on `sgemm_tcu_wg` to confirm no SMEM-write regression.
- **Descriptor-format change ripples to SimX, FPGA test harness, any
  external WGMMA caller.** Audit before merge.
- **All WGMMA tests fail between Phase 1 (RTL switch) and Phase 2
  (kernel migration).** Mitigate by landing both in one branch.

---

## Implementation order

Plan unchanged from the original proposal §6 / Phase 1 audit; this
addendum only adds the kernel + descriptor rewrite as a prerequisite
for the §4 RTL design to deliver as claimed:

1. **Phase 0 (this addendum)** — descriptor encoding update +
   `sgemm_tcu_wg` fp16 kernel rewrite to k-major.
2. **Phase 1-3** of [wgmma_simx_v3_rtl_phase1_audit.md](wgmma_simx_v3_rtl_phase1_audit.md)
   — write `_abuf` / `_bbuf` / `_tbuf`, rewire `VX_tcu_unit`, smoke test.
3. **Phase 4-5** — sparse `_mbuf`, delete legacy fetch/gather files.
4. **Phase 6** — §4.10.3 prefetch, §4.10.4 ping-pong (separate commits).

§4.10.2 (burst-A) is no longer a separate phase — it's free under §3's
k-major arrangement.
