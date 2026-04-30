# RTL §4 refactor — Phase 1 audit (storage / interfaces)

**Status:** design note for the WGMMA tile-buffer refactor described in
[wgmma_simx_v3_proposal.md §4](wgmma_simx_v3_proposal.md#4-proposed-design-implementation-independent).
Surfaces interface decisions before Phase 2 commits to module shapes.

---

## 1. Storage math (canonical configs)

All sizes in 32-bit words. `bank-row = NUM_BANKS = TCU_NT` words.

### Constants from VX_tcu_pkg.sv

| | NT=8 NRC=8 | NT=32 NRC=8 |
|---|---|---|
| TCU_TC_M / TC_N / TC_K            | 4 / 2 / 2  | 8 / 4 / 4 |
| TCU_WG_M_STEPS / N_STEPS / K_STEPS | 2 / 4 / 2 | 2 / 4 / 2 |
| TCU_A_BLOCK_SIZE = TC_M × TC_K    | 8 words    | 32 words |
| TCU_B_BLOCK_SIZE = TC_K × TC_N    | 4 words    | 32 words |
| TCU_BLOCK_CAP = NT                | 8 words    | 32 words |
| TCU_A_SUB_BLOCKS                  | 1          | 1 |
| TCU_B_SUB_BLOCKS                  | 2          | 1 |
| NUM_BANKS                         | 8          | 32 |

### Per-block A buffer (VX_tcu_abuf storage)

Holds **M_STEPS A-blocks** = the current k-stripe for one warp.

```
A_STRIPE_WORDS      = TCU_WG_M_STEPS × TCU_A_BLOCK_SIZE
A_STRIPE_BANK_ROWS  = ceil(A_STRIPE_WORDS / NUM_BANKS)
```

| | NT=8 NRC=8 | NT=32 NRC=8 |
|---|---|---|
| A_STRIPE_WORDS      | 16 | 64 |
| A_STRIPE_BANK_ROWS  | 2  | 2  |

### Shared B buffer (VX_tcu_bbuf storage)

Holds **1 bank-row** (or 2 with §4.10.4 ping-pong). One bank-row may hold
1+ (k,n) blocks depending on `B_BLOCK_WORDS` vs `NUM_BANKS`.

```
B_BLOCKS_PER_BANK_ROW = NUM_BANKS / TCU_B_BLOCK_SIZE     (>= 1)
B_BUF_WORDS           = NUM_BANKS                         (1 bank-row)
```

| | NT=8 NRC=8 | NT=32 NRC=8 |
|---|---|---|
| B_BLOCKS_PER_BANK_ROW | 2 | 1 |
| B_BUF_WORDS           | 8 | 32 |

### Reduction vs current per-block tile-buffer

Per Q-block storage, NT=8 NRC=8:

| | Today | Proposed |
|---|---|---|
| A storage   | TILE_M×TILE_K = 32 words | 16 words |
| B storage   | TILE_K×TILE_N = 32 words | 0 (shared) |
| Per-Q total | **64 words**             | **16 words** |
| Q=2 grand   | 128 words                | 32 + 8 (shared B) = **40 words** (3.2×) |
| Q=4 grand   | 256 words                | 64 + 8 = **72 words** (3.6×) |

(§4.6 quotes ~5× at NT=32 NRC=8 — confirmed in same direction.)

---

## 2. Refill triggers

### Per-block A refill key

A-stripe identified by `(desc_a, step_k)`. Refill when key changes:

```
new_a_key = {req_desc_a, req_step_k}
refill_a  = req_valid && (new_a_key != resident_a_key)
```

On refill: issue `A_STRIPE_BANK_ROWS` LmemReqs starting at
`desc_a_row_base + step_k × A_K_STRIDE_BANK_ROWS`.

### Shared B refill key

The B bank-row index for a given `(step_k, step_n)`:

```
b_block_idx     = step_k × N_STEPS + step_n
b_bank_row_idx  = b_block_idx / B_BLOCKS_PER_BANK_ROW   (integer div)
new_b_key       = {req_desc_b, b_bank_row_idx}
refill_b        = any_q_valid && (new_b_key != resident_b_key)
```

**Sub-bank-row reuse for free**: at NT=8 NRC=8, two consecutive
`step_n` values (0,1) share `b_bank_row_idx=0` — bbuf hits without refill.
The proposal's §4.9 trace overstates B fetch count for this config; actual
B traffic is half what §4.9 shows.

---

## 3. Bus widths to VX_tcu_core (unchanged)

Conclusion: **no change to `tbuf_rs1_data` / `tbuf_rs2_data` widths.**

Rationale:

- `tbuf_rs1_data` is `[TCU_BLOCK_CAP-1:0][XLEN-1:0]` =
  one A-block worth of words. Today's gather selects 1 of M_STEPS×K_STEPS
  blocks. New gather selects 1 of M_STEPS blocks (k-stripe is implicit
  in "what's resident"). Width unchanged.
- `tbuf_rs2_data` is `[TCU_WG_RS2_WIDTH-1:0][XLEN-1:0]` =
  one bank-row's worth. Today's gather pulls from the full
  `K_STEPS × N_STEPS` B tile via `step_k`/`step_n`. New gather pulls
  from the resident bank-row only — width still 1 bank-row, indexing
  collapses to identity (or sub-bank-row select via `step_n[lower bits]`).

### `b_off` in tcu_core (unchanged formula, different meaning)

[VX_tcu_core.sv:200-207](../../hw/rtl/tcu/VX_tcu_core.sv#L200-L207):

```
b_off = (OFF_W'(step_n) & OFF_W'(TCU_B_SUB_BLOCKS-1)) << LG_B_BS;
```

- NT=32 NRC=8: `TCU_B_SUB_BLOCKS=1` → mask is 0 → `b_off=0` always.
  step_n is implicitly absorbed by the bbuf's bank-row identity.
- NT=8 NRC=8:  `TCU_B_SUB_BLOCKS=2` → `b_off ∈ {0, 4}` based on
  `step_n[0]`. step_n[0] selects the (k,n) block within the bank-row;
  upper step_n bits go into the bbuf refill key.

The formula stays. The split between "indexes within a bank-row" (lower
step_n bits → b_off) and "selects which bank-row is resident" (upper
step_n bits → bbuf key) is what falls out naturally, no code change in
tcu_core.

### `a_off` in tcu_core (unchanged formula, different meaning)

```
a_off = (OFF_W'(step_m) & OFF_W'(TCU_A_SUB_BLOCKS-1)) << LG_A_BS;
```

- All canonical configs: `TCU_A_SUB_BLOCKS=1` → `a_off=0` → step_m
  implicitly absorbed by abuf's per-stripe storage indexing in gather.

If TC_M ever shrinks so `TCU_A_SUB_BLOCKS > 1`, same split as B:
lower step_m bits → a_off, upper bits → abuf storage row.

---

## 4. Format-aware lane extraction

Both `_abuf` and `_bbuf` need format-aware extraction (fp32/fp16/fp8
sub-word lanes within a 32-bit word). Today this lives in
[VX_tcu_tbuf_gather.sv](../../hw/rtl/tcu/VX_tcu_tbuf_gather.sv).

Plan: each module has its own inline extraction (small, formats are few).
If synth shows duplication > N gates, factor a helper later.

---

## 5. Meta path (deferred to Phase 4)

Sparse `_mbuf` mirrors `_abuf` shape (per-block, keyed on `(desc_a, step_k)`).
Stripe meta storage:

```
META_STRIPE_WORDS = TCU_WG_M_STEPS × META_STRIDE_<fmt>
```

`META_STRIDE_<fmt>` is format-dependent (see
[VX_tcu_tbuf_fetch.sv:96-104](../../hw/rtl/tcu/VX_tcu_tbuf_fetch.sv#L96-L104)).
Defer detailed sizing to Phase 4 — common-path dense (Phases 2-3) doesn't
need this.

---

## 6. LMEM arbitration

Current: `BLOCK_SIZE` masters (one per warp's tbuf) →
[VX_tcu_unit.sv:142-162](../../hw/rtl/tcu/VX_tcu_unit.sv#L142-L162).

New (dense): `BLOCK_SIZE × abuf` + `1 × bbuf` = `Q + 1` masters.
Sparse adds `BLOCK_SIZE × mbuf` → `2Q + 1`.

Priority hint: `bbuf` first (one fetch unblocks all Q cores), then
abuf round-robin. Use existing `VX_mem_arb` with `ARBITER="P"` (priority).

---

## 7. Open questions for Phase 2/3

1. **bbuf refill while `any_q_valid=0`?** If a WGMMA gap leaves
   different (k,n) requested next cycle, refill on observation, not
   on dispatch. Choose: trigger on first valid that differs from
   resident key. Same for abuf.

2. **Lock-step Q semantics under refill.** A single `bbuf` refill
   blocks all Q `tcu_core[b].ready`. A single `abuf[b]` refill blocks
   only `tcu_core[b].ready` — but the `lane_dispatch` lockstep
   propagates that to all Q anyway. Confirm in Phase 3 by trace.

3. **`(desc_a, step_k)` reuse across uop sequence.** Within a k-stripe,
   `step_k` is constant — abuf refill fires once per stripe entry.
   Cross-WGMMA reuse: if next WGMMA's `(desc_a, step_k=0)` matches
   the resident key, no refill. Falls out of Step 2's key check.

4. **Meta `_mbuf` placement (Phase 4).** Per-warp like abuf, or could
   share the abuf module via a `WITH_META` parameter. Pick after
   building abuf and seeing the FSM shape.

---

## Phase exit checklist

- [x] Storage math for NT=8 NRC=8 and NT=32 NRC=8 documented
- [x] Refill key formulas defined (A, B)
- [x] Bus widths confirmed unchanged
- [x] `a_off` / `b_off` in tcu_core confirmed unchanged
- [x] Sub-bank-row B reuse documented
- [x] LMEM arb topology defined (Q+1 → 1)
- [ ] Build a/b modules (Phase 2)
- [ ] Wire into unit, smoke test (Phase 3)
