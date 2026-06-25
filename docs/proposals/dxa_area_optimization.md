# DXA Coarse-Grain Area Optimization — U55C @ 300 MHz

**Target:** xcu55c-fsvh2892 (-2L), 300 MHz / 3.333 ns, Vivado 2024.1.
**DUT:** `VX_dxa_core_top` → `VX_dxa_core` (NUM_REQS=1, NUM_DXA_UNITS=1).
**Companion doc:** `dxa_addr_gen_timing.md` (timing closure, now MET at WNS +0.012).
**Status going in (`nt32nw32_fix7`):** timing closed, but area-imbalanced —
**LUT 13627, LUTRAM 584, FF 5375, DSP 10, BRAM 0**. The board is LUT-bound
with BRAM completely unused; this doc attacks that imbalance.

---

## 0. What is actually in the DUT (scope correction)

The 8-point review list spans the whole DXA subsystem, but only a subset is
inside the synthesized DUT. Establishing this first avoids "optimizing" blocks
that contribute zero to the reported numbers.

```
VX_dxa_core_top
└── VX_dxa_core  (NUM_REQS=1, GMEM_OUT_PORTS=1)
    ├── VX_dxa_desc_table     ← desc_store (768b × 16)        ┐ the entire
    ├── VX_dxa_req_arb (1:1)  ← VX_stream_arb, OUT_BUF=0      │ LUTRAM=584
    ├── VX_elastic_buffer     ← req_queue (~290b × 16, LUTRAM)┘
    ├── VX_dxa_dispatch (1:1)
    ├── VX_dxa_worker
    │   ├── VX_dxa_setup      ← 3× VX_multiplier, staged/active dup
    │   ├── VX_dxa_addr_gen   ← (timing doc; 1 DSP)
    │   ├── VX_dxa_gmem_req   ← slot table 8×~29b, 2 PEs
    │   └── VX_dxa_smem_wr    ← barrel shifters + 1 runtime multiply
    └── 2× VX_mem_arb (1:1)
```

**Not in the DUT:**
- `VX_dxa_rsp_buf` — **dead code**, instantiated nowhere (gmem_req streams
  responses straight to smem_wr via the `sw_*` channel; the rsp_buf BRAM was
  removed in that refactor). Review point 3 has **zero DUT impact**.
- `VX_dxa_completion` (`compl_fifo`) — instantiated in `VX_mem_unit`, part of
  the full-chip completion/txbar path, **not** the DUT. Review point 2 has
  **zero DUT impact**.
- `VX_dxa_unit` — the SFU issue-side adapter (decodes wgather lanes → dxa_req).
  Its internal `rsp_buf` is a 2-deep `VX_elastic_buffer` for the writeback
  header, **not** a data RAM. Not in the DUT.

### Where the LUTs actually are

`LUTRAM=584` is only **4%** of the 13627 LUTs. It is entirely the two
wide-shallow RAMs (`desc_store` + `req_queue`). The other **13043 logic LUTs**
are combinational: the `smem_wr` gather/scatter barrel shifters, the rank-gated
32-bit decode/arithmetic in `setup`, and the `addr_gen` odometer. **BRAM/DSP
rebalancing addresses the 584 LUTRAM + a few multiply-LUTs; the bulk of the
win must come from logic simplification in `setup` and `smem_wr`.** This doc
covers both levers and is honest about which is which.

### Why BRAM is 0 (and why that is not purely an oversight)

Every DXA storage structure is **16 deep**. BRAM is efficient when deep and
narrow (RAMB36 = 512×72 or 1024×36); a 768b×16 table maps to ~11 RAMB18, each
~97% empty in depth. LUTRAM is the *correct* primitive for wide-shallow, which
is why the original author chose it. Forcing BRAM is still worthwhile **here**
only because the U55C has 2016 BRAM (abundant) and the design is LUT-bound — so
spending underfilled BRAM tiles to reclaim scarce LUTs is a favorable trade.
This is a deliberate area-rebalance, not a "fix"; the doc states the BRAM cost
honestly per item.

---

## 1. `desc_table` → true BRAM  (review point 6) — primary BRAM lever

**Current:** `VX_dp_ram` with `OUT_REG=0` (combinational read). `FORCE_BRAM`
*is* true for 768b×16, but the BRAM path is only taken when `OUT_REG!=0`; with
a combinational read the RAM can never be a true BRAM (BRAM has a registered
read port), so it falls to distributed RAM. This is the larger half of the 584
LUTRAM.

**Change:** set `OUT_REG=1` and absorb the resulting 1-cycle read latency.

The read feeds the launch path: `req_queue.head → desc_read_addr (comb) →
desc_store → desc_data → setup.launch_accept`. Registering the read inserts one
bubble between the queued request and `launch_accept`. DXA launches are
infrequent (one per transfer; transfers drain for hundreds of cycles), so +1
launch-latency cycle is throughput-irrelevant. Two ways to absorb it:

- **(a) Fetch stage (recommended):** a tiny 2-state FSM in `VX_dxa_core` —
  present `desc_read_addr` from the queue head, wait one cycle, then drive
  `dispatch_in_if.valid` with the registered `desc_data`. ~3 FFs of control.
- **(b) Speculative read:** issue the read whenever the queue is non-empty;
  `desc_data` is then valid the cycle the queue presents the request. No bubble
  if the head is stable for ≥1 cycle (it always is — the queue holds it until
  `req_ready`).

**Cost/benefit (estimate, pending synth):** removes ~300–400 LUTRAM; adds ~6–11
RAMB18 (768-wide, 16-deep, underfilled). BRAM 0 → ~11. Net LUT ↓, BRAM ↑ — the
exact rebalance requested.

---

## 1b. `req_queue` → BRAM (same lever, bonus)

**Current:** `VX_elastic_buffer SIZE=16, LUTRAM=1`, DATAW = `$bits(dxa_req_data_t)`
≈ 290 b (UUID_WIDTH=44 dominates). The other half of the 584 LUTRAM.

**Change:** `LUTRAM=0`. `VX_elastic_buffer`'s internal `VX_fifo_queue`/`VX_dp_ram`
then inherits `FORCE_BRAM` (true for 290b×16) **provided** the queue uses a
registered-read RAM (it does for SIZE>1). Converts the remaining ~200–280
LUTRAM to ~5 RAMB18.

**Caveat:** like §1, wide-shallow → underfilled BRAM. Recommended only because
LUT is the bottleneck. If we want to *minimize total BRAM*, do §1 only and keep
`req_queue` in LUTRAM.

---

## 2. `VX_dxa_setup` — multiplies, shifts, and decode dedup (review point 4)

This is the highest-value **logic-LUT** target. Three findings, in priority
order:

### 2a. Two of the three phase-0 "multiplies" are power-of-2 shifts

`elem_bytes = 1 << esize_enc` (esize ∈ {0,1,2,3} for 1/2/4/8 B). Therefore:
- `row_len_bytes = tile0 × elem_bytes` ≡ `tile0 << esize_enc`
- `dim0_offset   = coord0 × elem_bytes` ≡ `coord0 << esize_enc`

Both currently consume a full 32×32 `VX_multiplier` (mul0, mul1 phase 0) plus
their registered-operand FFs. Replacing with a 2-bit variable shift removes two
multiplies' worth of DSP + operand registers + capture muxing. The author
already exploits this exact identity for `per_lane_stride_bytes` (line 128) but
not here. **The genuine multiplies — `coordN × strideN` and the wrap deltas
`(tileN−1) × strideN` — stay on DSP** (that is the correct use of DSP and
answers "ensure multipliers use DSP").

### 2b. Collapse 3 multipliers → 2

After 2a, phase 0's only true multiply is `coord1 × stride0` (mul2). Phases 1–3
already use just mul0 (delta) + mul1 (offset). So a single genuine-multiply pair
{mul_offset, mul_delta} covers every phase; **mul2 is deletable** once the two
phase-0 shift-multiplies move to shifters. Net: 3 → 2 `VX_multiplier`
instances. Setup latency is unchanged (phase 0 still needs `coord1×stride0` in
one slot; the shifts are combinational and don't occupy a multiplier). DSP usage
drops by however many DSPs the third 32×32 consumed (≈2).

### 2c. `staged_*` / `active_*` register duplication (note, lower priority)

Every parameter exists twice: `s_*` (staged, next transfer) and `r_*` (active,
live). This is ~17 fields incl. three `[4][32]` arrays — on the order of ~500
FFs duplicated to let setup overlap drain. It is **FF cost, not LUT** (the board
is LUT-bound, FF 5375 has headroom), so deleting the staging would help the
wrong resource while reintroducing a 4–10-cycle setup bubble per transfer.
**Recommend keeping the staging.** Flagged only for completeness.

### 2d. Decode-mux dedup (modest LUT)

`dec_tile{1..4}`, `dec_stride{0..3}`, `dec_size{1..4}` each carry a
`(dec_rank >= k) ? field : default` mux — 13 rank-comparator+mux cones. The
rank compare is shared-able: derive `rank_ge2..rank_ge5` once (4 bits) and index
with those instead of re-evaluating `dec_rank >= k` in each. Minor, but it is
pure dedup with no behavioral change.

---

## 3. `VX_dxa_smem_wr` — kill the runtime multiply, bound the shifters (point 8)

### 3a. `beat_offset = stride_words × visit_count` → accumulator (definite win)

Line 475: `beat_offset = stride_words * SMEM_ADDR_WIDTH'(visit_count)` is a
**runtime multiply** synthesized in LUTs (raw `*`, not a `VX_multiplier`/DSP).
But `visit_count` is a per-beat counter that increments by exactly 1 per
multicast receiver and resets at each word boundary. So the product is a running
sum: **maintain `beat_offset_r`, add `stride_words` each replay beat, reset to 0
at word boundary.** This deletes the multiplier entirely (no DSP, no LUT
multiply) and replaces it with one `SMEM_ADDR_WIDTH` adder — strictly cheaper,
and it removes a multiply from the multicast address path. Pure win, no
behavioral change (the replay walk is already a deterministic 1-receiver/beat
iteration).

### 3b. Barrel shifters are the dominant logic-LUT mass (inherent, partial)

The real LUT cost here is the gather/scatter:
- load shift `fb_load_data_compressed << new_bit_offset` (byte-granular, ≤ CL),
- K-major scatter `km_elem_bytes_slice << (km_in_word_off << 3)`,
- sw-capture compress `sw_payload >> {sw_byte_offset,3'b0}`.

These are intrinsic to unaligned CL→SMEM placement and cannot be removed without
changing the data contract. Two bounded mitigations:
- The K-major drain shift is already correctly bounded to `GMEM_DATAW` (line
  311) via `verilator split_var`; the **row-major** load shift operates on the
  full `FILL_CAP*8` width. Since `new_bit_offset < SMEM_WORD_SIZE*8`, the live
  result is bounded — confirm Vivado isn't synthesizing a full-width
  `FILL_CAP*8` barrel when only `SMEM_DATAW + offset` bits can be nonzero, and
  add an explicit width bound if it is.
- `rm_byteen` is a per-byte genvar comparator forest (`SMEM_WORD_SIZE` cones).
  For row-major it computes a leading-mask + trailing-mask; this is a
  prefix/threshold pattern (`i >= offset && i < level`) that can be a single
  decoder pair instead of N independent compares. Modest, structural.

These are **logic** wins (the part that actually dominates the 13k LUTs), but
they are refinements, not a single coarse-grain collapse — the gather/scatter
shifter is load-bearing.

---

## 4. `VX_dxa_gmem_req` — already lean (review point 7)

Slot table is 8×~29 b (registers, combinational read), two 8-bit priority
encoders, busy/oob bitvector maintenance, one outstanding counter. No
duplicated logic, no oversized arithmetic, nothing BRAM-eligible at depth 8.
**No substantial coarse-grain win exists here** — calling it out explicitly so
we don't churn it for sub-1% noise. (The 8×29b slot table *could* go to a
`VX_dp_ram`, but at depth 8 that is LUTRAM either way — no benefit.)

---

## 5. `VX_dxa_req_arb` OUT_BUF (review point 5)

**Current:** `VX_dxa_core` instantiates `req_arb` with `OUT_BUF` defaulted to 0.
For the DUT (NUM_REQS=1, NUM_OUTPUTS=1) an unregistered 1:1 passthrough is
correct and costs nothing. But the rule "register the arb output when inputs or
outputs > 1" is not enforced — a multi-socket build (NUM_REQS>1) would synthesize
an unregistered N:1 arbitration cone straight into `req_queue`, a timing risk.

**Change:** make it conditional in `VX_dxa_core`:
```
.OUT_BUF ((NUM_REQS > 1) ? 3 : 0)   // 3 = registered (skid) output
```
Zero DUT-area change (NUM_REQS=1 → 0), correct for multi-socket. (`VX_dxa_unit`'s
`dxa_req_buf` is already a 2-deep elastic buffer, so the issue side is covered;
this is purely the core-side arb.)

---

## 6. `VX_dxa_rsp_buf` — delete dead code (review point 3)

Instantiated nowhere. Its header comment is also self-contradictory ("LUTRAM=0
forces BRAM" above a `LUTRAM(1)` instantiation), confirming it drifted out of
use during the direct-drain refactor. **Recommend deleting the file.** Zero area
change (already absent from synthesis); removes a misleading reference. If it
were ever revived, `LUTRAM=1` is actually correct for its 512b×8 shape (BRAM
would be absurdly underfilled at depth 8).

---

## 7. `VX_dxa_completion` `compl_fifo` (review point 2)

Not in the DUT (lives in `VX_mem_unit`). On its own merits:
- **BRAM?** No. DATAW=`BAR_ADDR_W` (~9 b), DEPTH=`NUM_WARPS` (32) → 288 bits,
  below `FORCE_BRAM`'s threshold and absurd to spend a RAMB on. Keep LUTRAM.
- **Needed?** Yes, but possibly shallower. It decouples the multicast release
  burst (≤ popcount(cta_mask) events, emitted ≤ 1/cycle on the last drain word)
  from the txbar consumer's accept rate. The depth-`NUM_WARPS` worst case
  assumes the consumer can stall for the entire burst. If `txbar_bus_if.ready`
  is guaranteed ≥ 1 accept / 2 cycles (it is, per the module's own comment),
  the standing queue never exceeds a few entries and DEPTH could drop to e.g. 4
  with an assertion — saving LUTRAM in the full chip. **Not a BRAM target; a
  depth-tuning note for the full-chip build, out of DUT scope.**

---

## 8. Prioritized plan and expected deltas

Ordered by (LUT saved / risk). Deltas are estimates pending re-synthesis of
`dut/dxa` (NT32/NW32); each lands as its own `nt32nw32_area{N}` build.

| # | Change | Module | Resource move | Risk |
|---|--------|--------|---------------|------|
| 1 | `beat_offset` multiply → accumulator | smem_wr §3a | −LUT (mult), −0 DSP | low — local, behavior-identical |
| 2 | ×elem_bytes multiplies → shifts; 3→2 multipliers | setup §2a/2b | −~2 DSP, −operand FFs, −LUT | low — power-of-2 identity |
| 3 | `desc_table` `OUT_REG=1` + fetch stage | desc_table + core §1 | −~350 LUTRAM, +~11 BRAM | med — 1-cycle launch handshake |
| 4 | `req_queue` `LUTRAM=0` | core §1b | −~230 LUTRAM, +~5 BRAM | low — param flip |
| 5 | `req_arb` `OUT_BUF` conditional | core §5 | 0 (DUT); timing-safe multi-socket | low |
| 6 | delete `VX_dxa_rsp_buf` | — §6 | 0 (cleanup) | none |
| 7 | decode-mux + `rm_byteen` dedup | setup/smem_wr §2d/3b | −LUT (modest) | low |

**Headline outcome:** items 3+4 take LUTRAM 584 → ~0 and BRAM 0 → ~16, directly
fixing the "0 BRAM" imbalance. Items 1+2 convert the remaining hidden multiplies
to shifts/adders/DSP and free ~2 DSP + operand FFs. Item 7 chips at the
gather/scatter logic-LUT mass. The intrinsic barrel-shifter LUTs in `smem_wr`
(§3b) are the floor — they are the cost of unaligned CL→SMEM placement and
cannot be rebalanced into BRAM/DSP.

**Honesty caveat:** total LUT is dominated by §3b logic, not the 584 LUTRAM, so
expect the BRAM rebalance (items 3+4) to drop *total* LUT by ~500–600 (≈4%),
not to transform the number. The multiply/shift cleanups (1+2) and decode dedup
(7) are what trim the logic-LUT mass; their magnitude is uncertain until synth.

---

## 9. Validation plan

1. rtlsim `--dxa` suite (38/38) green after **each** item — no functional drift.
2. Re-synthesize `dut/dxa` per item; record LUT/LUTRAM/FF/DSP/BRAM + WNS in the
   build table (extend the `dxa_addr_gen_timing.md` progression).
3. Confirm WNS stays ≥ 0 (the desc_table fetch stage and the smem_wr accumulator
   touch launch/multicast paths, not the closed addr_gen recurrence — but the
   BRAM read on the launch path must be re-timed).

---

## 10. Measured results (`nt32nw32_area`)

Implemented all of §1–§6 (the §3b/§7 logic-LUT refinements were deferred —
`smem_wr` barrel shifters left intact). Re-synthesized NT32/NW32 @ 300 MHz and
re-ran the full `--dxa` suite (**38/38 PASS**).

| Metric | `fix7` (before) | `area` (after) | Δ |
|---|---:|---:|---|
| WNS | +0.012 ns | **+0.017 ns** (MET) | +0.005 |
| Fmax | 301.1 MHz | **301.6 MHz** | +0.5 |
| Total LUT | 13627 | **13148** | −479 (−3.5%) |
| Logic LUT | 13043 | 13130 | +87 |
| LUTRAM | 584 | **18** | −566 (−97%) |
| FF | 5375 | 5624 | +249 |
| DSP | 10 | **7** | −3 |
| BRAM | 0 | **12×RAMB36 + 1×RAMB18** | +13 tiles |

Per-module: `desc_table` LUTRAM→0 / **9 RAMB36**; `req_queue` LUTRAM→0 /
**3 RAMB36 + 1 RAMB18**; `setup` **6 DSP** (was ~9); `smem_wr` **0 DSP** (the
accumulator) and **9644 LUT = 73% of the DUT** — confirming §0/§3b: the
gather/scatter barrel shifter is the LUT floor and the next coarse-grain target.
The 18 residual LUTRAM is `gmem_req`'s 8-entry slot table. As predicted (§0/§8),
the BRAM rebalance fixes the "0 BRAM" imbalance and trims total LUT ~3.5%; the
bulk LUT mass remains `smem_wr` logic.

### Postmortem: descriptor fetch-stage deadlock (found + fixed)

The first integration of §1's registered-read `desc_table` shipped a 1-deep
fetch *register* that drove `desc_read_addr` from the live queue head while
holding a request under dispatch back-pressure. When dispatch stalled (frequent
for apps streaming many back-to-back transfers — `sgemm*`), the held request and
the registered `desc_read_data` drifted out of alignment → wrong transfer
geometry → the transfer never completed → barrier never released → scheduler
timeout (`active_warps=stalled_warps`). `dxa_copy` (few transfers, even 5-D +
multicast) never stalled dispatch, so it masked the bug in single-test smokes;
only the full suite's streaming GEMMs exposed it (7 rtlsim deadlocks).

**Fix:** a 2-state fetch FSM (`FETCH_IDLE`/`FETCH_PRESENT`) that holds the queue
head — and therefore `desc_read_addr`, hence `desc_read_data` — stable until
dispatch consumes, popping only on consume. Request and descriptor are then
always presented in alignment, even under back-pressure. Launch latency is +1
cycle, negligible against a multi-hundred-cycle drain. Lesson: a registered-read
RAM behind a stallable consumer needs the *address* held for the read's whole
in-flight window, not just the data latched for one cycle.

---

## 11. `smem_wr` barrel-shifter rebalance (the LUT floor)

§10 left `smem_wr` at 9644 LUT (73% of the DUT). A netlist breakdown localized
it: **`fb_data` = 6631 LUTs (66% of smem_wr)** — two *variable* barrel shifters
writing the 640-bit fill buffer `fb_data_r`:
- row-major **load** shift (`<< new_bit_offset`, position the CL at its in-word
  byte offset), and
- K-major **drain** shift (`fb_data_r >> elem_bytes*8` every beat) — also the
  binding timing path (`elem_bytes → km_shift_bits → fb_data_r`).

Both are removed from `fb_data_r`:

1. **K-major → read pointer.** `km_rd_off_r` walks a 64-bit read window
   (`fb_data_r[km_rd_off_r*8 +: 64]`) instead of shifting the register each beat.
   The per-beat variable shift into `fb_data_r` is gone (and with it the worst
   timing path).
2. **Row-major → one capture-stage positioning shift.** The capture *compress*
   (`>> byte_offset`, drop the CL leading offset) and the fb-load *reposition*
   (`<< smem_off`) merge into a **single** shift at capture. A constant pre-shift
   by `POS_BIAS = SMEM_WORD_SIZE-1` bytes keeps the combined right-shift amount
   non-negative for both modes (`POS_BIAS + byte_offset − smem_off` row-major;
   `POS_BIAS + byte_offset` K-major). `pend`/`defer` now hold pre-positioned
   payloads (widened to `FILL_CAP*8`), so **fb-load is a plain copy**.

`fb_data_r`'s next-state is then only *copy / fixed whole-word shift / hold* — no
variable barrel shift. The single remaining variable shifter (at capture, into
`pend`) replaces the old compress; the byteen masking (`rm_byteen`,
`fb_byte_offset_r`) is unchanged.

### Result (`nt32nw32_area2`, 38/38 PASS, WNS +0.012 MET)

| Metric | `area` | `area2` | Δ |
|---|---:|---:|---|
| **`smem_wr` LUT** | 9644 | **7035** | **−2609 (−27%)** |
| Total LUT | 13148 | **11451** | −1697 (−12.9%) |
| FF | 5624 | 7626 | +2002 (abundant) |
| WNS | +0.017 | +0.012 | still MET |
| DSP / BRAM / LUTRAM | 7 / 13 / 18 | 7 / 13 / 18 | — |

The worst path moved to the capture positioning shift (`gmem_req →
pend_data_r`), which closes at +0.012. The FF rise is the wider pre-positioned
`pend`/`defer` (FF is 0.29% used — no constraint). `smem_wr` remains the largest
block (7035 LUT), now mostly the capture shifter + the multicast/byteen logic;
further reduction would need a narrower capture shift or output-side windowing
(timing-sensitive — the drain/output path is fast today and worth keeping so).

### Cumulative vs the pre-optimization `fix7` baseline

| Metric | `fix7` | final (`area2`) | Δ |
|---|---:|---:|---|
| Total LUT | 13627 | **11451** | −2176 (−16%) |
| LUTRAM | 584 | 18 | −97% |
| DSP | 10 | 7 | −3 |
| BRAM | 0 | 13 tiles | +13 |
| WNS | +0.012 | +0.012 | MET |
