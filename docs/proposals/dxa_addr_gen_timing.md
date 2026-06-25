# DXA Address-Generator Timing — U55C @ 300 MHz

**Target:** xcu55c-fsvh2892 (-2L), 300 MHz / 3.333 ns, Vivado 2024.1.
**DUT:** `VX_dxa_core_top` (`hw/syn/xilinx/dut/dxa`, NT32/NW32).
**Status going in:** WNS **−0.476 ns** (build `nt32nw32_fix4`), after the
six `VX_dxa_smem_wr` / `VX_dxa_addr_gen` fixes already committed
(`6f3b76b7`). All remaining failure is in `VX_dxa_addr_gen`.

---

## 1. The single violating critical path

`report_timing -nworst 100` returns **100 paths and one topology.** Every
endpoint is the same register-to-register family inside `addr_gen`:

```
sources : gmem_cursor_r / row_len_r / line_idx_r        (row + beat state)
   │
   ▼  bytes_span = first_off + row_len_r                (32-bit add, CARRY8)
   ▼  num_lines  = (bytes_span + LINE-1) >> CL_OFF_BITS  (CARRY8 ×4)
   ▼  is_last_line = (line_idx_r + 1 >= num_lines)       (CARRY8 compare, fo=38)
   ▼  valid_end / valid_start  → cur_valid_length        (mux + subtract)
   ▼  km_step_in_row = cur_valid_length * km_tile1_r     (DSP48E2 multiply+ALU)
   ▼  smem_byte_addr_r0_i_32                             (post-DSP LUT)
dest    : smem_byte_addr_r  (FF)
```

Worst slack **−0.476 ns**, data-path delay **3.790 ns**, 16 logic levels
(`CARRY8=5  DSP_*=6  LUT=5`). Delay budget from the report:

| Segment | ~delay | Note |
|---|---:|---|
| `gmem_cursor_r` → `num_lines` (add + shift) | **1.14 ns** | CARRY8 ×5, **row-constant** |
| `num_lines` → `is_last_line` → DSP `B` input | 0.39 ns | fanout-38 compare + 2 LUTs |
| DSP48E2 multiply + fused ALU (+`smem_byte_addr_r`) | 1.85 ns | inherent, ~constant |
| DSP `P` → post-LUT → FF `D` | 0.11 ns | width fix-up |

### Why this is the only path

`num_lines`, `last_end_off`, `last_aligned` (the last-CL `valid_end`
selectors) are all pure functions of `first_off + row_len_r`. Both
operands are **row-constants** — `row_len_r` is fixed per transfer and
`first_off = gmem_cursor_r[CL_OFF_BITS-1:0]` only changes at a row wrap.
Yet today they are recomputed combinationally **every beat** and sit in
series ahead of the DSP multiply that updates `smem_byte_addr_r`. The
1.14 ns add/shift chain is being paid on every CL even though its value is
identical for all CLs of a row.

---

## 2. Optimization — register the inputs, keep the one DSP

The DSP itself fits the budget (~1.85 ns < 3.333 ns). It fails only because
~1.5 ns of combinational geometry (`num_lines` add + `is_last_line` compare
+ `valid_length` select) is **stacked in front of it**. The fix is therefore
not to *remove* the DSP — which would cost a second multiplier for the
boundary steps and grow the datapath — but to **feed it from registers** so
the long chain is no longer in series with the multiply. This keeps the
single DSP (area-neutral) and was chosen over a DSP-free precompute-and-mux
scheme precisely to avoid the extra multiplier and respect block reuse.

Three registers, all updated only at **row entry** (`start` / each wrap)
except the last, which advances per beat one step ahead:

- `num_lines_r` — CLs spanned by the current row (row-constant).
- `last_vend_r` — `valid_end` for the row's last CL
  (`last_aligned ? LINE : last_end_off`), the only row-dependent input to
  the `valid_end` mux (row-constant).
- `is_last_line_r` — whether the **currently presented** CL is the row's
  last. Computed one beat ahead from the *next* state, so the 32-bit
  `line_idx + 1 >= num_lines` compare never sits in front of the DSP.

`num_lines_r`/`last_vend_r` are computed from `span = first_off + row_len`:
at `start` from `setup_params`, at a wrap from the post-step cursor
`next_cursor = gmem_cursor_r + step_delta` (already computed for the
`gmem_cursor_r` update — reused, no extra adder). `is_last_line_r` advances:
`(line_idx+2 >= num_lines_r)` on an interior step, `(wrap_num_lines <= 1)`
on a wrap, `(start_num_lines <= 1)` at `start`.

This splits the one long path into short register-to-register paths, none of
which has the add-chain *or* the compare in series with the DSP:

1. `gmem_cursor_r` → `span` → `{num_lines_r, last_vend_r}` — the 32-bit
   add/shift, **no DSP** (wrap-only, ends at a register).
2. `line_idx_r` / `num_lines_r` → `is_last_line_r` — the 32-bit compare,
   **no DSP** (ends at a register).
3. `is_last_line_r` (FF) → small mux → `cur_valid_length` (≤ line-size
   subtract) → **DSP** → `smem_byte_addr_r` — the DSP path is now
   `register → mux+sub → DSP`, ~2.3 ns, ~+1.0 ns of slack.

### Correctness

`num_lines`/`last_vend` are invariant across a row, so a value latched at row
entry is correct for every beat of that row. `is_last_line_r` is seeded at
`start` for CL0 and re-derived on every advance for the CL about to be
presented (interior, wrap-to-new-row, and single-CL-row cases all covered),
so it equals the combinational `is_last_line` of the original design beat for
beat. `out_last` and `out_valid_length` are therefore bit-identical; only
their source moves from logic to registers.

### Cost

Three registers (`32 + (CL_OFF_BITS+1) + 1` bits). **No new DSP** (the same
single multiplier as today), no new BRAM; the wrap-time `span` reuses the
existing `next_cursor` adder. Net LUT usage **drops** — the per-beat
`bytes_span`/`num_lines`/`total_end`/compare cones collapse to one wrap-time
copy and a registered bit.

---

## 3. Validation plan

1. rtlsim regression: `dxa_copy`, `dxa_copy_mcast`, `dxa_kmajor_check`,
   `sgemm2_dxa_mcast` (row-major + K-major + multicast + boundary CLs).
2. Re-synthesize `dut/dxa` (NT32/NW32) → confirm WNS ≥ 0.
3. Full `--dxa` suite.

Expected: per-beat DSP path ≈ `register + mux + sub + DSP + post-LUT` ≈
2.3 ns vs 3.333 ns budget (≈ +1.0 ns slack) — closes the −0.476 ns gap with
real margin, not the ~+0.14 ns a `num_lines`-only fix would scrape by with.

### Result (incremental patch, `nt32nw32_fix5`)

WNS **−0.476 → −0.153 ns**, and the worst path **moved off the DSP**: it is
now `dim_count_r → … → is_last_line_r` (15 levels, CARRY8=8) — i.e. the
**once-per-row odometer→next-row `num_lines` wrap chain**. The DSP recurrence
is no longer the binding constraint; the remaining violation is the
geometry compute at a row boundary. This is what §4 attacks.

---

## 4. Two-loop counter architecture (the real fix)

Stepping back from the per-path patching: the block mixes an **inner loop**
(CL within a row, every beat) with an **outer odometer** (dims 1–4, only at a
row wrap) and rebuilds the outer quantities combinationally every beat. The
restructure separates them so the inner loop is pure counters/flags and all
the 32-bit arithmetic sits on once-per-row paths.

**Inner loop — counters/flags, no 32-bit compare, no DSP:**

- `cl_addr_r` — GMEM CL address, **++1/beat**, reload `next_cursor>>CL_OFF` at
  a wrap. Replaces the per-beat `(gmem_cursor>>CL_OFF)+line_idx` shift+add.
- `lines_left_r` — CLs remaining, **−−1/beat**, load `num_lines` at row entry.
  `is_last_line = (lines_left_r == 1)` — a register-fed **equality-to-1**,
  which *subsumes* the `is_last_line_r` pipelining of §2.
- `is_first_r` — set at row entry, cleared after the first beat. Replaces the
  two `line_idx == 0` cones (`byte_offset`, `valid_start`). `line_idx_r` is
  deleted entirely.
- `smem_byte_addr_r += km_step`, where the K-major step is a **2:1 mux**:
  - interior CL: `km_step_full = tile1 << log2(LINE)` — a **shift**, no multiply
    (`valid_length == LINE`).
  - first CL: `(LINE − first_off) · tile1` — one multiply fed **directly by
    registers** (`first_off`, `tile1`), both row-stable, so `reg → DSP → mux`.
  - last CL: don't-care (the wrap overrides `smem_byte_addr_r`).
  The long `valid_length` cone never precedes the multiply; the steady-state
  per-beat path carries no DSP at all.

**Outer odometer — unchanged, runs only at a wrap:** `dim_count`,
`gmem_cursor`, `num_lines`, `step_delta`, OOB, last-outer. `is_oob` /
`is_last_outer` stay combinational off `dim_count_r` (row-stable, off the
critical path).

This deletes `line_idx_r`, `num_lines_r`, `is_last_line_r`, and the per-beat
CL-address adder; the inner loop's only arithmetic is two ±1 counters. Net
LUT drops sharply (the per-beat 32-bit comparator forest is gone) and the
`is_last_line` compare disappears. The remaining binding path is the
wrap-time `num_lines` load (`lines_left_r`), now with a simpler register
fan-in than `fix5`'s `is_last_line_r` (no parallel `line_idx+2 >= num_lines`
compare). Synthesized as `nt32nw32_fix6`.

### Result (two-loop rewrite, `nt32nw32_fix6`)

WNS **−0.153 → −0.031 ns** (≈297 MHz). Area is essentially flat vs `fix5`
(addr_gen 1485 vs 1433 LUTs, 1 DSP unchanged) — the win here is *timing*, not
LUTs; the inner loop's combinational adder/compare were already modest, so the
counter form trades like-for-like while pulling the DSP and the `is_last_line`
compare off the path. The remaining violation is purely the once-per-row wrap
chain `dim_count[2] → dim2_steps → step_delta → next_cursor → num_lines →
lines_left_r` (logic 1.58 ns, route 1.77 ns).

## 5. Closing the wrap chain — equality, not magnitude

The binding segment is the odometer predicate `dim_count[N]+1 < dim_tile[N]`,
synthesized as a +1 plus a 32-bit magnitude compare (a CARRY8 chain) feeding a
fanout-37 `dim2_steps`. But an odometer index only ever reaches a **fixed**
bound, and it increments by 1, so "at the bound" is an **equality**:

```
with dim_count ∈ [0, tile-1]:  (dim_count+1 < tile) ⟺ (dim_count != tile-1)
                               (dim_count+1 >= tile) ⟺ (dim_count == tile-1)
```

Storing `dim_last_r = tile-1` at setup and testing equality replaces each
CARRY8 magnitude chain with a ~2-level XOR/NOR. `dim{0,1,2}_steps` become
`~dim{0,1,2}_last`, and `is_last_outer` is the AND of the four `dim*_last`
equalities. This is a pure combinational simplification — no pipelining, no
corner cases (unused dims have `tile=1 → tile-1=0`, so `dim_count==0` reads as
last, exactly as before). It collapses segment A of the wrap path. Synthesized
as `nt32nw32_fix7`.

A heavier alternative considered and rejected for now: **row-geometry
prefetch** (compute `step_delta`/`next_cursor`/next-`num_lines` one wrap ahead
from the row-stable `dim_count_r` and load registers at the wrap). It removes
the whole chain but needs per-odometer-branch next-state predicates to cover
single-CL rows; the equality rewrite is far cheaper and expected to suffice.
