# Bounded per-bank reservation cache (BRAM) for LLC atomics

## Summary

Replace the LLC AMO **per-hart reservation table** — one directly-indexed slot
for every possible hart (`1 << HART_ID_WIDTH`), replicated in every bank — with
a **small fixed set of reservation stations per bank**, line-indexed and stored
in **block RAM (`OUT_REG=1`)**. Default size is **load-matched**:
`max(NUM_WARPS, ceil(NUM_CORES × NUM_WARPS / NUM_LLC_BANKS))` (knob
`VX_CFG_AMO_RS_SIZE`) — it scales with total system concurrency divided across
LLC banks, with a single-core floor of `NUM_WARPS`. This is how real GPUs/CPUs
implement LR/SC: a bounded reservation monitor with spurious SC failures, not a
per-thread table.

This supersedes the earlier "per-line generation" stage-1 idea (which only
removed the associative-invalidate CAM but kept the 1024-slot storage and even
*added* per-hart generation flops). That change is reverted in favor of the
design below.

## Why the per-hart table must go (measured)

`HART_ID_BITS = NC_BITS + NW_BITS + NT_BITS`, so the table depth is
`1 << HART_ID_WIDTH` global harts — **1024** for a 1-core NT=NW=32 LLC, and a
single real SM at 64 warps × 32 threads would need **2048**. It is also
instantiated **per bank**, so an 8-bank LLC stores `8 × NUM_HARTS` reservation
slots even though the whole machine can have at most `NUM_HARTS` live
reservations total — an **8× over-provision**.

Synthesized (xcu55c, 300 MHz, 1024 harts, AMO=1, 8 banks), the reservation
logic dominates the cache:

| | per-hart table (baseline) |
|---|---|
| `VX_amo_unit` / bank | **14,518 LUT, 24,576 FF** |
| × 8 banks | **~116 K LUT, ~197 K FF** |
| share of whole `VX_cache_top` FF | **~70 %** |
| whole cache | 273 K LUT, 315 K FF, WNS −4.46 ns |

No commercial GPU spends ~70 % of its cache flip-flops on LR/SC bookkeeping.
The `VX_CFG_AMO_RS_SIZE = 4` knob in `VX_config.toml` is the vestige of the
original bounded design — it is currently `` `UNUSED_PARAM `` (the RTL ignores
it and builds the full per-hart table). This proposal makes that knob live
again, with a sensible default.

## RISC-V legality (why a bounded set is correct)

The unprivileged spec permits an implementation to hold **few or even a single**
reservation, and allows `SC` to **fail spuriously** for *any* reason (cache
eviction, conflict, another hart's activity). Forward progress is only required
for *constrained* LR/SC sequences (a handful of instructions, no other memory
ops between LR and SC) and is a *system* property — under contention it suffices
that **some** hart's SC succeeds each round, not a specific one. A bounded
reservation set with conflict eviction + SC backoff satisfies this and is
exactly what real hardware ships. Capacity/conflict eviction simply produces
extra spurious failures, which are always legal.

## Design: a line-indexed reservation cache (the `cache_tags` shape)

Per bank, a small **direct-mapped reservation cache** of `NUM_RS` entries,
**indexed by the reserved line's set-index bits** — structurally a miniature
tag array sitting beside `cache_tags`, so it inherits the proven sync-BRAM
look-ahead pattern.

```
localparam NUM_RS      = `VX_CFG_AMO_RS_SIZE;          // default NUM_WARPS
localparam RS_IDX_BITS = `CLOG2(NUM_RS);
localparam RS_TAG_BITS = `CS_LINE_ADDR_WIDTH - RS_IDX_BITS;

// one BRAM, OUT_REG=1, single read + single write port
struct { logic valid; logic [HART_ID_WIDTH-1:0] hart; logic [RS_TAG_BITS-1:0] tag; }
    rs_store [NUM_RS];

wire [RS_IDX_BITS-1:0] rs_idx = res_line_addr[RS_IDX_BITS-1:0];   // index by line
```

Entry width ≈ `1 + HART_ID_WIDTH + RS_TAG_BITS` (~24–30 bits). For `RS=NW=32`
that is `32 × ~28b ≈ 900 bits` per bank — well under one RAMB18.

### Operations (all index by the access line, one BRAM port)

| Op (at commit S1) | Action on `rs_store[rs_idx]` |
|---|---|
| **LR** `(H, L)` | write `{valid:1, hart:H, tag:tag(L)}` — claim the slot |
| **SC** `(H, L)` | `res_check = valid && hart==H && tag==tag(L)`; success path's store clears it via the write rule below |
| **committed write** `(L)` (plain store or AMO RMW) | if `valid && tag==tag(L)` → clear `valid` (breaks the reserver) |

The **tag check on invalidate is essential**: a write to a *different* line that
aliases the same `rs_idx` (`tag` mismatch) leaves the reservation intact — so a
write to line *M* never disturbs a reservation on line *L* in the same set,
preserving exact per-address semantics (the property the old associative scan
provided, now in O(1) with a single indexed compare instead of a
`NUM_HARTS`-wide CAM).

`hart` is stored so `SC` only matches its own LR. Two harts reserving the *same*
line share one slot (the later LR wins) — this is the spec-legal single-
reservation-per-line behavior and is exactly what you want for a lock (one
winner). Two harts on *different* lines that alias the same `rs_idx` conflict-
evict (spurious failure, legal). Set-associativity (`NUM_RS_WAYS`) is an optional
knob to cut conflict evictions; direct-mapped is the simplest BRAM-native point
and the default.

### BRAM with `OUT_REG=1` + look-ahead (no added latency)

`res_check` must be available the same cycle as the `SC` commit decision, but a
sync BRAM read is registered. Use the **`cache_tags` look-ahead**: drive the
read address with the *next* line index (available combinationally from `S0`
before the pipeline register), so the registered output presents
`rs_store[rs_idx]` in the `S1` cycle — bit/cycle identical to a combinational
read, zero added hit/commit latency.

```
VX_dp_ram #(.DATAW(RS_ENTRY_W), .SIZE(NUM_RS), .OUT_REG(1), .RDW_MODE("W")) rs_store (
    .read (1'b1),
    .write(rs_we),                 // LR set | SC/store clear
    .raddr(rs_idx_n),              // S0 look-ahead line index
    .waddr(rs_idx),                // S1 committed line index
    .wdata(rs_wdata), .rdata(rs_rdata));
// RDW bypass (registered, like cache_tags rdw_fill/rdw_write): forward the
// in-flight write when S1 updates the same index S0 is prefetching.
```

The read output (`res_check`) feeds only the SC outcome — it never feeds the
read address — so it is fully **prefetchable** (FIFO-class), unlike
`ipdom_stack`. AMO commits are already serialized by `commit_busy`, so back-to-
back same-line LR→SC collisions are the rare case handled by the registered RDW
bypass.

## Module / interface changes

- **`VX_amo_unit`**: delete `res_valid`/`res_line`(/`res_gen`/`line_gen`) arrays
  and the per-hart logic; instantiate `rs_store` (BRAM) + the three indexed ops.
  Inputs become `{res_reserve, res_clear, res_invalidate, res_hart_id,
  res_line_addr, res_line_idx_n}`; output `res_check`. `NUM_RS` from
  `VX_CFG_AMO_RS_SIZE`.
- **`VX_cache_amo`**: pass the look-ahead line index through (it already has
  `addr_st0`/`addr_st1`); drop the stage-1 `res_slot`/`SLOT_IDX_WIDTH` plumbing.
- **`VX_cache_bank`**: provide `rs_idx_n` from the S0 line index
  (`st0.req.addr[CS_LINE_SEL...]`, the same signal feeding `cache_tags`); revert
  the stage-1 `{set,way}` slot wiring.
- **`VX_config.toml`**: `VX_CFG_AMO_RS_SIZE` default → `NUM_WARPS` (was 4, dead);
  document it as reservation stations per LLC bank.
- Revert the stage-1 generation edits (`res_gen`/`line_gen`/flush-on-wrap).

## MEASURED result (implemented + synthesized)

xcu55c @ 300 MHz, 1024-hart `HART_ID`, AMO=1, 8-bank LLC; RS=32/bank. Validated
11/11 AMO regression (LR/SC conformance, 16-hart LR/SC CAS forward-progress,
RMW ops, writeback, multi-core mc-l2/l3), Verilator-clean.

| Metric | per-hart (baseline) | stage-1 gen | **bounded RS cache** | vs baseline |
|---|---|---|---|---|
| TOP LUT | 273,305 | 217,789 | **102,546** | **−62 %** |
| TOP FF | 315,216 | 385,556 | **113,837** | **−64 %** |
| RAMB36 | 471 | 471 | 471 | 0 |
| RAMB18 | 25 | 25 | **33** | +8 (1/bank) |
| WNS | −4.464 ns | −3.241 ns | **−0.430 ns** | **+4.03 ns** |
| TNS | −438,727 | −1,219,379 | **−1,919** | **229×** |
| `amo_unit`/bank LUT | 14,518 | 10,399 | **1,285** | −91 % |
| `amo_unit`/bank FF | 24,576 | 33,792 | **61** | −99.8 % |
| `amo_unit`/bank BRAM | 0 | 0 | **1 RAMB18** | +1/bank |

The reservation table leaves the critical path entirely (the residual −0.43 ns
is the AMO compute-stage `cmp_addr` high-fanout net to the MSHR/chain-stall
logic — unrelated to reservations).

## Projected area & timing (original estimate)

Per bank the reservation state collapses from `NUM_HARTS×(1+addr)` flops + a
`NUM_HARTS`-wide CAM to `NUM_RS×~28b` of BRAM + a single indexed compare:

| per bank | per-hart (measured) | bounded RS cache (projected) |
|---|---|---|
| LUT | 14,518 | ~tens (1 compare + control) |
| FF | 24,576 | ~tens (control only) |
| BRAM | 0 | ~1 RAMB18 (or LUTRAM if tiny) |

Across 8 banks: roughly **−115 K LUT and −195 K FF**, at **+~4 RAMB36**.
Projected whole-cache: ~273 K → **~155 K LUT**, ~315 K → **~120 K FF**. Timing:
the AMO unit leaves the critical path entirely (no 1024:1 `res_line` mux, no
`NUM_HARTS` flush fan-out, no CAM), so the AMO-rooted violation
(WNS −3.2…−4.5 ns, the worst endpoints were all in `amo_unit`) should clear;
remaining cache timing is unrelated to atomics. Confirm by synth.

## Forward progress & correctness

- **Constrained LR/SC, no contention**: the reserved line maps to one slot;
  unless an aliasing access or another LR/write to that slot intervenes, `SC`
  succeeds. ✓
- **Same-line contention (spinlock)**: serialized through one slot; the latest
  LR-er can succeed, others retry — **system progress guaranteed** each round. ✓
- **Conflict/capacity (aliasing on `rs_idx`)**: spurious `SC` failures, legal;
  mitigate with the standard `SC` backoff. Raise `NUM_RS` or add `NUM_RS_WAYS`
  if a workload shows pathological aliasing.
- **Aliasing-write safety**: invalidate compares the stored `tag`, so a write to
  a different line in the same set does **not** break the reservation. ✓
- **SC ownership**: stored `hart` ensures `SC` matches its own `LR`. ✓

## Risks & validation

- **Forward progress under contention** is the headline risk (this is exactly
  what the per-hart table was over-built to guarantee). Validate with the
  multi-hart same-line LR/SC contention test in addition to functional AMO.
- **RDW correctness** on the `rs_store` look-ahead (LR-set vs SC/store-clear at
  the same index) — mirror the `cache_tags` registered bypass; cover with AMO=1.
- **Full `cache` + `amo` regression** (simx oracle + rtlsim, incl. `wb*`,
  `threads8`, `mc-l2/l3`), then an `xilinx/dut/cache` synth (AMO=1, realistic
  hart count) to confirm the area/timing projection.

## Recommendation

Adopt the bounded reservation cache as the AMO reservation implementation:
`RS = NUM_WARPS` per bank, line-indexed, BRAM `OUT_REG=1` with `cache_tags`
look-ahead. It is RISC-V-legal, matches commercial GPU practice, removes ~70 %
of the cache's flops and the AMO timing path, and reactivates the long-dead
`VX_CFG_AMO_RS_SIZE` knob as a real area/robustness dial. Revert the per-hart
table and the stage-1 generation experiment.
```
                 per-hart        stage-1 gen        bounded RS cache (this)
 storage/bank    1024×33b FF      1024×49b (FF+LUTRAM)   NUM_RS×~28b BRAM
 invalidate      NUM_HARTS CAM    O(1) gen bump          O(1) indexed tag-compare
 LUT/bank        14.5K            10.4K                  ~tens
 FF/bank         24.6K            33.8K                  ~tens
 fwd-progress    by over-provision  by over-provision    bounded + spurious-fail+backoff (real-HW)
```
