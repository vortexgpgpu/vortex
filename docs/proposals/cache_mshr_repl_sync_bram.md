# Native sync-BRAM (OUT_REG=1) for VX_cache_mshr and VX_cache_repl

## Goal

Convert the data RAMs in `VX_cache_mshr` (`mshr_store`) and `VX_cache_repl`
(`plru_store` / `fifo_store`) from the **async-patch BRAM** path
(`OUT_REG=0`, `RADDR_REG=1` → `VX_async_ram_patch` netlist surgery) to a
**native synchronous BRAM** (`OUT_REG=1`), **without changing cache hit
latency**, in order to **reduce total area** (reclaim the per-instance LUT
overhead the patch carries) and shed the synthesis-time patch dependency.

## The proven template already lives in this cache: `cache_tags`

`VX_cache_tags.tag_store` is already a native sync BRAM and is the pattern to
copy:

```
VX_dp_ram #(.OUT_REG(1), .RDW_MODE("R")) tag_store (
    .read(~stall), .write(|line_write),
    .waddr(line_idx),      // registered current line (write)
    .raddr(line_idx_n),    // look-ahead: NEXT line, one cycle early (read)
    ...);
// Read-during-write hazard handled by registered bypass flags:
`BUFFER(rdw_fill, do_fill);
`BUFFER(rdw_write, do_write && (line_idx == line_idx_n));
//   tag_matches = raw_hit || rdw_fill;  read_dirty = ram_dirty || rdw_write;
```

The trick: a synchronous BRAM is 1-cycle, but driving its address with the
**look-ahead** value (`line_idx_n`, the next cycle's address, available
combinationally from `sel_req` before the S0 register) makes the registered
read present `ram[line_idx]` *in the same cycle the async read would have* —
bit/cycle identical, no added latency. RDW collisions (write to `line_idx`
vs. prefetch read of `line_idx_n`) are forwarded by registered bypass flags.

This is the same look-ahead technique already validated on `VX_fifo_queue`
(drive the BRAM with `rd_ptr + pop`).

## Hit-latency safety (the hard constraint)

Both target reads are **miss/fill/replay-path only** — a cache *hit* never
consumes them, so hit latency is structurally unaffected:

| RAM | read output | consumed when |
|---|---|---|
| `repl` (plru/fifo) | `victim_way` | `repl_valid = do_fill_st0` (fill only) |
| `mshr_store` | `dequeue_data` | `dequeue_valid = replay_valid` (replay only) |

`victim_way` *is* consumed combinationally at S0 (`evict_way_st0` →
`cache_tags.evict_way`), so it must stay same-cycle → **look-ahead required**
(cannot simply register-and-defer). The MSHR allocate(S0)→finalize(S1)
coupling that the deadlock note protects is **untouched** — only the
`mshr_store` *data* read changes.

## Redesign

### 1. `mshr_store` (primary win) — `VX_cache_mshr`

`dequeue_id_r` is a registered pointer whose next value `dequeue_id_n` is
computed combinationally from the control tables (`next_index`, `fill_id`,
`finalize_id`) and **never from `dequeue_data`** — no read→control feedback,
so it is prefetchable (FIFO-class, unlike `ipdom_stack`).

```
VX_dp_ram #(.OUT_REG(1), .RDW_MODE("R")) mshr_store (
    .read(1'b1), .write(allocate_valid),
    .waddr(allocate_id_r),
    .raddr(dequeue_id_n),          // was dequeue_id_r; look-ahead
    .wdata(allocate_data), .rdata(dequeue_data));
// RDW bypass: dequeue_data MUST be correct (replayed request payload).
`BUFFER(rdw_alloc, allocate_valid && (allocate_id_r == dequeue_id_n));
// forward allocate_data when rdw_alloc fires (mirror cache_tags rdw_write).
```

The `dequeue_data` bypass is **mandatory** (correctness, not just policy
quality): a stale read replays the wrong request. Cost is a handful of LUTs,
as in `cache_tags`.

### 2. `repl` (secondary) — `VX_cache_repl`

- **PLRU (`plru_store`, dp_ram)**: same change as `mshr_store` —
  `OUT_REG(1)`, `raddr = repl_line_n` (new look-ahead port; the bank already
  computes it as `line_idx_n` and passes it to `cache_tags`). A stale PLRU
  read only degrades *replacement quality*, not correctness (any victim is
  legal), so the RDW bypass is optional/quality-only.
- **FIFO (`fifo_store`, sp_ram)**: single-port read-modify-write of a per-line
  counter. `OUT_REG=1` look-ahead needs **two addresses** (read `repl_line_n`,
  write `repl_line`) → must promote `sp_ram`→`dp_ram`. Low value (see area),
  high relative complexity; **recommend deferring** or leaving on the patch.

Bank change: add `repl_line_n` port to `VX_cache_repl`, wire `line_idx_n`
(already present in `VX_cache_bank`) into it — mirrors the `cache_tags` wiring.

## MEASURED RESULT (supersedes the estimate below)

Synthesized `amo0_pre` (baseline) vs `amo0_post` (this change), AMO=0 LLC, 8 banks:

| Metric | amo0_pre | amo0_post | Δ |
|---|---|---|---|
| WNS | MET +0.059 ns | MET +0.031 ns | −0.028 (still meets 300 MHz) |
| LUT | 66,307 | **61,012** | **−5,295 (−8%)** |
| LUTRAM | 11,918 | **9,060** | **−2,858 (−24%)** |
| FF | 90,656 | **85,359** | **−5,297** |
| RAMB36 | 16 | 16 | **0** |
| URAM | 64 | 64 | 0 |

**The estimate below was wrong about the mechanism.** The live cache (cache_v2)
keeps `mshr_store`/`fifo_store` in **LUTRAM** (not BRAM-via-patch as the stale
Jun-22 `amo0_llc_nt32` build showed), because their dimensions don't trip
`FORCE_BRAM`. So `OUT_REG=1` did **not** move them to BRAM — it kept them in
distributed RAM but the native `g_sync` path is simply leaner than the old
`g_async` structure. Net: a **pure reduction** (−5.3K LUT, −2.9K LUTRAM,
−5.3K FF) at **zero BRAM cost**, timing still met — no LUT↔BRAM tradeoff.
Functionally validated: sgemmx -n128 cycle-neutral + 34/34 cache regression
(incl. AMO).

## Area cost estimate (STALE — from `amo0_llc_nt32`: 8 banks, MSHR_SIZE=16)

Per-instance, measured post-route:

| RAM (per bank) | LUT | RAMB36 | RAMB18 | path |
|---|---|---|---|---|
| `mshr_store` | **516** | 8 | 1 | async patch |
| `fifo_store` | ~10 | 0 | 1 | async patch |
| `tag_store` (native ref) | **0** | 2 | 0 | OUT_REG=1 |

Whole `VX_cache_top`: LUT 54,011 (4.14%), **RAMB36 471 (23.36%)**, RAMB18 25.

**Projected delta (OUT_REG=1, the requested change):**

| | LUT | RAMB36 | RAMB18 |
|---|---|---|---|
| `mshr_store` ×8 | **−~3,500 to −4,100** (516→~0/bank, less ~8 LUT bypass) | 0 (stays 64) | 0 |
| `fifo_store` ×8 (if done) | −~80 | 0 | 0 |
| **Total** | **≈ −4,000 LUT (~7% of the cache's LUTs)** | **0** | **0** |

So the requested change is a **pure LUT reclaim (~4K LUT) at zero BRAM cost** —
the patch overhead disappears (native BRAM puts address+output registers
inside the primitive, like `tag_store`'s 0 LUT). It does **not** change the
BRAM footprint. Final numbers will be confirmed by the post-change synth
(`amo0_post`) vs. the `amo0_pre` baseline.

### Bigger-picture note (the real area question)

`mshr_store` is **16 deep × ~576 bits** → its 8 RAMB36/bank sit at **~3% fill**
(16/512 depth). Across 8 banks that is **64 RAMB36 = 13.6% of the cache's BRAM**
held at near-zero utilization. `OUT_REG=1` keeps that BRAM as-is (it only
reclaims LUTs). If **total area / BRAM-relief** is the ultimate objective and
the design is BRAM-bound (the NT=32 core is 44% BRAM), the higher-impact move
is the *opposite*: put `mshr_store` in **LUTRAM** (16 deep is LUTRAM-ideal,
~1 LUT/bit), which **frees all 64 RAMB36** at a cost of ~4.6K distributed-RAM
LUTs. Pick by scarcity:

- **LUT-bound / want the patch gone** → `OUT_REG=1` native BRAM (this proposal): −4K LUT, BRAM unchanged.
- **BRAM-bound** → `LUTRAM=1`: −64 RAMB36, +~4.6K LUT.

`repl` is not an area factor either way (~10 LUT + 1 RB18/bank).

## Risks & validation

- **mshr_store RDW correctness** — the `rdw_alloc` bypass must be exact; this
  is the deadlock-sensitive MSHR. Validate with the full `cache` CI regression
  **including AMO** (AMO=1 path too, even though the area synth uses AMO=0).
- **Sim-visible** — `OUT_REG` changes the behavioral model, so rtlsim
  (sgemmx -n128 smoke + cache regression) genuinely exercises the new timing.
- **FIFO sp→dp promotion** — deferred; only attempt if PLRU/FIFO area matters.

## Recommendation

Implement `mshr_store` (clear ~4K LUT win, mandatory bypass) and `plru_store`
(clean, mirrors tags). Defer `fifo_store`. Measure `amo0_post` vs `amo0_pre`
and run the full `cache` regression (incl. AMO) before committing.
