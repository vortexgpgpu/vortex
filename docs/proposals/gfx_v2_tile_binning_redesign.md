# gfx_v2 — Parallel Tile Binning Redesign (bin-sort)

**Scope:** the on-device, SIMT-parallel triangle binning that replaces the
host-side serial `Binning()`
([sw/runtime/graphics.cpp](../../sw/runtime/graphics.cpp)) — the new on-wire
tile-buffer schema, the SIMT producer pipeline, the 1..N cooperative
rasterizer-consumption model, and the (small) RASTER front-end changes
([hw/rtl/raster/VX_raster_mem.sv](../../hw/rtl/raster/VX_raster_mem.sv),
[VX_raster_te.sv](../../hw/rtl/raster/VX_raster_te.sv)).
**Reference:** Laine & Karras 2011 (CUDARaster) — rejected as template, see §10;
TBDR tile-list binning (PowerVR/Mali/Apple, Larrabee bins).
**Tree:** `~/dev/vortex_v3/gfx_v2` (proposed branch `feature_gfx_v2`).
**Status:** Proposal — implements [gfx_v2_true_gpu_charter.md](gfx_v2_true_gpu_charter.md) §6.2/§6.3.
**Date:** 2026-06-07.
**Related:** [gfx_v2_true_gpu_charter.md](gfx_v2_true_gpu_charter.md),
[graphics_fixed_function_pipeline.md](../designs/graphics_fixed_function_pipeline.md),
[vortexpipe_architecture.md](../designs/vortexpipe_architecture.md),
[command_processor_control_plane.md](../designs/command_processor_control_plane.md).

---

## 1. Motivation

The "true GPU" charter requires binning to run **on the SIMT cores**, in
parallel, with the result consumed by **1..N cooperating RASTER instances**
over **shared, device-resident buffers**, with a **minimal memory footprint**
(no device→host copy, no worst-case preallocation). The current design bins
serially on the host CPU; CUDARaster's per-CTA queues, the "keep-format +
compact" path, and the dense tile×prim bit-matrix all fail one of *parallel*,
*footprint*, or *N-agnostic* (§10).

This proposal adopts a **sort-based ("bin-sort") schema**: coverage is a
single, exact-sized, packed `(bin, prim)` key array made tile-contiguous by a
**sort**, not by lists. The sort *is* the binning — no per-tile lists, no
segments, no compaction copy, no overflow path.

### 1.1 Cost model (what we are optimizing)

Two independent memory terms, scaling differently:

- **Prim records** — `rast_prim_t` (120 B) × *visible* triangles, stored
  **once**. Dominates high-poly scenes.
- **Coverage entries** — one membership per (tile, prim) overlap. Dominates
  large-triangle scenes; every *list/matrix* scheme wastes here via
  duplication, segment slack, dense-but-empty bits, or worst-case reserve.

Target: store each prim record exactly once, and store coverage as the
**information-theoretic minimum** — one packed entry per *actual* overlap,
zero slack, zero preallocation — while minimizing the *number* of entries by
binning coarsely and letting the existing RASTER hardware refine.

---

## 2. Key idea

1. **Bin coarsely; let hardware refine.** [VX_raster_te.sv](../../hw/rtl/raster/VX_raster_te.sv)
   already does recursive tile→block→quad descent, parameterized by
   `TILE_LOGSIZE`. Bin at **128 px** (`BIN_LOGSIZE = 7`) instead of 32 px and
   let the HW descend bin→…→4 px block→quad by re-evaluating edges (cheap, in
   silicon). Coarser bins ⇒ far fewer coverage entries ⇒ smaller DRAM array
   and smaller sort, at the cost of some redundant HW edge-tests on empty
   sub-tiles. This is the dominant footprint lever and costs almost no RTL.

2. **Coverage = a sorted packed-key array.** Each overlap is one key
   `(bin_id << PRIM_BITS) | prim_id`. Sorting ascending groups every bin's
   prims contiguously **and** — because `prim_id` is the low field and ids are
   assigned in submission order — leaves each bin's prims in **draw order for
   free**. Ordering and bucketing both fall out of one deterministic sort; no
   atomics-ordered scatter, no merge stage.

3. **One shared, N-agnostic buffer.** The binning never encodes the rasterizer
   count. 1..N RASTER instances cooperatively consume the same buffer via the
   existing static stripe (§6).

---

## 3. On-wire schema

Replaces `rast_tile_header_t`; `rast_prim_t` is **unchanged**
([sw/common/vx_gfx_abi.h](../../sw/common/vx_gfx_abi.h)).

```c
// Coarse-bin header (sorted by bin_id == binning/scan order).
// bin_x/bin_y stored decoded so the RASTER front-end needs no divide.
struct rast_bin_header_t {
  uint16_t bin_x, bin_y;   // bin coords (× BIN_SIZE = tile origin fed to te)
  uint32_t pids_offset;    // start index into the sorted pid array
  uint32_t pids_count;     // prims overlapping this bin
};

// Consume-facing coverage: a flat array of prim ids, sorted by bin then
// draw order. HW reads u32 pids exactly as today — only header granularity
// changes. (The 64-/32-bit composite keys are a TRANSIENT sort artifact,
// stripped to u32 pids in the header-scan stage; see §4.)
//   sorted_pids : uint32_t[P]
//   primbuf     : rast_prim_t[visible_tris]   (unchanged 120 B records)
```

**Key width (knob, §8).** Baseline **32-bit** composite key:
`prim_id[19:0]` (1 M tris) | `bin_id[31:20]` (4096 bins → up to 8192 px at
128 px). **64-bit** for large scenes: `prim_id[23:0]` (16 M) | `bin_id[63:24]`.
This sets the resolution/scene ceiling that §3.8 of
[vortexpipe_architecture.md](../designs/vortexpipe_architecture.md) flagged for
the old 16-bit fields.

---

## 4. SIMT producer pipeline

Six kernels, all standard data-parallel primitives — the parallelizable
property the design is chosen for. Sequenced by the CP (no host).

| # | Kernel | Work (per thread) | Output |
|---|--------|-------------------|--------|
| 1 | **Setup + count** | 1 thread/tri: frustum cull, clip→subtris, back-face cull, edge eqs (HDC, half-pixel offset), plane eqs `(z/w,u/w,v/w,1/w)`, bin-AABB, min-z. Write `primbuf[pid]`. `n = bins covered` (AABB, or exact overlap for a tighter count). | `primbuf`, `count[pid]` (0 if culled) |
| 2 | **Prefix-sum** | exclusive scan of `count[]` | `offset[pid]`, total `P` → allocate key array of **exactly P** from the resident pool |
| 3 | **Emit** | 1 thread/tri: write its `(bin,prim)` keys at private cursor `offset[pid]++` | composite-key array, size P, zero slack, **no atomics** |
| 4 | **Sort** | LSD radix sort keys ascending | bin-contiguous, draw-ordered keys |
| 5 | **Header scan** | flag `key[i].bin ≠ key[i-1].bin`; compact starts; strip low bits → `sorted_pids[i] = key[i] & PRIM_MASK` | `rast_bin_header_t[]` + `sorted_pids[]` |
| 6 | **RASTER HW** | consume bins as coarse tiles; te/be/qe descent → quads | quads → FS → OM |

Notes:
- **Exact sizing (1→2→3) ⇒ no overflow path.** The count pass yields `P`
  before allocation; nothing can overrun. This is the memory-optimal,
  fully-deterministic variant (vs. single-pass atomic-bump, §8).
- **Clipping** (stage 1) is the one variable-output step: near/guardband clip
  emits 0–7 subtriangles; each becomes its own `prim_id`/record. Handle via a
  subtri count folded into `count[]` (a clipped tri contributes its subtris'
  bin coverage), keeping stages 2–6 uniform.
- **Determinism:** composite-key sort ⇒ bit-exact reproducible output
  independent of thread timing — required by the SimX↔RTL parity work. No
  atomic-order dependence anywhere.

---

## 5. Memory footprint

For `V` visible tris and `P` coverage entries (Σ bins covered):

| Buffer | Size | Lifetime |
|--------|------|----------|
| `primbuf` | `120·V` B | persistent (consume) |
| composite keys | `4·P` B (32-bit) / `8·P` | transient (freed after stage 5) |
| radix scratch | `≤ 4·P` / `8·P` (double-buffer) | transient |
| `sorted_pids` | `4·P` B | persistent (consume) |
| `rast_bin_header_t[]` | `12·(non-empty bins)` B | persistent |

Peak ≈ `120·V + (8..16)·P` during binning; settles to `120·V + 4·P` for
consume. `P` is minimized by the 128 px bins (coarse). No per-tile reserve, no
segment slack, no dense matrix. **`primbuf` (120 B/tri) is the remaining
lever** → prim-record compression (§8), treated as orthogonal follow-up.

---

## 6. 1..N cooperative consumption (shared buffers)

The binning emits **one** N-agnostic buffer set
(`rast_bin_header_t[]`, `sorted_pids[]`, `primbuf[]`). Any `N` RASTER
instances consume it cooperatively by the **existing static stripe**
([VX_raster_mem.sv](../../hw/rtl/raster/VX_raster_mem.sv) `INSTANCE_IDX` /
`NUM_INSTANCES`), unchanged:

```
  instance i of N → bin_headers[i], bin_headers[i+N], bin_headers[i+2N], …
  each header → sorted_pids[offset .. offset+count) → primbuf[pid]
```

`start_tile_count = (count + (N-1-idx)) >> log2(N)`, stride `N`, and the
cluster-global `INSTANCE_IDX = CLUSTER_ID*NUM_RASTER_CORES+i`
([VX_graphics.sv](../../hw/rtl/VX_graphics.sv)) carry over verbatim — the
striping now indexes **bin** headers instead of tile headers. The same binned
buffer runs on a 1-rasterizer or N-rasterizer build with **zero re-binning and
zero data duplication** — the property that selected the flat-sorted
representation over per-producer queues (§10).

**Load balance.** Static stripe = current behavior. If hot bins skew
utilization, the shared `sorted_pids`/headers make a **dynamic work-pull** (an
atomic bin cursor the instances draw from) a drop-in — same buffers, no format
change. Baseline = static stripe; dynamic pull is an optional toggle.

---

## 7. RASTER front-end changes (small)

Consuming the new schema is a bounded edit to the back-end; te/be/slice/edge/qe
are reused as-is.

- **Header type & granularity** — read `rast_bin_header_t` (bin_x/bin_y +
  pids_offset/count) instead of `rast_tile_header_t`. `pids_offset`/`count`
  widen to 32-bit. Tile-fetch FSM in
  [VX_raster_mem.sv](../../hw/rtl/raster/VX_raster_mem.sv) is otherwise
  identical (header → pid → prim-data).
- **Bin size** — `TILE_LOGSIZE` parameter → `BIN_LOGSIZE` (7). `VX_raster_te`
  descends `BIN_LOGSIZE → BLOCK_LOGSIZE` (5 levels at 128→4 px vs 3 today);
  `TILE_FIFO_DEPTH = 1 << (2·(BIN_LOGSIZE − BLOCK_LOGSIZE))` grows — check the
  per-instance FIFO BRAM cost against the U55C budget.
- **pid stream** — `sorted_pids` is plain `uint32`; the existing pid→prim
  address path (`pid × pbuf_stride`) is unchanged.
- **DCRs** — `TBUF_ADDR`/`PBUF_ADDR`/`PBUF_STRIDE`/`SCISSOR` stay static (pool
  addresses + draw state). `TILE_COUNT` (non-empty bin count from stage 5) is
  **read from resident memory**, not a host-baked DCR, since the host can't know
  it pre-submit — see
  [gfx_v2_cp_graphics_frontend.md](gfx_v2_cp_graphics_frontend.md) §5.

All RTL deltas must close **300 MHz on U55C** and are modeled in **SimX
first** as the oracle (§9).

---

## 8. Design knobs

| Knob | Baseline | Trade |
|------|----------|-------|
| **Bin size** | 128 px (`BIN_LOGSIZE=7`) | larger ⇒ fewer keys / smaller sort / more HW descent + bigger te FIFO |
| **Sizing** | exact two-pass (count→scan→emit) | vs single-pass atomic-bump: one less tri pass, but sort restores order anyway; two-pass = zero waste + deterministic |
| **Sort** | composite-key LSD radix | vs counting-sort-by-bin + intra-bin order: radix does bucket + draw order in one deterministic pass |
| **Key width** | 32-bit (≤1 M tris, ≤4096 bins) | 64-bit for large scenes; sets res/scene ceiling |
| **Prim record** | 120 B unchanged | compress (Q?.16 attribs / compact setup + front-end expand) — orthogonal follow-up |
| **Load balance** | static stripe | optional dynamic atomic bin-cursor pull |

---

## 9. Validation & phasing

1. **SimX model first** (oracle): mirror the six kernels in the SimX graphics
   path / a device-side model, reusing `sw/common/gfx_render.cpp` reference
   primitives; pass the existing PNG gfx suite (`tests/graphics/gfx_*`,
   `gfx_raster`/`gfx_draw3d`) bit-for-bit vs host `Binning()`.
2. **SIMT kernels** for stages 1–5 (setup+full triangle setup on-device per
   charter §6.1), validated against the SimX model.
3. **RASTER front-end** (§7) RTL change; SimX↔RTL parity diff; U55C timing.
4. **CP sequencing** (charter §6.4): VS → bin pipeline → RASTER, zero host.

---

## 10. Rejected alternatives (recorded to avoid revival)

| Design | Why rejected |
|--------|--------------|
| **CUDARaster per-CTA queues + merge** | shaped by Fermi's *lack* of HW coverage (we have RASTER); per-CTA segments waste memory, merge stage adds work, and the output **bakes in the producer count** → not N-agnostic (§6). |
| **Keep-format + on-device compaction (Option A)** | flat per-tile PID lists duplicate the pid per covered tile; needs a separate compaction copy. Worse footprint, extra pass. |
| **Tile-major bitmask (per-tile bitset over prims)** | O(tiles×prims) dense scan over sparse data (the global-bbox prune barely helps full-screen); materializes a dense, mostly-zero `tile×prim` bit-matrix (+ per-prim `tileMask`), strictly worse on footprint below ~1.5 % coverage density. Bin-sort is the same idea with **scatter not gather** and **sparse keys not dense bits** — and keeps its one virtue (draw order from key, not bit position). The dense bitset survives only as a transient *in-fine-stage* enumeration detail if ever needed. |

---

## 11. Open items

- **MSAA** sample coverage — deferred; bin-sort is sample-count-agnostic at
  the binning layer (coverage is per-bin), sample expansion lands in the
  fine/OM path (charter §6.8).
- **Primitive types** beyond triangle list — strips/fans/lines expand to tris
  in setup (stage 1) before binning; gated until wired.
- **Guaranteed worst-case bound** — exact sizing makes peak = `120·V +
  16·P_actual`; a residency planner needs a conservative `P` estimate from a
  cheap pre-count (or the setup count pass) to reserve from the pinned pool.
- **prim-record compression** — the next footprint lever after this lands.
