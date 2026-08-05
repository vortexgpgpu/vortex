# cgo27_motivation

Motivation harness for the CGO'27 paper. Runs the **same GEMM** (`D = C + A·B`,
A row-major / B col-major, fp16→fp32) through many HW execution paths on the
**same input**, per-path cycles + MPM counters, each verified vs a CPU reference.
The point: find the workload/size where the **optimal HW path changes**.

## HW config (set in `Makefile`; overrides `VX_config.toml` defaults)

Chosen to look like a **~1/4 H100 SM per core** (occupancy), scaled to a small
open GPU. Memory sized proportionally to Hopper (see below).

| Knob | Value | `VX_CFG_*` |
|---|---|---|
| Clusters | 1 | `NUM_CLUSTERS=1` |
| Cores | 4 | `NUM_CORES=4` |
| Socket size | 1 | `SOCKET_SIZE=1` |
| Warps / core | 16 | `NUM_WARPS=16` |
| Threads / warp | 32 | `NUM_THREADS=32` |
| Issue width | 4 | `ISSUE_WIDTH=4` |
| L1 dcache | 32 KB | `DCACHE_SIZE=32768` |
| LMEM (shmem) | 64 KB | `LMEM_LOG_SIZE=16` |
| L2 cache | 1 MB, enabled | `L2_ENABLE`, `L2_CACHE_SIZE=1048576` |
| ICache | 16 KB | (default) |
| Barriers | 32 | `NUM_BARRIERS=32` (2-stage needs 2/CTA, 3-stage needs 3/CTA) |
| Clock | 400 MHz | (sim default) |
| XLEN | 64 | |

Extensions enabled: `EXT_TCU`, `EXT_DTCU`, `EXT_DXA`.

### Hopper-proportional memory rationale

Per core = 16×32 = **512 threads, issue 4** ≈ **1/4 of an H100 SM** (2048 threads,
issue 4). On-chip memory scaled ×1/4; global L2 scaled by core count (4/132).

| Memory | H100 / SM | scale | here |
|---|---|---|---|
| L1 dcache | ~128 KB | ×1/4 | 32 KB |
| Shared mem (LMEM) | up to 228 KB | ×1/4 ≈ 57 | 64 KB (2^16) |
| L2 (global) | 50 MB (÷132 SM) | ×4/132 ≈ 1.5 MB | 1 MB |
| Register file | 256 KB | ×1/4 | ~64 KB (not tuned here) |

## HW modes (`arg->mode`)

| # | Path | Notes |
|---|---|---|
| 0 | in-core SIMT | scalar MAC, software fp16→fp32 (no HW fp16 in SIMT) |
| 1 | in-core TCU (WMMA) | naive: load frag → mma per K |
| 2 | in-core TCU + DXA | naive: single-buffer, sync per K |
| 3 | DTCU_cluster | one engine per cluster, D → L2, native tile 64×32 |
| 4 | DTCU_socket | one engine per socket, D → that socket's L1, native tile 32×16 |
| 5 | in-core TCU + DXA — pipelined | **3-stage** smem pipeline: DXA runs 2 tiles ahead |
| 6 | in-core TCU + DXA — pipelined | **2-stage** smem pipeline: DXA runs 1 tile ahead (ref: sgemm2_dxa) |
| 7 | hetero: SIMT+TCU+DTCU | partition output tiles across units (Phase C) |
| 8 | hetero: all units | (Phase C) |

## Apps (`arg->prologue` / `arg->epilogue`) — 8 total

1. baseline `D = C + A·B`
2. + ReLU
3. + GELU
4. + Residual (`+ R`)
5. + Scale (per-channel)
6. + Softmax (row-wise; cross-tile reduction)
7. dequant(int8→fp16) + bias + GELU
8. dequant(int8→fp16) + softmax

In-core modes fuse pro/epilogue into the operand load / output store. DTCU modes
have no epilogue HW (only ZERO_ACC / NO_TMA flags + a bank-swizzle build knob), so
they run pro/epilogue as **separate SIMT passes** (extra memory round-trips) — this
asymmetry is what the epilogue sweep is designed to expose.

## Experiments

- **Exp 1** — sweep size (≥5) × 8 apps × HW {0–6}; find sizes where best-HW flips
  per app. SIMT skipped at large sizes (O(MNK) scalar in SimX is too slow).
- **Exp 2** — at large sizes, sweep sim knobs (`DTCU_SWIZZLE`, `DTCU_MACS_PER_CYCLE`,
  `DTCU_SMEM_BANKS`, `DTCU_MAX_OUTSTANDING`; each is a `-D` rebuild) + hetero split
  ratio. Coarse first, fine if variance is high.

## Measured results — 2026-08-05

Three shapes on the stock `Makefile` config (1 cluster, 4 cores, `SOCKET_SIZE=1` → 4
sockets, 16 warps × 32 threads per core, 32 regs/thread, issue width 4; LMEM 64 KB per
core, L1 32 KB per socket, L2 1 MB per cluster, 64 B lines). App 1 (no epilogue), so
modes 3/4 do **not** pay the extra epilogue launch. Commit `fa8588e4c`.

Reproduce one cell with:

```
make run-simx OPTS="-M 512 -N 256 -K 128 -m 3"
```

`MAC = M·N·K`. A **unit** is what actually executes: a *core* for the in-core modes, all
4 of which run; an *engine* for the DTCU modes, of which exactly **1** is active — the
harness submits one descriptor from one thread, so mode 4 leaves three of its four socket
engines idle (`busy == busy_max` in its counters confirms this).

| mode | units | 128×64×32 · 0.5 wave ||| 256×128×64 · 2 waves ||| 512×256×128 · 8 waves |||
|---|---|---|---|---|---|---|---|---|---|---|
| | | cycles | MAC/cyc | /unit | cycles | MAC/cyc | /unit | cycles | MAC/cyc | /unit |
| 0 SIMT | 4 cores | 190,995 | 1.37 | 0.34 | *skipped* | — | — | *skipped* | — | — |
| 1 TCU | 4 cores | 14,626 | 17.92 | 4.48 | 96,992 | 21.62 | 5.41 | 385,339 | 43.54 | 10.88 |
| 2 TCU+DXA | 4 cores | 15,468 | 16.95 | 4.24 | 102,382 | 20.48 | 5.12 | **357,205** | **46.97** | **11.74** |
| 3 DTCU_cluster | 1 engine | 25,061 | 10.46 | **10.46** | 149,305 | 14.05 | **14.05** | 1,097,497 | 15.29 | **15.29** |
| 4 DTCU_socket | 1 engine | 25,553 | 10.26 | 10.26 | 159,573 | 13.14 | 13.14 | 1,154,569 | 14.53 | 14.53 |
| 5 TCU+DXA 3-stage | 4 cores | 17,351 | 15.11 | 3.78 | 101,651 | 20.63 | 5.16 | 362,150 | 46.33 | 11.58 |
| 6 TCU+DXA 2-stage | 4 cores | 23,170 | 11.31 | 2.83 | 105,968 | 19.79 | 4.95 | 378,565 | 44.32 | 11.08 |

SIMT completes only at the smallest shape; at 256×128×64 it had not finished after 25
minutes of simulation.

**Read the two throughput columns against each other — they say opposite things and both
are true.** On aggregate `MAC/cyc` the in-core path wins everywhere and pulls away
(17.9 → 43.5) because it turns a fuller machine into throughput. Per *unit* the engine
wins everywhere (10.5 → 15.3 against a core's 4.5 → 10.9). What moves with size is the
**margin**: the engine is 2.3× a core at the smallest shape, 2.6× in the middle, and only
1.4× once the cluster is full.

### Why the gap widens: the two paths are bound by different walls

The in-core path and the engine both get faster with size, but for unrelated reasons, and
only one of them keeps improving. This is the mechanism behind the widening ratio, and it
is *not* occupancy — that story only covers the first step.

**In-core (mode 1) is memory-latency bound.** `stall_lsu` is 93 % of cycles at
256×128×64 and 94 % at 512×256×128, so cycles are set by total load latency, and indeed
`load_lt` grows ×3.77 while cycles grow ×3.97. MACs grow ×8, so throughput doubles.
Per-core counters for the two steps:

| step | occupancy | loads | avg load latency | DRAM rd / L2 rd | MAC/cyc |
|---|---|---|---|---|---|
| 128×64×32 | 8/16 warp slots | 7,936 | 64.5 cyc | 63 % | 17.92 |
| 256×128×64 | 16/16 (2 waves) | 49,632 | **106.4 cyc** | 58 % | 21.62 |
| 512×256×128 | 16/16 (8 waves) | 326,112 | **61.0 cyc** | **26 %** | 43.54 |

* **small → mid (×1.21 only):** occupancy doubles, but average load latency gets *worse*
  (64.5 → 106.4) because twice the warps means more queueing. The two nearly cancel.
* **mid → large (×2.02):** occupancy is unchanged — both are full. The gain is two
  compounding effects. **(a) Fewer loads per MAC, ×1.22:** loads grow ×6.57 against MACs'
  ×8, because K = 64 → 128 doubles the K-tiles per block and amortizes the per-block C
  fragment load and D store over twice the MACs. **(b) Each load is cheaper, ×1.74:**
  average latency falls 106.4 → 61.0 because the grid grows from 8×16 to 16×32 blocks, so
  each A row-panel is shared by 16 blocks instead of 8 and each B panel by 32 instead of
  16 — L2 reuse rises and DRAM reads per L2 read fall from 58 % to 26 %. 1.22 × 1.74 =
  2.12, against 2.02 measured.

**The engine is compute bound, so none of that reaches it.** At 512×256×128 mode 3
reports `compute = 1,052,160` of 1,097,497 cycles (95.9 %), `tma_mem_wait` 6.9 %, and its
loader idle 62 % of the time waiting for a free operand buffer
(`tma_buf_starve = 685,129`). It has bandwidth to spare; the MAC array is the wall. So a
better L2 hit rate buys it almost nothing — 14.05 → 15.29 /unit, ×1.09, against the
in-core path's ×2.02. A third operand buffer would buy nothing either.

That is the honest mechanism: **growing the GEMM lowers the in-core path's wall and leaves
the engine's untouched.** Not that the engine got worse.

Two smaller readings:

1. **Do not draw the comparison from one shape.** The aggregate ratio against mode 1 is
   *not* monotonic — 1.71× → 1.54× → 2.85×. The smallest shape fills only 32 of the
   cluster's 64 warp slots, so it handicaps the in-core path and **flatters the engine**.
2. **Pipelining needs K depth to pay for itself.** At K=32 there is a single K-tile, so
   the 2-stage mode 6 is the *worst* in-core variant (2.83 /unit against single-buffer
   mode 2's 4.24) — barrier cost with nothing to prefetch. At K=128 the ordering reverses.

## Build / run

Out-of-tree build at `vortex/build/` (sources come from `VORTEX_HOME`). A new test
needs `build/tests/regression/cgo27_motivation/` with this Makefile copied in.

```
cd vortex/build
./ci/blackbox.sh --driver=simx --app=cgo27_motivation --perf=1 --args="-M 1024 -N 512 -K 64 -m <mode>"
```

### Choosing the GEMM shape

`-M m -N n -K k`: absolute dimensions. Anything omitted keeps its default
(M=128, N=64, K=32).

There used to be a `-s <mult>` flag meaning "mult × the DTCU native tile". It is gone.
With two DTCU engines whose tiles differ (cluster 64×32, socket 32×16) there is no
single native tile left for a multiplier to multiply, and the harness's whole premise is
that every mode runs the *same* GEMM. The size ladder now lives in `sweep_exp1.py` /
`sweep_exp2.py`, which expand each rung to explicit `-M/-N/-K`; their rungs reproduce
the old `-s N` expansion (M=64·N, N=32·N, K=16·N) so earlier sweep data stays comparable.

**Modes 3/4 take any shape; the in-core modes do not.** The DTCU rounds its tile
counts up and clamps the ragged trailing tile in hardware — operands past the matrix
are never fetched (the scratchpad is zero-filled, like the DXA copy engine's `cfill`)
and the D store leaves those bytes disabled, so nothing outside D is written. Only the
descriptor's `uint16_t` M/N/K field width binds (≤ 65535).

The in-core paths still need exact multiples, and the harness checks that up front
against the modes `-m` selected. At `NUM_THREADS=32`, fp16→fp32:

| dim | mode 0 | modes 1/2/5/6 | modes 3/4 | all modes |
|---|---|---|---|---|
| M | — | 16 (`tcu tileM`) | any | **16** |
| N | 32 (`NUM_THREADS`) | 16 (`tcu tileN`) | any | **32** |
| K | — | 32 (`tcu tileK`) | any | **32** |

So `-m 4 -M 100 -N 48 -K 20` is legal (all three axes ragged), while the same shape
on `-m 1` is rejected.

Modes 3/4 additionally cap each dimension at 65535 (`dtensor_desc_t` stores M/N/K as
`uint16_t`), and with an elementwise app (`-a 2`/`-a 3`) they also need `N % 32 == 0`
because the epilogue pass reuses the mode-0 grid.

Consequences worth knowing: `-K 16` can never run modes 1/2/5/6 (16 < `tcu tileK`=32),
and the default `-K 32` is **exactly one** K tile — so modes 5/6 have nothing to
prefetch and degenerate to mode 2. Use `-K 64` or higher to exercise the pipelining.

### Selecting the path and the app

`-m <mode>`: which HW path to run — `all` (default) or a single mode by index:
`0`=in-core SIMT, `1`=in-core TCU, `2`=in-core TCU+DXA, `3`=DTCU_cluster,
`4`=DTCU_socket, `5`=TCU+DXA pipelined (3-stage smem), `6`=TCU+DXA pipelined (2-stage
smem). Running a single mode skips the others entirely (verify, stats and the shape
check only apply to the modes that ran), so e.g. `-m 1` runs just the WMMA path without
the slow SIMT mode 0. `-a <app>`: epilogue/app id 1..8.

Examples:
```
--args="-M 1024 -N 512 -K 64"        # all 7 modes on a 1024x512x64 GEMM
--args="-M 1024 -N 512 -K 64 -m 5"   # only the 3-stage pipelined path
--args="-M 1024 -N 16  -K 64 -m 1"   # legal: N=16 is fine for the WMMA path alone
--args="-M 512 -N 256 -K 128 -m 4"   # only the socket-placed DTCU
```

All flags reject non-integers rather than silently truncating: `-K 1.7` errors out
instead of becoming `-K 1` (`atoi` used to stop at the `.`).

## Gotchas

- **SIMT has no fp16** (march `rv*imaf`, no Zfh) — mode 0 converts in software.
- **Modes 3 and 4 differ by PLACEMENT, not by flag.** They submit a byte-identical
  descriptor (bar `shape_n_size`, which each engine bounds differently) and differ only
  in which start instruction the kernel issues — `dtensor_cluster_start` vs
  `dtensor_socket_start`. `DTENSOR_FLAG_NO_TMA` still exists in the ISA but no harness
  mode uses it, and the old "blocking must never beat overlapped" tripwire is gone with
  it: neither placement is required to win, which is the question being measured.
  Neither engine has epilogue HW, so an elementwise app costs both a second launch.
- **Software pipelining must go through shared memory, not registers.** `mma_sync`
  pins its operands to fixed physical f-registers (C/D f0-f7, A f10-f17, B f24-f31),
  so 24 of 32 are reserved and a second prefetched operand set cannot fit — an
  earlier register-double-buffered mode 5 spilled and never finished at the 512x256x128
  rung.
  Modes 5/6 therefore differ in smem pipeline DEPTH instead.
- **Keep barrier objects short-lived in deep pipelines.** `vortex::barrier` holds
  bar_id_ + num_warps_, so three long-lived barriers = 6 live integer registers; that
  alone spilled the 3-stage kernel (15 sp accesses, 109,507 cycles vs 18,142 once the
  barriers were scoped to each use).
- Vendored `llvm-vortex` needs glibc 2.35 on this focal host → installed at
  `tools/glibc-2.35`, binaries patchelf'd.
