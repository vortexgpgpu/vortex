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

Grouped by **what executes**, so the numbering is deliberately sparse:

| # | Path | Notes |
|---|---|---|
| 0 | in-core SIMT | scalar MAC, software fp16→fp32 (no HW fp16 in SIMT) |
| 1 | in-core TCU (WMMA) | naive: load frag → mma per K |
| 2 | in-core TCU + DXA | naive: single-buffer, sync per K |
| 3 | in-core TCU — pipelined, **LSU-staged** | **2-stage** smem pipeline, block copies its own tiles. Control for mode 5 |
| 4 | in-core TCU — pipelined, **LSU-staged** | **3-stage** smem pipeline, block copies its own tiles. Control for mode 6 |
| 5 | in-core TCU + DXA — pipelined | **2-stage** smem pipeline: DXA runs 1 tile ahead (ref: sgemm2_dxa) |
| 6 | in-core TCU + DXA — pipelined | **3-stage** smem pipeline: DXA runs 2 tiles ahead |
| 7 | DTCU_socket | one engine per socket, D → that socket's L1, native tile 32×16 |
| 8 | DTCU_cluster | one engine per cluster, D → L2, native tile 64×32 |
| 9 | hetero: TCU + DTCU_socket | **not built** — reports `skipped=1` |
| 10 | hetero: TCU + DTCU_cluster | **not built** |
| 11 | hetero: TCU + both engines | **not built** |
| 12 | workgroup WGMMA + DXA | multi-warp CTA shares one staged tile; warp 0 produces; `wgmma` reads smem directly |
| 13 | workgroup WGMMA, SW copy | same geometry, the CTA copies its own tiles — the DXA control for 12 |

⚠️ **This numbering changed on 2026-08-05.** Previously 3=DTCU_cluster, 4=DTCU_socket,
5=3-stage, 6=2-stage. Both pairs moved *and* swapped order, and 3/4 were then reused for
the LSU-staged pipelines, so a mode number from an older log means something different.
The `[MOTI]` line carries `name=`, and the sweep scripts hard-error on a mismatch rather
than mislabelling a column.

**3/5 and 4/6 are matched pairs.** Identical tile geometry, stage count, barrier count and
lmem footprint — the only difference is who copies A and B into Local Memory: the block's
own threads (`kernel_modes/k_smem_stage.h`) or the DXA engine. That is what makes the copy
engine's contribution measurable at each pipeline depth; neither pair could show it alone.
Keeping the barrier *count* equal matters as much as the stage count, since barriers are a
per-CTA resource and a version needing fewer would win on occupancy instead.

One thing genuinely differs, and it is the finding rather than a flaw: a DXA fill is
**async**, so mode 5/6 issue the fill for tile k+1 and compute tile k while it runs. An LSU
fill is the block's own instruction stream, so nothing overlaps *inside* a CTA — the
prefetch distance only buys overlap across resident CTAs.

**Both engine modes tile by rows and build their own descriptors.** The kernel fills a
`dtensor_desc_t` from the addresses already in `kernel_arg_t`; the host only allocates the
array and zeroes it. Mode 7 makes one descriptor per **socket** (each socket has its own
engine, so they run concurrently, `engines=4 active=4`); mode 8 makes one per **core** into
the single cluster engine's queue. Only the *row* origin differs per slice: A and C/D are
row-major so a slice is a contiguous band, and B is shared untouched.

The submitter is picked by *where the block landed*, not by thread id — a socket engine is
only reachable from a core inside that socket, so the launch must be one block per core
for both modes:

```c
const uint32_t core = (uint32_t)vx_core_id();
if ((core % VX_CFG_SOCKET_SIZE) != 0) return;    // one submitter per socket
const uint32_t sock = core / VX_CFG_SOCKET_SIZE;
const uint64_t d = arg->desc_addr + (uint64_t)sock * sizeof(dtensor_desc_t);
moti_fill_desc(...); moti_publish_desc(d);       // fence + AMO, see below
while (0 == dtensor_socket_start(d)) ;
```

⚠️ **A fence alone does not publish the descriptor.** Core stores are write-through and
fire-and-forget — nothing acknowledges them — so `fence` has no completion to wait on and
the engine's descriptor read can pass the fill. `moti_publish_desc` follows the fence with
`dtensor_check()`'s AMO, which takes the cache's AmoProbe path and resolves at the LLC,
forcing the fill out. Without it, mode 8's four slices produced 6,144 errors: the three
descriptors the engine read as still-zero each retired instantly and set `done`.

⚠️ **Nothing about the split is passed in `kernel_arg_t`.** Socket size and count are
`VX_CFG_SOCKET_SIZE` / `VX_CFG_NUM_CORES`, which the kernel is already compiled with; the
element format ids are `constexpr`; the descriptor address is arithmetic on `desc_addr`.
This is not a style preference — see the Gotchas: growing that struct moves every mode's
cycle count.

## Files

One device program per mode, and the harness scaffolding shared once. A mode's cycle
count must not depend on which other modes exist in the tree — see the Gotchas for the
measurement that forced this.

| file | what it is |
|---|---|
| `main.cpp` | the driver: build the inputs, loop the requested modes, verify, report |
| `Makefile` | builds the x86 driver plus one `.vxbin` per mode. `make sizes` prints each program's `.text` against the 16 KB icache |
| `sweep_exp1.py` / `sweep_exp2.py` | the size×app and the sim-knob sweeps |
| `common.h` | `kernel_arg_t`, the host/device ABI — both sides include it. **Do not add fields** |
| `epilogue.h` | app id → epilogue, shared host/device |
| **`host/`** | |
| `host_modes.h` | the mode registry: ids, names, `ModeState` (Implemented / Reserved / Planned) |
| `run_modes.h` | `run_mode_0()` … `run_mode_8()`, one per mode, each returning its `ModeSpec`: kernel entry, ISA requirement, launch geometry, lmem stages, whether the host programs DXA |
| `host_run.h` | `run_case()` — the one piece of scaffolding every mode shares. Contains no mode branches |
| `host_args.h` | argument parsing and the shape checks that run before any device work |
| `host_types.h` | element conversions, the counter record, the ULP comparison |
| **`kernel_modes/`** | |
| `kernel_m<N>.cpp` | the GPU program for mode N. Its kernel body lives here and nothing else does |
| `wmma_common.h` | tile geometry `ctx`, fragment helpers, `h2f` — every mode |
| `k_smem_stage.h` | LSU operand staging — modes 3, 4 |
| `k_dtcu_desc.h` | descriptor construction — modes 7, 8 |
| `k_epilogue.h` | the standalone epilogue pass — modes 7, 8 |
| `kernel.cpp` | placeholder. `common.mk` needs `VX_SRCS` to name a file; this one defines no entries |
| **`docs/`** | the RFCs and `dtcu_figures.html`, the source of the published Artifact |

A header in `kernel_modes/` exists only when more than one mode needs it. Everything else
sits in the one `.cpp` that uses it.

Adding a mode: a `kernel_modes/kernel_m<N>.cpp`, a `run_mode_N()` in `host/run_modes.h`,
an id in `host/host_modes.h`. Then `-m all` and confirm no other mode moved.

`make sizes` prints each program's `.text` against the 16 KB icache.

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

- **Exp 1** — sweep size (≥5) × 8 apps × HW {0,1,2,5,6,7,8}; find sizes where best-HW flips
  per app. SIMT skipped at large sizes (O(MNK) scalar in SimX is too slow).
- **Exp 2** — at large sizes, sweep sim knobs (`DTCU_SWIZZLE`, `DTCU_MACS_PER_CYCLE`,
  `DTCU_SMEM_BANKS`, `DTCU_MAX_OUTSTANDING`; each is a `-D` rebuild) + hetero split
  ratio. Coarse first, fine if variance is high.

## Measured results — 2026-08-05

Three shapes on the stock `Makefile` config (1 cluster, 4 cores, `SOCKET_SIZE=1` → 4
sockets, 16 warps × 32 threads per core, 32 regs/thread, issue width 4; LMEM 64 KB per
core, L1 32 KB per socket, L2 1 MB per cluster, 64 B lines). App 1 (no epilogue), so
modes 7/8 do **not** pay the extra epilogue launch. Commit `fa8588e4c`.

Reproduce one cell with:

```
make run-simx OPTS="-M 512 -N 256 -K 128 -m 8"
```

`MAC = M·N·K`. A **unit** is what actually executes: a core for the in-core modes (4 of
them), a socket engine for mode 7 (4 of them, all active), the cluster engine for mode 8
(1). The harness prints the count as `engines=N active=N` on the `dtcu:` line.

| mode | units | 128×64×32 · 0.5 wave ||| 256×128×64 · 2 waves ||| 512×256×128 · 8 waves |||
|---|---|---|---|---|---|---|---|---|---|---|
| | | cycles | MAC/cyc | /unit | cycles | MAC/cyc | /unit | cycles | MAC/cyc | /unit |
| 0 SIMT | 4 cores | 190,995 | 1.37 | 0.34 | 1,145,460 | 1.83 | 0.46 | 9,581,708 | 1.75 | 0.44 |
| 1 TCU | 4 cores | 14,584 | 17.97 | 4.49 | 97,240 | 21.57 | 5.39 | 377,131 | 44.49 | 11.12 |
| 2 TCU+DXA | 4 cores | 15,647 | 16.75 | 4.19 | 100,281 | 20.91 | 5.23 | 354,814 | 47.28 | 11.82 |
| 3 TCU 2-stage, **LSU** | 4 cores | 21,011 | 12.48 | 3.12 | 145,829 | 14.38 | 3.60 | 631,840 | 26.55 | 6.64 |
| 4 TCU 3-stage, **LSU** | 4 cores | 42,951 | 6.10 | 1.53 | 198,111 | 10.59 | 2.65 | 817,439 | 20.52 | 5.13 |
| 5 TCU+DXA 2-stage | 4 cores | 21,350 | 12.28 | 3.07 | 100,683 | 20.83 | 5.21 | 380,441 | 44.10 | 11.02 |
| 6 TCU+DXA 3-stage | 4 cores | 15,086 | 17.38 | 4.34 | 104,245 | 20.12 | 5.03 | 359,502 | 46.67 | 11.67 |
| 7 DTCU_socket | 4 engines | **14,389** | **18.22** | 4.55 | **56,449** | **37.15** | 9.29 | **325,477** | **51.55** | 12.89 |
| 8 DTCU_cluster | 1 engine | 51,305 | 5.11 | 5.11 | 168,725 | 12.43 | 12.43 | 1,140,949 | 14.70 | **14.70** |
| 12 TCU wg + DXA | 4 cores | 66,908 | 3.92 | 0.98 | 258,543 | 8.11 | 2.03 | 994,415 | 16.87 | 4.22 |
| 13 TCU wg, SW copy | 4 cores | 55,867 | 4.69 | 1.17 | 219,648 | 9.55 | 2.39 | *pending* | — | — |

27 runs, 27 `PASSED!`, zero mismatches. Every mode now has its own device program, so
these numbers do not depend on which other modes exist in the tree — the previous table
was measured from a combined binary where they did.

**SIMT completes at every shape now** and is the motivation number the harness exists to
produce: 25.4× slower than the same GEMM on the in-core TCU at 512×256×128, 1.75 MAC/cyc
against 44.49. It was previously reported as "did not finish after 25 minutes"; on its own
190 KB device program it runs to completion.

**Four socket engines are the fastest path at every shape** — ahead of four cores running
WMMA at all three, by 9 % at the largest (51.55 against mode 2's 47.28 MAC/cyc).

**What the DXA engine is worth, isolated — and why modes 3/5 and 4/6 could not tell you.**
Those pairs hold tile geometry, stage count, barrier count and lmem fixed and vary only
who copies, which sounds like the right experiment. It is not enough, because all four
launch **one warp per block**: a warp stages a tile, issues one `mma_sync` against it and
throws it away. Sixteen warps resident on a core are sixteen unrelated CTAs each copying
its own private tile, so there is nothing for a copy to amortise over. Three things have
to hold first:

1. **Reuse** — the staged tile feeds more than one MMA.
2. **Warp specialisation** — a producer warp separate from the consumers. Modes 2/5/6
   already contain `is_dxa = (get_sub_group_id() == 0)`, but with one warp per block the
   producer and the consumer are the same warp and the async copy overlaps nothing.
3. **The consumer reads shared memory directly.** `load_matrix_sync` pulls the fragment
   into registers, so the LSU load *count* does not drop — 49,632 → 47,520, 4 %. DXA only
   makes each load cheaper (95.5 → 65.8 cycles) and pays for it on the SFU
   (`stall_sfu` 13,360 → 27,741).

**Modes 12/13 have all three** — an `ISSUE_WIDTH`-warp CTA sharing one staged tile, warp 0
as producer, `wgmma_sync` taking B as a shared-memory descriptor — and differ only in
whether the copy is a DXA descriptor or the CTA's own loads:

| | 128×64×32 | 256×128×64 | 512×256×128 |
|---|--:|--:|--:|
| 12 DXA, C pass removed | 14,093 | 71,583 | 335,171 |
| 13 SW copy, C pass removed | 16,634 | 91,418 | 494,464 |
| **what DXA is worth** | **1.18×** | **1.28×** | **1.48×** |

The engine pays, and by more as the shape grows. The 0.98–1.0× from the single-warp pairs
was a statement about those kernels, not about DXA.

⚠️ **Those two rows have the C pass removed, and that matters.** A wgmma context refuses
to load an accumulator from memory (`vx_tensor.h:789`) — the warpgroup accumulator is
distributed differently from a per-warp WMMA fragment even at the same tile shape, so
seeding it the WMMA way puts C in the wrong lanes (24,173 of 32,768 wrong, exactly one
warp in four correct). So `D = C + A·B` splits into: accumulate from zero, store, then
read D, read C, write D — four M·N accesses where the in-core modes fuse C and make one.
That is **58–79 %** of these modes, measured by compiling the pass out
(`-DMOTI_WG_NO_C`, whose D is wrong on purpose). It is worked around, not solved; the fix
is to combine C while the accumulator is still in registers and store once, as CUTLASS
does in its Hopper epilogue.

**Deepening the staged tile makes it worse.** Two K-steps per stage instead of one costs
1.40×/1.82× at 128×64×32 and 1.39×/1.83× at 256×128×64 — every shape, both modes. Local
Memory is a *per-CTA* resource, so doubling the stage halves the resident CTAs, and
halving the copies does not pay for halving the latency hiding. **Reuse has to grow along
N**, one staged tile feeding several output tiles, not along K.

**The epilogue costs 12/13 nothing.** app 2 and app 6 land within 0.3 % of app 1 (257,844
and 258,601 against 258,543): the C pass they are already forced to make absorbs it. Modes
7/8 pay a second launch for the same thing — mode 7 goes 14,389 → 73,973 at app 2.


**Splitting the CLUSTER GEMM four ways costs, it does not pay** — and separating the two
effects is the point of having done both. There is one cluster engine, so four descriptors
add no parallelism, only four times the `DESC_REQ`/`DESC_WAIT` round trip and four
pipeline fills; each slice is also *half* a tile, because the cluster tile is 64 rows and
a quarter of M is 32 at the smallest shape.

| mode 8 | 128×64×32 | 256×128×64 | 512×256×128 |
|---|--:|--:|--:|
| 1 descriptor (whole GEMM) | 25,061 | 149,305 | 1,097,497 |
| 4 descriptors (per core) | 51,461 | 168,613 | 1,140,573 |
| cost | **2.05×** | 1.13× | 1.04× |

The penalty is nearly all *fixed*, so it amortises away — 105 % at the smallest shape,
3.9 % at the largest. To go back to one descriptor, change the slice count in
`moti_dtcu_cluster` (`kernel_modes/kernel_m8.cpp`) from `VX_CFG_NUM_CORES` to 1; the host allocates a slot
per core either way.

So **mode 7's win is the engine count, not the tiling.** Applying the identical split to a
single engine makes it slower everywhere.

**The two throughput columns say opposite things, and both are true.** Per *unit* the
single cluster engine is still the most efficient thing in the table at the largest shape
(14.71, against a core's 10.88 and a socket engine's 12.93) because its 64×32 tile gets 4×
the operand reuse of the socket engine's 32×16 — even paying the four-descriptor penalty.
Aggregate, that one engine loses to four smaller ones by 3.5×. Tile efficiency and
throughput point in opposite directions; the placement decision is which one you are
buying.

**Descriptors are built by the kernel, and that is now in the measurement.** Both engine
modes fill their own `dtensor_desc_t` from the addresses already in `kernel_arg_t` (see
the mode 7/8 note above). Against host-staged descriptors that costs mode 7 a fixed
~500 cycles — +3.6 % at the smallest shape, +0.5 % at the largest. It is a correction, not
a regression: a host-staged descriptor costs zero *measured* cycles, which hides exactly
the per-GEMM control cost these modes exist to quantify.

⚠ **Mode 5 is bimodal — do not quote 23,170 as a measurement.** Its per-unit stalls match
unpipelined mode 2 to within a few percent (`lsu` 8,485 vs 8,427, `sfu` 3,041 vs 3,389)
yet it spends 7,700 more cycles, all of it barrier idle charged to no functional unit.
Growing an unrelated struct by 16 bytes moved it to 15,548. Two buffers leave the DXA
transfer and the stage's compute close enough in length that the instruction schedule
decides which wins; three buffers give enough slack that it stops mattering.

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

**The engine is compute bound, so none of that reaches it.** At 512×256×128 mode 8
reports `compute = 1,052,160` of 1,097,497 cycles (95.9 %), `tma_mem_wait` 6.9 %, and its
loader idle 62 % of the time waiting for a free operand buffer
(`tma_buf_starve = 685,129`). It has bandwidth to spare; the MAC array is the wall. So a
better L2 hit rate buys it almost nothing — 14.05 → 15.29 /unit, ×1.09, against the
in-core path's ×2.02. A third operand buffer would buy nothing either.

That is the honest mechanism: **growing the GEMM lowers the in-core path's wall and leaves
the engine's untouched.** Not that the engine got worse.

### Widening the engine: `DTCU_MACS_PER_CYCLE` alone does nothing

The engine's compute model is `max(mac, operand read, accumulator RMW) + latency`, so the
MAC array is only one of three terms and at the default parameters it is **exactly tied**
with the accumulator. For the cluster tile (64×32, K-tile 16): `tile_macs = 32,768`, so
`mac = 32768 / DTCU_MACS_PER_CYCLE = 2048` cycles at the default 16, while
`accum = 2·64·32 / DTCU_ACC_BANKS = 2048` cycles at the default 2 banks. Raising one
without the other changes nothing at all — measured at 512×256×128, mode 8:

| build | cycles | `compute` |
|---|---|---|
| default (`MACS=16`, `ACC_BANKS=2`) | 1,097,497 | 1,052,160 |
| `-DDTCU_MACS_PER_CYCLE=64` | **1,097,497** | **1,052,160** — bit-identical |
| `+ -DDTCU_ACC_BANKS=8` | 406,369 | 265,728 |
| `+ -DDTCU_SMEM_BANKS=8` | **377,077** | 265,728 |

A 4× wider MAC array on its own is worth **zero**. With the accumulator widened to match
it is 2.7×, and adding operand-SRAM banks (the `read` term, which becomes co-dominant once
the other two drop to 512) takes it to 2.9×. `dtcu_params.h` says as much: the comment on
`DTCU_MACS_PER_CYCLE 16` explains it was chosen to match the in-core TCU **at
`NUM_THREADS=4`**, and this harness runs `NUM_THREADS=32`.

Sweep these together, never alone — `sweep_exp2.py`'s `KNOBS` varies one factor at a time
and will report `DTCU_MACS_PER_CYCLE` as having zero sensitivity, which is true and
misleading.

#### The two placements respond to width completely differently

Scaling all three terms together at 512×256×128 (`MACS_PER_CYCLE` / `ACC_BANKS` /
`SMEM_BANKS` = 16/2/2, 32/4/4, 64/8/8). Reproduce with the CONFIGS **environment**
variable, never `make CONFIGS=` — see Gotchas:

```
CONFIGS="-DDTCU_MACS_PER_CYCLE=64 -DDTCU_ACC_BANKS=8 -DDTCU_SMEM_BANKS=8" \
  make run-simx OPTS="-m 7 -M 512 -N 256 -K 128"
```

| width | 7 socket ×4 | MAC/cyc | speedup | 8 cluster | MAC/cyc | speedup |
|---|--:|--:|--:|--:|--:|--:|
| 1× | 324,469 | 51.71 | — | 1,140,573 | 14.71 | — |
| 2× | 243,869 | 68.80 | 1.33× | 663,865 | 25.27 | 1.72× |
| 4× | 223,977 | **74.91** | 1.45× | 411,997 | 40.72 | **2.77×** |

**The cluster engine is compute-bound and the socket engines are not.** Widening pays the
cluster engine 2.77× and the socket engines only 1.45×, and by 4× the socket variant has
clearly saturated (2× → 4× buys just 1.09×). Its 32×16 tile is a quarter the area of the
cluster's 64×32, so per tile it spends proportionally far more of its time on descriptor
fetch, operand fill and store drain — none of which a wider MAC array touches. Widen the
cluster engine; replicate the socket engine.

Even so the socket variant stays ahead at every width — 1.84× at 4× — and at 74.91 MAC/cyc
it is **1.60× the whole 4-core cluster** (mode 2's 46.97). The DTCU is not intrinsically
outmatched by the cores; the default parameterisation and a single engine were.

**Replicating the engine beats widening it, at equal silicon.** Four unmodified socket
engines reach 51.71 MAC/cyc; one cluster engine with `MACS_PER_CYCLE`, `ACC_BANKS` and
`SMEM_BANKS` all raised 4× reaches 40.72 — and those are comparable budgets, 4 MAC arrays
and 4 accumulators either way. Replication also wins because the two respond to width
completely differently, which is its own section below. The default engine is not
undersized; it is under-replicated.

Three smaller readings:

1. **Do not draw the comparison from one shape.** Mode 8's aggregate ratio against mode 1
   is *not* monotonic — 1.71× → 1.54× → 2.85× slower. The smallest shape fills only 32 of
   the cluster's 64 warp slots, so it handicaps the in-core path and **flatters the
   engine**.
2. **Pipelining needs K depth to pay for itself.** At K=32 there is a single K-tile, so
   the 2-stage mode 5 is the *worst* in-core variant (2.83 /unit against single-buffer
   mode 2's 4.24) — barrier cost with nothing to prefetch. At K=128 the ordering reverses.
   See the ⚠ above before reading much into mode 5's absolute number.
3. **A socket engine's per-unit throughput climbs faster than the cluster engine's**
   (4.65 → 12.93, ×2.8, against 5.09 → 14.71, ×2.9) because each of the four gets a
   quarter of the rows: at the smallest shape a slice is 32 rows — one tile row — so
   descriptor and pipeline-fill overhead dominates. The two converge as the slices grow.

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

- **`kernel_arg_t`'s SIZE is part of the experiment's configuration.** The struct's own
  comment says to append new fields at the end; that is necessary but not sufficient.
  Appending four fields (64 → 80 B) moved mode 2 **+15.8 %** and mode 5 **−32.9 %** —
  every kernel reads the struct, so growing it reshuffles codegen in paths that were not
  touched. Reverting restored all five modes to the digit. **Anything a kernel can derive
  from a build constant (`VX_CFG_SOCKET_SIZE`, `VX_CFG_NUM_CORES`) or from an address it
  already has must not be added.** If a new field is genuinely unavoidable, re-measure
  every mode, not just the one being changed.
- **`-m 0` was unselectable until 2026-08-05.** `parse_u32` rejected 0 as "not a positive
  integer", so the SIMT baseline could only be reached through `-m all` — and the error
  named the wrong problem. `-m` now parses with `allow_zero`; the matrix dimensions still
  do not.
- **`make run-simx`, not `./cgo27_motivation`.** The simulator is rebuilt with the test's
  `CONFIGS` by the `run-simx` target. Running the binary directly reuses whatever
  `libsimx.so` was last built, which silently reports different ISA extensions — DXA modes
  come back `skipped=1` and the TCU mode segfaults.
- **SIMT has no fp16** (march `rv*imaf`, no Zfh) — mode 0 converts in software.
- **Modes 7 and 8 differ by PLACEMENT, not by flag.** They submit a byte-identical
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
