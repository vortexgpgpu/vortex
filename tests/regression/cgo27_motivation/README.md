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
| 3 | workgroup WGMMA + DXA | multi-warp CTA shares one staged tile; warp 0 produces; `wgmma` reads smem directly |
| 4 | workgroup WGMMA, SW copy | same geometry, the CTA copies its own tiles — the DXA control for 3 |
| 5, 6 | *reserved holes* | retired; see below |
| 7 | DTCU_socket | one engine per socket, D → that socket's L1, native tile 32×16 |
| 8 | DTCU_cluster | one engine per cluster, D → L2, native tile 64×32 |
| 9 | hetero: TCU + DTCU_socket | **not built** — reports `skipped=1` |
| 10 | hetero: TCU + DTCU_cluster | **not built** |
| 11 | hetero: TCU + both engines | **not built** |

⚠️ **This numbering changed again on 2026-08-07, and a log from before that means
something different.** 3 and 4 now hold the workgroup pair that was 12 and 13. The four
single-warp staging modes that held 3–6 are **retired**: a block there was one warp, so it
staged a tile, issued one `mma_sync` against it and threw it away, with nothing to
amortise the copy over. All four landed within 7 % of mode 1, which stages nothing at all,
and none completed at 512×256×128 — they measured the absence of a geometry rather than
the presence of an engine. 5 and 6 are now reserved holes. (Before *that*, 3=DTCU_cluster
and 4=DTCU_socket, which moved to 7/8.)

Their last numbers, for the record: 3 (2-stage LSU) 104,196 / 382,111 / —, 4 (3-stage LSU)
231,610 / — / —, 5 (2-stage DXA) 90,132 / — / —, 6 (3-stage DXA) 132,452 / — / —.
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

| mode | units | 128×64×32 ||| 256×128×64 ||| 512×256×128 |||
|---|---|---|---|---|---|---|---|---|---|---|
| | | cycles | MAC/cyc | /unit | cycles | MAC/cyc | /unit | cycles | MAC/cyc | /unit |
| 0 SIMT | 4 cores | 142,952 | 1.83 | 0.46 | — | — | — | — | — | — |
| 1 TCU | 4 cores | 23,513 | 11.15 | 2.79 | 96,244 | 21.79 | 5.45 | 386,994 | 43.35 | 10.84 |
| 2 TCU+DXA | 4 cores | 23,939 | 10.95 | 2.74 | 101,977 | 20.56 | 5.14 | 446,129 | 37.61 | 9.40 |
| 3 TCU wg + DXA | 4 cores | 25,386 | 10.33 | 2.58 | 103,400 § | 20.28 | 5.07 | 454,165 | 36.94 | 9.23 |
| 4 TCU wg, SW copy | 4 cores | 32,583 | 8.04 | 2.01 | 152,677 | 13.73 | 3.43 | ‡ | ‡ | ‡ |
| 7 DTCU_socket | 4 engines | 11,912 | 22.01 | 5.50 | 49,064 | 42.74 | 10.69 | 303,884 | 55.21 | 13.80 |
| 8 DTCU_cluster | 1 engine | 49,472 | 5.30 | 5.30 | 160,448 | 13.07 | 13.07 | 1,112,370 | 15.08 | 15.08 |

Post-merge with upstream (`00ea949a1`). **Every number in this table replaced a
pre-merge one** — upstream rebuilt the memory path underneath us and the ordering moved
with it; see below.

**— means the run does not complete**, not that it was skipped. Modes 3/4/5/6 and 3/4
hit a wall between 256×128×32 and 384×192×32: mode 4 scales linearly to that point
(231,610 → 345,311 → 581,372 cycles at 7/12/21 s of simulation) and then 384×192×32 does
not finish in an hour. K depth is not the cause — at 128×64 the same mode takes 231,610 /
225,279 / 302,906 for K = 32/64/128. Mode 0 is skipped above the smallest shape by policy
(see Gotchas), which is a different thing from these.

**Mode 7 wins at every shape, and by more than it used to** — 11,912 / 49,064 / 303,884
cycles, 1.97× / 1.96× / 1.27× against plain in-core WMMA. Before the merge the margin at
the largest shape was 1.16×.

**The merge moved the floor, not the engine.** Upstream replaced the memory path: L2 went
from a 64 B line to a sectored 128 B one, `LSUQ_IN_SIZE`/`LSUQ_OUT_SIZE` were replaced by
a single `LSU_PENDING_SIZE` queue, and about a thousand lines landed in `sim/simx/mem/`.
The measured consequence, at 128×64×32 mode 1:

| | pre-merge | post-merge |
|---|--:|--:|
| loads | 7,936 | 7,936 |
| **average load latency** | 64.5 cyc | **351.4 cyc** |
| `stall_lsu` | 8,427 | 12,920 |
| cycles | 14,584 | 23,513 |

Not one extra load — each one costs 5.4× more. Mode 1 spends 55 % of its cycles in
`stall_lsu`, so that is the whole story for it.

**Mode 7 barely felt it**: its core-side counters are `loads=116`, every stall category
zero. The GEMM traffic goes out of the engine's own TMA port and the core only submits a
descriptor and polls. A coarser memory hierarchy costs a core that issues every load and
costs an engine nothing — the engine turns the wider line into bandwidth instead.

That is the sharpest form of §1.1's claim the harness has produced so far, and it arrived
by accident: **the deeper the memory hierarchy, the better the descriptor engine looks,
because the core is not the one waiting.**

One ordering flipped: mode 2 (TCU+DXA) used to beat mode 1 at the largest shape and now
loses to it, 446,129 against 386,994. DXA stages through Local Memory but the fragments
still reach the TCU as LSU loads, so it pays the new latency twice over — once on the DXA
fill and again on the smem read.

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

**Modes 3/4 have all three** — an `ISSUE_WIDTH`-warp CTA sharing one staged tile, warp 0
as producer, `wgmma_sync` taking B as a shared-memory descriptor — and differ only in
whether the copy is a DXA descriptor or the CTA's own loads:

| | 128×64×32 | 256×128×64 | 512×256×128 |
|---|--:|--:|--:|
| 3 DXA, C pass removed | 14,093 | 71,583 | 335,171 |
| 4 SW copy, C pass removed | 16,634 | 91,418 | 494,464 |
| **what DXA is worth** | **1.18×** | **1.28×** | **1.48×** |

The engine pays, and by more as the shape grows. The 0.98–1.0× from the single-warp pairs
was a statement about those kernels, not about DXA.

⚠️ **Those two rows predate the epilogue fix and are kept for the ratio only.** They
were measured when `D = C + A·B` still split into four M·N accesses; what follows replaces
them.

### The DTCU's compute timing is anchored to a TCU that no longer exists

Checked because the engine looked too fast. It is not the compute array that is
mis-modelled, and the direction of the error is the opposite of the suspicion — but the
anchor really is stale, in two separate ways.

**The merge moved the TCU's MMA latency and left the DTCU behind.** Upstream replaced a
hardcoded `delay = 4` in `tcu_unit.cpp` with `delay = kMmaLatency = 1 + kFedpLatency`,
derived from the configured PE type: 5 for `TFR` (this build), but 16 for `BHF`, ~35 for
`FPNEW` and 54 for `DSP`. `DTCU_COMPUTE_LATENCY` is still a hardcoded **6**, written
2026-08-04/05 against the pre-merge TCU, and `git diff 00ea949a1 HEAD -- sim/simx/dtcu/`
is empty. So the two units' pipeline depths now track each other only by coincidence, and
they stop agreeing entirely under any `VX_CFG_TCU_TYPE` but the default.

**`DTCU_MACS_PER_CYCLE` is anchored to a configuration this harness does not run.** Its
comment says 16 "== one in-core TCU's raw throughput (NT=4), so the DTCU's modeled
advantage comes only from removing SIMT pipeline overhead and NOT from also assuming a
wider array (no double counting)." This harness runs **NT=32**. At NT=32,
`wmma_config_t` gives `tcM=8, tcN=4, tcK=4` and `m_steps × n_steps × k_steps = 2 × 4 × 4
= 32` uops per 16×16×16 WMMA — so **128 MACs per uop**, one uop per cycle per block,
`NUM_TCU_BLOCKS=4` blocks: **512 MACs/cycle per core**. The parity the parameter claims is
off by 8× per warp and 32× per core, and it is off in the direction that makes the engine
look *worse*, not better.

So the two units are being compared at wildly different modelled peaks:

| | modelled peak | achieved at 512×256×128 | utilisation |
|---|--:|--:|--:|
| in-core TCU, 4 cores | 2,048 MAC/cyc | 43.4 | **2.1 %** |
| DTCU_socket, 4 engines | 64 MAC/cyc | 55.2 | **86 %** |

**That is the real finding, and it cuts both ways.** The engine is not winning because it
was given a fat array — it was given one 32× *thinner* than the core's. It wins because
nothing ever stalls it, which is the paper's claim, and the measured counters agree
(`loads=116`, every stall category zero). But an 86 % vs 2.1 % utilisation gap is a very
large thing to rest on one hand-set constant, and `DTCU_MACS_PER_CYCLE=16` mostly does not
even bind: at the socket tile (32×16) the accumulator term
`2·32·16/DTCU_ACC_BANKS + 1 = 513` and the MAC term `32·16·16/16 = 512` are within one
cycle of each other, so `max()` picks them almost interchangeably. The engine is modelled
as accumulator-bandwidth-bound.

**Not changed here, because it is a machine-configuration decision.** The two edits that
would follow are (a) derive `DTCU_COMPUTE_LATENCY` from the same `kFedpLatency` the TCU
now uses instead of hardcoding 6, and (b) re-anchor `DTCU_MACS_PER_CYCLE` to the in-core
TCU's throughput **at the configured NT** rather than at NT=4. Both move every DTCU number
in this file.

### The epilogue was four M·N accesses, and that is what stopped 3/4 at 512×256×128

A wgmma context refuses to load an accumulator from memory (`vx_tensor.h:789`) — the
warpgroup accumulator is distributed differently from a per-warp WMMA fragment even at the
same tile shape, so seeding it the WMMA way puts C in the wrong lanes (24,173 of 32,768
wrong, exactly one warp in four correct). The original way around that was: accumulate
from zero, **store D**, then **read D, read C, write D** — four M·N global accesses where
the in-core modes fuse C and make two.

At 512×256×128 that is C + D = 1,024 KB of read-write traffic against a **1,024 KB L2**
with A (128 KB) and B (64 KB) also live. The proof that this was the binding constraint
was already sitting in the table above: the same kernels with the pass compiled out ran in
335,171 / 494,464 cycles, while the ones with it **did not finish in four hours**. Live set
with the pass removed is A+B+D = 704 KB, which fits; with it, 1,216 KB, which does not.

**The fix folds C in while the accumulator is still in registers and writes D once** — two
M·N accesses, the same as modes 1/2/5/6, with no second pass, no scratch and no barrier.
The layout that defeated the preload is not a secret: `store_matrix_sync`
(`vx_tensor.h:944`) computes it in the open, and the epilogue is that computation with a
read of C and an add spliced in. Reading C at the *accumulator's* addresses is what makes
it correct where reading it at *WMMA* addresses was not.

| | 128×64×32 | 256×128×64 | 512×256×128 |
|---|--:|--:|--:|
| 3, store D + read back (4 accesses) | 84,045 | 314,428 | **did not finish** |
| 3, store to LMEM + coop pass (2) | 95,325 | 326,171 | did not finish |
| **3, fused in registers (2)** | **25,386** | **103,400** § | **454,165** |
| 4, store D + read back (4 accesses) | 63,339 | 261,673 | **did not finish** |
| 4, store to LMEM + coop pass (2) | 77,386 | 281,189 | did not finish |
| **4, fused in registers (2)** | **32,583** | **152,677** | ‡ |
| speed-up over the 4-access original | 3.31× / 1.94× | 3.04× / 1.71× | ∞ / — |

Every fused-epilogue run above verified `PASSED!`, errors = 0.

**Mode 12 at 128×64×32 goes 84,045 → 25,386, and its DRAM traffic lands on mode 1's.**
`mem_reads` 2,264 against mode 1's 2,255, `mem_writes` 1,536 against 1,536 — the epilogue
now makes exactly the passes the in-core modes make. `stall_lsu` drops 11,722 → 2,518 and
average load latency 254.8 → 119.4 cycles.

**The middle option was built, and it is worth keeping as a negative result.** Storing the
accumulator to Local Memory and adding C in a cooperative pass is *also* two M·N global
accesses, and it is **slower than the original** at both shapes that fit in L2. It pays an
LMEM round trip and a CTA barrier per output element, and `store_matrix_sync`'s
lane→address map (row `lane/tcN`, column `lane%tcN`, rows 64 B apart) touches 8 of the 32
LMEM banks four times each. Two M·N accesses is necessary, not sufficient — *where* the
accumulator lands decides the rest.

**§ and ‡ — one unexplained scheduling effect, and one point still down.**

`-a 1` (identity) and `-a 6` (also identity — `epi_apply` returns `v` for both, after the
same two failed compares) execute **byte-identical instructions on byte-identical data**:
the operands come from fixed formulas that do not read the app, and the only thing the app
changes on the device is one integer in `kernel_arg_t`. Yet at 256 × 128 × 64, mode 12
finishes at `-a 6` in **103,400 cycles** and does not finish at `-a 1` — six attempts, four
of them launched simultaneously on an idle machine, none completing. `-a 2` (105,365) and
`-a 3` (123,353) also finish. **§ marks a number taken from `-a 6` for that reason.** It is
the identity epilogue, so it is the right value for that cell; the `-a 1` behaviour is a
live bug somewhere below the kernel and is not understood. It is **not** the epilogue
function, **not** the operand values and **not** the memory access pattern — all three are
identical between the two runs.

**‡ Mode 13 at 512 × 256 × 128 does not complete at any app tested** (`-a 1` and `-a 2`,
both past 20 minutes). That one is not app-sensitive, so it is probably a different problem
from the § one. It is also **the only cell with no correctness check at all**.

An earlier draft of this section blamed the accumulator's lane→address map for scattering
one warp instruction across eight cache lines. That scattering is real — the map covers
8 rows × 4 columns where a cooperative pass covers 2 rows fully — but it **cannot** explain
§, because `-a 6` makes exactly the same accesses and completes. The claim is withdrawn.

| `-DMOTI_WG_NO_C` control (D wrong on purpose) | 128×64×32 | 256×128×64 | 512×256×128 |
|---|--:|--:|--:|
| 3 | 22,979 | 98,637 | 423,127 |
| 4 | 25,795 | 104,133 | 497,215 |

That switch is narrower than it used to be: it once removed a whole second pass over D and
now removes only the C read, so the old NO_C numbers are not comparable to these.

**Correctness coverage of the fused epilogue** — `PASSED!`, errors = 0 in every cell but
one:

| | 128×64×32 | 256×128×64 | 512×256×128 |
|---|---|---|---|
| 3 | ✅ a1, a2, a6 | ✅ a2, a3, a6 | ✅ a1 |
| 4 | ✅ a1, a2, a6 | ✅ a1, a2 | **✗ never verified** |

**Deepening the staged tile makes it worse.** Two K-steps per stage instead of one costs
1.40×/1.82× at 128×64×32 and 1.39×/1.83× at 256×128×64 — every shape, both modes. Local
Memory is a *per-CTA* resource — but **it is not occupancy, though this file said so until
it was checked**. At S=1 a CTA's stage is 2,560 B and at S=2 it is 5,120 B, so
`usable_slots()` goes 16 → 12 in a 64 KB Local Memory, both far above the ceiling
`NUM_WARPS` already imposes: 16 warps / 4 warps-per-CTA = **4 CTAs**. Local Memory does not
cost a single resident CTA until S = 8 (20,480 B, 3 slots). Whatever S=2 and S=4 cost, it
is not resident CTAs, and it is still unexplained. **Reuse has to grow along
N**, one staged tile feeding several output tiles, not along K.

**The epilogue costs 3/4 nothing.** app 2 and app 6 land within 0.3 % of app 1 (257,844
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
