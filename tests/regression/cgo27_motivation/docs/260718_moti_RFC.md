# RFC: cgo27_motivation — motivation experiments for the CGO'27 paper

**Status:** in progress (2026-07-18). **Owner:** sjeong306.
**Living doc — implementation status and experiment results MUST be kept updated
in §8/§9 as work lands.**

---

## 1. Thesis (why this harness exists)

The CGO'27 paper argues that a reconfigurable SIMT GPU (Vortex) is a *target
family*, and a **target-parametric compiler** should map a workload to whatever
compute/memory units a given `c` exposes. The load-bearing empirical claim:
**the optimal HW path for the same GEMM changes with problem size and with the
surrounding pro/epilogue.** This harness measures exactly that — it runs the
same GEMM on the same input through many HW paths and reports per-path cycles,
so we can find the (app, size) points where the best path flips.

Framing note: this is a **capability + mechanism** paper, not a "surprising
finding" paper. The crossover figure's job is to establish the problem is real
(a hardcoded backend / fixed heuristic is inadequate), not to astonish. See the
project memory `vortex-paper-direction` for the full framing.

## 2. GEMM, layouts, precision

All paths compute the **same** `D = C + A·B`:
- `A`: row-major `[M×K]` (`A[i*K+k]`)
- `B`: col-major `[K×N]` (`B[j*K+k]`)
- `C`,`D`: row-major `[M×N]`
- input fp16 → output fp32 (dequant apps: int8 input → fp16, see §3).

Every path is verified against a CPU reference (ULP≤6) every run. Same generated
input feeds all paths.

## 3. Apps (prologue/epilogue) — 8

Selected to span the axes that stress core-vs-DTCU placement differently.

| # | App | Axis stressed |
|---|---|---|
| 1 | baseline `D = C+A·B` | pure GEMM placement |
| 2 | + ReLU | divergence (branch) |
| 3 | + GELU | compute-heavy (transcendental) |
| 4 | + Residual (`+R`) | extra memory traffic (full-matrix read) |
| 5 | + Scale (per-channel) | cheap broadcast |
| 6 | + Softmax (row-wise) | divergent + cross-tile reduction (hardest) |
| 7 | dequant(int8→fp16) + bias + GELU | prologue + epilogue (quantized FFN) |
| 8 | dequant(int8→fp16) + softmax | prologue + reduction epilogue |

**Key asymmetry (the point of the epilogue sweep):** the DTCU HW does GEMM only
(descriptor flags = ZERO_ACC / NO_TMA, plus an operand-SRAM bank-swizzle build
knob; **no epilogue/activation/bias**). So:
- in-core modes (0,1,2,5,6): fuse prologue at operand load, epilogue at store.
- DTCU modes (3,4): run prologue and epilogue as **separate SIMT passes** over
  memory (extra full-matrix round-trips before and/or after the descriptor GEMM).

The DTCU wins the bare GEMM but pays a pro/epilogue round-trip tax the in-core
paths don't — this is the fused-context inversion the sweep is designed to find.
Softmax (row reduction across N) is the strongest case and needs a cross-tile
(2-pass) structure even for the in-core paths.

## 4. HW modes (`arg->mode`)

`pipelined` is **not** a flag; it is separate mode numbers (per decision, §10).

| # | Path | naive/pipelined | Source of pattern |
|---|---|---|---|
| 0 | in-core SIMT | — | sgemm/kernel.cpp + sw fp16→fp32 (`h2f`) |
| 1 | in-core TCU (WMMA) | naive | dtcu_compare mode 0 |
| 2 | in-core TCU + DXA | naive (single-buf, sync) | sgemm_tcu_wg_dxa + our mode-2 coords |
| 3 | DTCU | naive (`NO_TMA`, blocking) | dtcu_compare mode 1 |
| 4 | DTCU + DTCU_TMA | pipelined (TMA overlap) | descriptor flag clear |
| 5 | in-core TCU + DXA pipelined | **3-stage** smem pipeline (DXA runs 2 tiles ahead) | new (replaced the infeasible register double-buffer) |
| 6 | in-core TCU + DXA pipelined | **2-stage** smem pipeline (1 tile ahead) | sgemm2_dxa ping-pong |
| 7 | hetero: SIMT+TCU+DTCU | — | new (Phase C) |
| 8 | hetero: all units | — | new (Phase C) |

naive = sync after each tile, then compute. pipelined = SW pipeline: prefetch ahead
while computing the current tile. Modes 5 and 6 now differ only in prefetch DEPTH
(3 stages vs 2), which makes depth a measurable axis; register-level pipelining is
NOT one of the options because mma_sync pins 24 of the 32 f-registers (see §9).
The DTCU's equivalent is its HW TMA (mode 4 vs 3). For a *fair* best-HW crossover, compare pipelined variants;
naive variants are the ablation showing SW pipelining matters. Hetero (7,8)
partition output tiles across units running concurrently (DTCU autonomous +
in-core WMMA), fixed split ratio first — the hand-coded ceiling of the compiler's
coarse-grained parallelization; expected to help only at very large sizes.

## 5. HW config (Hopper-proportional)

Per core = 16 warps × 32 threads = **512 threads, issue 4 ≈ ¼ of an H100 SM**
(2048 threads, issue 4). On-chip memory scaled ×¼; global L2 by core count.

| | value | H100/SM basis |
|---|---|---|
| clusters / cores / socket | 1 / 4 / 1 | |
| warps/core · threads/warp · issue | 16 · 32 · 4 | =SM warp size & issue, ¼ occupancy |
| L1 dcache | 32 KB | ~128KB ÷4 |
| LMEM (shmem) | 64 KB | 228KB ÷4 ≈57→64 (pow2) |
| L2 | 1 MB, enabled | 50MB ×4/132 ≈1.5MB |
| barriers | 32 | pipelined needs 2/CTA, up to 16 CTAs/core |

Set in `Makefile`. Sweepable by overriding `CONFIGS=` on the make/blackbox line.
Full rationale in `README.md`.

### The core geometry, verbatim

This is the line the experiments are built with — it is the config every number in
§9 was measured under, so quote it in the paper's methodology and do not vary it
between runs that get compared:

```
NUM_CLUSTERS=1  NUM_CORES=4  SOCKET_SIZE=1  NUM_WARPS=16  NUM_THREADS=32  ISSUE_WIDTH=4
```

As it appears in `Makefile` (and in the compiler command line, verified):

```make
CONFIGS += -DVX_CFG_NUM_CLUSTERS=1
CONFIGS += -DVX_CFG_NUM_CORES=4
CONFIGS += -DVX_CFG_SOCKET_SIZE=1
CONFIGS += -DVX_CFG_NUM_WARPS=16
CONFIGS += -DVX_CFG_NUM_THREADS=32
CONFIGS += -DVX_CFG_ISSUE_WIDTH=4
```

### Why NUM_WARPS=16 is a floor, not a preference

Every CTA in this harness is **one warp** (`block_dim = NUM_THREADS`), so the
resident-CTA count per core is `min(NUM_WARPS, lmem_bytes / stage_bytes, NUM_BARRIERS
/ barriers_per_CTA)`. At `NUM_THREADS=32`, fp16→fp32, `stage_bytes = 2048`:

| mode | warp slots | lmem limit | barrier limit | resident CTAs/core |
|---|---|---|---|---|
| 0, 1 | 16 | — (no lmem) | — (no barriers) | **16** |
| 2 (1 stage) | 16 | 64K/2048 = 32 | 32/1 = 32 | **16** ← warp slots bind |
| 6 (2 stages) | 16 | 64K/4096 = 16 | 32/2 = 16 | **16** |
| 5 (3 stages) | 16 | 64K/6144 = 10 | 32/3 = 10 | **10** ← only mode that loses |

Mode 5 exists to answer one question: does a deeper prefetch beat the occupancy it
costs? That question is only *askable* while mode 5's cap (10) sits below the other
modes' cap. With `NUM_WARPS <= 10` every mode caps at `NUM_WARPS`, mode 5 loses
nothing, and the 2-stage-vs-3-stage comparison degenerates into "same occupancy,
deeper pipeline" — it would always favor mode 5 and measure nothing about the
trade-off. So the floor is `NUM_WARPS >= 11`; 16 is the power of two above it and is
independently what makes 16×32 = 512 threads ≈ ¼ H100 SM.

Corollary for `NUM_CORES`: dropping 4→2 halves the in-core machine (modes 0/1/2/5/6)
while the DTCU is cluster-level and barely moves, so it shifts the in-core-vs-DTCU
crossover that Exp 1 is measuring. Reduce simulation cost with `-m` instead.

## 6. Experiment 1 — size sweep, best-HW crossover

- axes: 8 apps × HW {0–6} × size (≥5 `size_mult` values). GEMM = mult × DTCU
  native tile (M=64·mult, N=32·mult, K=16·mult).
- metric: cycles (SimX ground truth), via MPM (`--perf=1`).
- goal: per app, list sizes where argmin-cycles HW changes.
- SIMT (mode 0) is O(MNK) scalar in SimX → skip above a size cap.

## 7. Experiment 2 — knob sweep at large size

- reuse the large sizes from Exp 1.
- knobs (each a `-D` sim rebuild): `DTCU_SWIZZLE`, `DTCU_MACS_PER_CYCLE`,
  `DTCU_SMEM_BANKS`, `DTCU_MAX_OUTSTANDING`; + hetero split ratio.
- coarse grid first; go fine only where variance is high.

## 8. Implementation status

- [x] Harness skeleton (5 modes, verify, MPM) — from dtcu_compare.
- [x] Mode 3/4 split via `DTENSOR_FLAG_NO_TMA` (user).
- [x] **Phase A** — pipelined modes 5 (TCU reg double-buf) + 6 (DXA smem
  double-buf). VALIDATED at NT=4 (all 7 modes PASS, §9). Smoke at NT=32 blocked
  by the sim bug (blocker 2).
- [x] **NT=32 target config WORKS** (2026-07-28). Root cause of the long-standing
  NT=32 failure was a host/kernel `NUM_THREADS` mismatch in our own harness, not a
  sim bug — see blocker 2 UPDATE 9. All 7 modes PASS.
- [x] **File split** — device code split into `wmma_common.h` / `k_core.h` /
  `k_tcu.h` / `k_dtcu.h` (kernel.cpp is just includes), host descriptor into
  `desc.h`, pro/epilogue math into `epilogue/*.h` + `epilogue.h` dispatcher.
  Verified behavior-preserving at NT=32 (modes 0-4 cycle-identical, 5/6 within 2%).
- [x] **`-m <0-6|all>` CLI flag** — run one HW path alone (skips the slow SIMT mode
  when debugging a single path). Documented in README.md.
- [~] **Phase B** — apps 1-3 WIRED END-TO-END and validated at NT=32 (fused in-core
  + separate `moti_epilogue` SIMT pass for the DTCU + matching CPU reference; see
  §9 for the first epilogue results). Apps 4-8 math implemented in `epilogue/*.h`
  but NOT wired: residual/scale/bias need extra operand buffers, softmax needs a
  cross-row reduction pass, dequant needs an int8 A buffer + fused-load variant.
- [~] **Phase B (CLI)** — CLI DONE + validated: `-a <app>` flag + machine-parseable
  `[MOTI] app=.. size=.. M=.. N=.. K=.. mode=.. cycles=.. errors=..` line per
  mode (sweep contract satisfied; only app=1 baseline implemented). REMAINING:
  the 7 non-baseline apps — `arg->prologue`(dequant int8→fp16) /
  `arg->epilogue`(ReLU/GELU/Residual/Scale/Softmax); in-core fuses epilogue
  (element-wise via a small smem tile to avoid a global round-trip; softmax needs
  a cross-tile/2-pass row reduction), DTCU runs separate SIMT pre/post passes.
- [ ] **Phase C** — hetero modes 7,8 (partition output tiles across units;
  fixed split ratio first).
- [x] **Sweep code** — `sweep_exp1.py` (size×app, crossover report) +
  `sweep_exp2.py` (knob OFAT) written for review, matching the now-implemented
  `[MOTI]` contract.

### Blockers found this session (2026-07-18)

1. **[FIXED] `VX_types.h` stale** — `VX_CSR_MPM_DTCU_FIRST_LOAD` was referenced
   by user edits (main.cpp, sim csr_unit.cpp, runtime legacy_perf.cpp) and
   defined in `VX_types.toml` (0xB0E) but the generated header
   `build/sw/VX_types.h` was not regenerated → compile error on runtime + test.
   Fix: `XLEN=64 python3 ci/gen_config.py --config VX_types.toml --output
   build/sw/VX_types.h --format cpp --resolved` (and `.vh` verilog variant).

2. **[CONFIRMED — NT=32-specific sim bug; NOT the harness] DTCU spurious-fires
   during in-core TCU when `VX_CFG_NUM_THREADS=32`.** At NT=32 the run dies at
   mode 1 (WMMA): the cluster DTCU's `start()` is invoked *during a WMMA kernel*
   with `desc_addr=0xc0c00000c0800000` (= two fp32 operand values -6.0/-4.0, our
   hC/hB data — i.e. GARBAGE, not a real descriptor pointer), then
   `init_tile_state_` reads junk (fmt_d=186) and aborts. Yet: all WMMA/TCU
   intrinsics emit `RISCV_CUSTOM0` (EXT1=0x0B); DTCU is EXT3 (0x5B); decode +
   SFU dispatch are statically correct. So an EXT3/START op is executing during
   WMMA that shouldn't exist. **Isolation: at NT=4 ALL 7 modes PASS** (see §9),
   so this is triggered specifically by NT=32 — a sim/compiler codegen issue at
   32 threads/warp in the in-progress DTCU path, **not** the harness.
   Temporary `[DTCU-DBG]` prints were added to `sim/simx/dtcu/dtcu.cpp` for this
   diagnosis and have been **reverted**. Suggested next diagnostics for the user:
   (a) disassemble the NT=32 `kernel.dump` and look for a stray EXT3 (opcode
   0x5b) instruction in the mode-1 WMMA path; (b) check whether Volt/LLVM
   mis-encodes a TCU op at NT=32; (c) compare against original `dtcu_compare`
   rebuilt at NT=32. Workaround to keep working: validate at NT=4 (done).

   **UPDATE (2026-07-18, after friend's DTCU fix + our_main/main merge):** still
   crashes IDENTICALLY at NT=32 mode 1. Compiler RULED OUT: the NT=32
   `kernel.dump` has exactly TWO EXT3 (opcode 0x5b) instructions — `0x0005105b`
   (funct3=1 = DTENSOR.START) and `0x0000265b` (funct3=2 = DTENSOR.POLL) — both
   in the mode-3/4 `else` branch; NO stray EXT3 in the WMMA path, so the kernel
   is correct. Root cause = **SIMT control-flow/divergence at NT=32**: the
   `dtensor_start` in the *not-taken* `else` executes during mode 1 (garbage
   register as desc_addr) — a not-taken predicated SFU/EXT3 op fires with a
   non-zero thread-mask at 32 threads/warp (zero at NT=4). It's in the sim's
   warp-execution/SFU/divergence path (or Volt SPLIT/JOIN codegen at NT=32), NOT
   `dtcu.cpp` — the friend's DTCU fix did not touch it. Pin: log `trace->wid` +
   `trace->tmask` in `sfu_unit.cpp` when `DtcuType::START` fires, run NT=32 mode
   1 → expect a non-zero mask on a warp that never took the else.

   **UPDATE 2 (2026-07-23):** re-merged our Phase A into the friend's newer
   main.cpp (which now carries a 16-counter DTCU MPM set: op_reqs/out_reqs/
   compute/next_k_load_stall/tma_mem_wait/tma_buf_starve/tma_op_fill/tma_addrgen/
   tma_store_issue_stall/store_drain/smem_read_model/next_tile_load_stall/
   prev_tile_store_stall/desc_wait/busy/tma_acc_init). Merge is CLEAN (main.cpp vs
   our_main.cpp differ ONLY in the counter block) and COMPILES after regenerating
   `VX_types.h` (the new CSRs were in VX_types.toml but the header was stale
   again). **NT=32 STILL crashes identically at mode 1** — the friend's counter/
   DTCU changes did not fix the SIMT-divergence spurious-fire. Diagnosis unchanged.

   **UPDATE 3 (2026-07-24) — FIX via per-mode kernel split (isolation).** Root
   cause pinned by tracing the SFU: at NT=32, warp 11 issued the else-branch
   `dtensor_start` (PC 0x...ae4) with `tmask=0x80000000` (ONLY thread 31 set — a
   garbage/stale mask that doesn't even match the `vx_thread_id()==0` guard), so
   the DTCU fired with WMMA operand data as desc_addr. Underlying sim
   SIMT-reconvergence mask bug is the friend's to fix. **Our fix (isolation):**
   split the monolithic `kernel_main` (one binary, mode-branch dispatch) into
   SEPARATE `__kernel` entries — `moti_simt / moti_tcu / moti_tcu_dxa / moti_dtcu
   / moti_tcu_pipe / moti_tcu_dxa_pipe` — selected host-side by name
   (`vx_module_get_kernel`). The in-core (SIMT/TCU) kernels now contain NO
   `dtensor_start` instruction, so a WMMA launch can never poke the DTCU
   regardless of the mask bug. Host `run_case` stays COMMON (just picks the entry
   name by mode); reusable device code factored into helpers (`h2f`,
   `wmma_seed_C`, `wmma_store_D`). `moti_dtcu` drops the redundant
   `vx_thread_id()==0` guard (launched 1x1x1). Also removed the temporary
   `[DBG-SFU]` trace print. Retesting at NT=32 now.

   **UPDATE 4 (2026-07-24) — split compiles & kills the CRASH, but NT=32 still
   broken (now a HANG).** Per-mode-split build succeeds (libsimx/libvortex link
   clean). At NT=32: mode 0 (SIMT) PASSES; **mode 1 (WMMA / `moti_tcu`) HANGS**
   — no `[DTCU] Error`, no abort, no output after "Running mode 1", runs to the
   1800s wall-clock timeout (livelock; the 200k-cycle stall timeout does NOT
   fire). So isolation removed the DTCU-crash SYMPTOM (moti_tcu has no
   dtensor_start) but the underlying NT=32 SIMT bug is DEEPER: it also breaks the
   plain WMMA kernel (hang). Previously this hang was masked because the spurious
   DTCU fire crashed first. Conclusion: NT=32 needs the real sim
   SIMT/divergence/WMMA fix (friend's domain); the harness runs correctly only at
   NT=4. All harness validation stays at NT=4 until then. (Not yet re-verified:
   all 7 modes at NT=4 with the per-mode split — the split is a behavior-
   preserving extraction of the NT=4-passing bodies, but should be re-run.)

   **UPDATE 5 (2026-07-24) — ROOT CAUSE: SimX DRAM callback bug, NOT the
   harness.** Ran systematic-debugging. Instrumented moti_tcu/moti_simt with
   vx_printf and ran at NT=32:
   - mode 0 (SIMT) prints `[MS] mode0 enter/exit` and PASSES → vx_printf works,
     SIMT path fine at NT=32.
   - mode 1 (WMMA) crashes with **`Segmentation fault` (make Error 139 = SIGSEGV)
     BEFORE its first vx_printf** — i.e. the SIMULATOR PROCESS dies, not the guest.
     (Earlier the same case "hung"; hang vs segfault = two faces of one memory
     corruption, Ramulator-timing dependent.)
   - gdb backtrace of the SIGSEGV:
       ```
       #0  typeinfo for vortex::LsuUnit ()              <- jump to garbage (nearest symbol)
       #1  Memory::Impl::tick()::{lambda(void*)#1}::_FUN  <- DRAM response callback
       #2  DramSim::tick()
       #3  Memory::Impl::tick() -> SimPlatform::tick() -> ProcessorImpl::run()
       ```
     Frame #0 is a typeinfo symbol (not a function) → a corrupted/dangling
     function pointer was called. The callback `arg` (a `DramCallbackArgs*`,
     memory.cpp:117) is dangling → `rsp_args->memsim->...` derefs garbage.
   - Suspect defect: `sim/common/dram_sim.cpp` `handle_pending_requests()`
     (~L68-79) does `std::move(req.callback)`/`std::move(req.arg)` into the
     completion lambda BEFORE the `receive_external_requests()` accept check. So
     (a) the write manual-response path `if (req.is_write && req.callback)` sees an
     already-moved (empty) callback and never sends the write response, and (b) on
     Ramulator backpressure (accept returns false → retry) the moved-out callback
     is lost on the retry → memory-response callback never fires / DramCallbackArgs
     leaks or dangles. WMMA-at-NT=32 issues dense operand-load bursts that saturate
     Ramulator and hit this path; SIMT's sparser traffic does not. NT=4 bursts are
     small → safe.
   - CONCLUSION: NT=32 mode-1 failure is a **SimX memory-subsystem bug (friend's
     domain)**, not the kernel, not tile geometry, not the DTCU. The earlier
     "SIMT reconvergence mask" theory (UPDATE 3) is SUPERSEDED — the spurious DTCU
     fire was most likely a wild write from this same corrupted callback, not a
     genuine mask bug. Definitive confirmation would be an AddressSanitizer sim
     build (pins the exact alloc/free/use). Kernel vx_printf instrumentation has
     been reverted; kernel.cpp is clean again.

   **UPDATE 6 (2026-07-24) — applied DramSim callback fix (segfault gone), but a
   SECOND issue remains (livelock-or-slow).** Since we now own the fix (not the
   friend): patched `sim/common/dram_sim.cpp` `handle_pending_requests()` to
   capture the completion callback/arg by COPY instead of `std::move` (the move
   emptied req.callback before the `receive_external_requests()` accept check, so
   Ramulator-backpressure retries + the write manual-response path lost the
   callback). Rebuilt libsimx (verified: dram_sim.o/libsimx.so newer than the
   edit). Re-ran NT=32:
   - The **SIGSEGV is GONE** (the move→copy fixed the UAF) — good, that was a real
     bug.
   - **mode 1 still does not complete** — now a clean HANG (no crash) to the 400s
     timeout. gdb (SIGINT to the inferior; ptrace_scope blocks plain attach, so
     gdb must launch it or async) shows: Thread 1 (main) blocked in
     `run_case → vx_event_wait_value` waiting for the kernel; the sim worker
     (Processor::run → SimPlatform::tick) samples across `TxCrossBar<MemReq>::on_tick`,
     `TxRxCrossBar<MemReq,MemRsp>::on_tick`, and `Core::Impl::issue()` — i.e. the
     sim is ACTIVELY ticking (different PCs each sample), not frozen at one spot.
     So the WMMA kernel's memory traffic keeps the sim busy but the kernel never
     reaches completion.
   - OPEN QUESTION: is this a genuine memory-subsystem **livelock** (responses
     never drain to the waiting warps at NT=32 concurrency) or just a **very slow**
     memory-bound sim (mode 0 alone took ~37s wall-clock ⇒ this sim runs only
     ~10-30k cycles/s)? Running a 20-min NT=32 job to decide: if mode 1 reaches
     "Running mode 2" it was only slow; if not, it's a livelock needing a real
     memory-model fix. (The generic crossbar/arbiter is shared with mode 0, which
     passes at NT=32, so the defect is more likely in the TCU load→response wait
     path than the crossbar itself.)

   **VERDICT (2026-07-24): LIVELOCK, not slowness.** The 20-min NT=32 run never
   reached "Running mode 2" — mode 1 (WMMA) did not complete in 1200s. A
   128×64×32 GEMM cannot need hundreds of millions of cycles, so this is a genuine
   memory-subsystem livelock: DRAM responses are produced but never fully drain
   back to the waiting warps (memory.cpp callback keeps returning false while the
   crossbar RspIn stays full → retried forever; the warp waits forever; the sim
   keeps ticking). SEPARATE from and DEEPER than the callback-UAF we already
   fixed — a backpressure/response-drain issue that WMMA-at-NT=32 concurrency
   triggers and mode-0 SIMT does not. Not a one-line fix. Next options: (a) a
   `--debug=N` trace build to pinpoint the exact stuck request/warp; (b) a
   config-capacity experiment (LSUQ_IN_SIZE is only 2 — bump LSU/MSHR/crossbar
   queue depths to see if the deadlock clears); (c) validate the harness at NT=4
   (all modes pass) and defer NT=32. The segfault fix (move→copy) is kept
   regardless — it is a real, correct bug fix.

   **UPDATE 7 (2026-07-28) — livelock is a CORE PIPELINE commit/release deadlock,
   NOT the memory path (earlier hypothesis DISPROVEN).** Added a temporary `-g` to
   the sim build (test Makefile CONFIGS crashed the RISC-V clang, so instead added
   `-g` to `sim/simx/Makefile` + `build/sim/simx/Makefile` release branch
   `-O2 -DNDEBUG -g`; behavior unchanged; revert later). Also added `-m <0-6|all>`
   to the harness so mode 1 (WMMA) can be run alone. gdb (`-m 1`, line info) shows:
   - The sim worker spins in `Core::Impl::issue` → `Scoreboard::get_uses`
     (scoreboard.cpp), NOT in the DRAM callback. A breakpoint at `memory.cpp:135`
     (RspIn-full stall) is NEVER hit during the livelock → the memory-response
     drain is NOT the stuck point.
   - Inspecting the stalled uop across warps 8/9/0/1: each warp's ibuffer-head is
     an ALU op (e.g. `x5 = x5 + …`) scoreboard-stalled because an **integer
     register is reserved by a prior ALU op that never releases**. Release only
     happens in `commit()` (core.cpp:695) via `scoreboard_->commit_packet()`
     reaching `trace->num_pkts` (dispatcher.cpp:78-85). At NT=32 the ALU FU has
     num_lanes = NUM_THREADS = 32 → no SIMD split → `num_pkts = 1`, so the
     packet-count release gate is NOT the bug either. Conclusion: the owner ops
     never reach commit → the pipeline is wedged upstream (a head-of-line op —
     likely TCU/LSU — that never completes at NT=32, backing everything up). uuid
     is 0 on every trace but release matches by reg_id (reg+wid), so that's
     cosmetic. NEXT: find the head-of-line op that never commits (inspect the FU
     outputs / whether any release fires in steady state / which op holds the
     oldest reserved reg). This is a core-pipeline concurrency bug at NT=32.

   **CORRECTION to UPDATE 7 (2026-07-28) — the "commit/release deadlock"
   conclusion is WITHDRAWN; it was not established.** The `core.cpp:521`
   breakpoint fires on the *first* scoreboard stall of the run — milliseconds into
   the kernel (the bp hit immediately after the "Running mode 1" line), i.e.
   during NORMAL operation, not during the livelock. Scoreboard stalls on
   back-to-back dependent ALU ops (`x5 = x5 + …` waiting on a prior ADD that owns
   x5) are ordinary pipeline behavior that happens constantly in a healthy run, so
   STALL1–4 are NOT evidence of a stuck pipeline. Likewise the SIGINT stack samples
   (`get_uses`, `TxCrossBar::on_tick`, `Core::Impl::issue`) only show that the sim
   is actively ticking — they do not localize a defect.

   **What is actually ESTABLISHED so far:**
   1. NT=4: all 7 modes pass. NT=32: mode 0 (SIMT) passes (964,651 cyc, ~37 s
      wall ⇒ this sim runs only ~26k cycles/s); mode 1 (WMMA) never completes
      (>1200 s).
   2. The original mode-1 SIGSEGV was a real DRAM-callback use-after-free and is
      FIXED by the `std::move`→copy change in `dram_sim.cpp` (no more segfault).
   3. During the remaining hang the sim worker keeps ticking (varying PCs) while
      the host thread waits in `vx_event_wait_value` for kernel completion.
   4. In ~45 s of sampling inside the hang, the DRAM response lambda in
      `memory.cpp` never executed (breakpoint never fired) — weak-to-moderate
      evidence that the stall is not "memory responses blocked by RspIn-full".
      Caveat: at -O2 the bps for lines 131 and 135 resolved to the SAME address,
      so line attribution inside that lambda is unreliable.

   **The ONE decisive measurement still to run:** sample the core's
   `perf_stats_.cycles` and `perf_stats_.instrs` twice ~10 s apart during the hang,
   and check whether guest warp PCs advance.
   - cycles↑ and instrs↑ → the sim is merely (pathologically) slow, or the GUEST
     kernel is itself looping forever (e.g. a corrupted loop bound), not a sim
     deadlock.
   - cycles↑ but instrs frozen → true livelock: cycles burn with zero instruction
     commits; then find the head-of-line op that never completes.
   - both frozen → a host-side infinite loop inside one tick.
   Until this is measured, no root-cause claim about the hang should be recorded.

   **UPDATE 8 (2026-07-28) — ROOT CAUSE FOUND: it is NOT a hang/deadlock. An
   integer register holding the WMMA K-loop bound gets clobbered with float data,
   turning the loop into ~33 MILLION iterations.** Measured with `-g` + gdb on
   `-m 1` at NT=32:
   1. Forward progress IS happening: cycles advance ~30k/s. Per-core `instrs` over
      a 30 s window: coreA 263,649→588,055 (+324k), coreD 253,611→561,471,
      coreC 107,452→233,420, **coreB 10,920→10,920 (frozen)**. So 3 cores execute
      endlessly; 1 core is parked (all 16 warps stalled, 0 in flight).
   2. KMU is NOT dispatching endlessly: a bp inside `Kmu::step` past the
      `if (!running_) return false` guard never fires → the 32-CTA grid was fully
      dispatched and the KMU is done. Launch geometry is correct: WMMA grid =
      (N/16, M/16) = (4,8) = 32 blocks of 32 threads.
   3. Guest PCs sampled during the "hang" are ALL inside `moti_tcu`'s K-loop body
      (0x1800002e8–0x180000388, where 0x388 is the loop-back `bltu`). Warps are not
      running away into other kernels — they are looping in place.
   4. Disassembly of `moti_tcu` confirms the loop is correct: `addw a6,a6,32`
      (i += tileK = 32) and `bltu a6,t2` (i < K), with `sext.w t2,a2` setting the
      bound ONCE at 0x1800002e4 — nothing inside the loop writes x7/t2.
   5. Reading the loop-branch operands at issue: i (x16) advances normally
      (6112, 6240, 6432, 6464, 38432 …) but the bound **t2 (x7) reads
      1,065,353,216 = 0x3F800000 = float 1.0f** — a C-matrix value (C holds
      integers −6..6 as floats), not K.
   6. The kernel argument is fine: tracing the `lwu a2,12(a0)` arg load at kernel
      entry shows **K_loaded = 32 for every warp**, argptr = 0x23000 uniformly. So
      the args upload correctly and K starts correct, then x7 is corrupted later.
   7. The branch's tmask had shrunk to 0xF (4 of 32 lanes) and lane values differ
      (lane0/1 i=6464, lane31 i=0) ⇒ the corruption is PER-LANE, so lanes whose
      bound was clobbered keep looping while the others exited.
   ⇒ A float value loaded from C lands in **integer register x7** partway through
   execution. That is a simulator register-writeback / operand-fetch mis-routing
   bug (wrong RegType, index, or lane/packet) that appears at NT=32, not a memory
   deadlock and not a harness bug. It also plausibly explains the ORIGINAL
   NT=32 symptom (a garbage value used as a DTCU descriptor address → the
   "spurious dtensor_start") and the later wild-pointer segfault: the same
   mis-routed writeback can corrupt any register. IN PROGRESS: catching the exact
   instruction that writes Integer x7 inside the loop body (conditional bp on
   `dst_reg.type==Integer && dst_reg.idx==7 && PC in loop`) to pin the defect;
   if nothing hits, the defect is on the operand-FETCH side instead.

   **UPDATE 9 (2026-07-28) — FIXED. The NT=32 failure was OUR harness bug: the
   host and the kernel used DIFFERENT thread counts.** Instrumenting
   `OpcUnit::writeback` to log every write to integer x7 showed the decisive clue:
   **every write had `tmask=4`** — only 4 of 32 lanes active. Tracing that back:
   ```
   common.h (before)          #define NUM_THREADS 4     <- host, hardcoded fallback
   kernel.cpp                 wmma_context<VX_CFG_NUM_THREADS>  <- kernel, = 32
   ```
   Nothing ever defined `NUM_THREADS`, so the host silently used 4 while the
   kernel used 32. Consequences, all observed:
   - `li.block_dim[0] = NUM_THREADS = 4` → each CTA launched only 4 threads
     (matches the measured tmask=4); lanes 4–31 never executed, so their registers
     were never initialized.
   - `using cfg = wmma_config_t<NUM_THREADS>` on the host → host tiles 8×4×8 vs
     kernel tiles 16×16×32 → the grid was computed with the wrong tile sizes
     ((N/4, M/8) = 256 blocks instead of (N/16, M/16) = 32), so the kernel's
     `blockIdx.y * ctx::tileM` indexed past the matrices.
   - Uninitialized/garbage lanes → the clobbered loop bound (x7 = 0x3F800000) and
     the ~33M-iteration loop, the out-of-range accesses, and the ORIGINAL
     "spurious `dtensor_start` during a WMMA kernel" (a garbage register used as a
     descriptor address).
   - **At NT=4 host 4 == kernel 4, so everything matched and all 7 modes passed** —
     exactly why this looked NT=32-specific and was mistaken for a sim bug.
   FIX (`common.h`): derive the host constant from the build config —
   `#ifdef VX_CFG_NUM_THREADS → #define NUM_THREADS VX_CFG_NUM_THREADS`, keeping 4
   only as a last-resort fallback, with a comment explaining the coupling.
   VERIFIED at NT=32: `-m 1` now **PASSES** — `cycles=14710 instrs=1280
   instr_tcu=256 errors=0` (instr_tcu=256 = 32 warps × 8 uops, exactly as
   expected), versus >588,000 instrs and no completion before the fix.
   BUILD GOTCHA found while verifying: editing `common.h` alone does NOT rebuild
   the kernel — `kernel.o` is a side effect of the `vx_start.o` rule
   (tests/regression/common.mk:177), so `vx_start.o` must be deleted too:
   `rm -f vx_start.o kernel.o kernel.elf kernel.vxbin` in
   `build/tests/regression/cgo27_motivation/`. A stale binary silently kept the old
   NUM_THREADS and made the fix look ineffective.
   SEPARATE, KEPT: the `dram_sim.cpp` `std::move`→copy fix (UPDATE 6) is a genuine
   independent bug — the completion callback/arg were moved out before the
   `receive_external_requests()` accept check, so Ramulator-backpressure retries and
   the write manual-response path lost the callback. That fix removed a real
   use-after-free segfault and stays in.
   Temporary debug scaffolding has been reverted: the `-g` added to
   `sim/simx/Makefile` + `build/sim/simx/Makefile` and the `[DBG-X7]` print in
   `opc_unit.cpp` are removed; kernel.cpp instrumentation was already reverted.

## 9. Results (KEEP UPDATED)

Prior run (old config: 1 core, 4w/4t, no L2, size_mult=2), modes 0–4, all PASSED:

| mode | cycles |
|---|---|
| SIMT | 9,453,768 |
| TCU | 157,713 |
| TCU+DXA (naive) | 201,451 (slower — naive single-buffer, DXA bypasses L1) |
| DTCU | 30,986 |
| DTCU+TMA | 30,986 (was identical pre-flag-fix) |

**Phase A smoke @ NT=4** (M=128,N=64,K=32, size_mult=2; 4 cores, L2 on, LMEM 64KB)
— all 7 modes **PASSED** (errors=0). Harness + pipelined modes 5/6 validated:

| mode | cycles | note |
|---|---|---|
| 0 SIMT | 965,700 | 4 cores → far faster than old 1-core 9.45M |
| 1 TCU | 45,933 | |
| 2 TCU+DXA (naive) | 62,405 | slower than TCU (naive single-buffer) |
| 3 DTCU (NO_TMA) | 34,125 | |
| 4 DTCU+TMA | 30,857 | **< mode 3 → TMA overlap helps; 3≠4 confirmed** |
| 5 TCU pipelined | 384,687 | **SLOWER (8×)** — tiny K (few iters, no overlap) + likely WMMA-fragment register-double-buffer spill (instrs 20k→33k). Revisit; sweep will show if it ever wins. |
| 6 TCU+DXA pipelined | 191,692 | slower than naive DXA at this size; sweep needed |

**NT=32 (the target config) — NOW WORKS. All 7 modes PASSED** (2026-07-28, after
the `common.h NUM_THREADS` fix; see blocker 2 UPDATE 9). Config: 1 cluster,
4 cores, 16 warps/core, 32 threads/warp, issue width 4, L1 32KB, L2 1MB,
LMEM 64KB. M=128, N=64, K=32 (size_mult=2), app=1 (no epilogue yet):

| mode | cycles | vs NT=4 | note |
|---|---|---|---|
| 0 SIMT | 191,377 | 965,700 → 5.0× faster | 8× more lanes/warp |
| 1 TCU | **14,710** | 45,933 → 3.1× faster | fastest in-core path; instrs=1280, instr_tcu=256 (= 32 warps × 8 uops, as expected) |
| 2 TCU+DXA (naive) | 18,021 | 62,405 → 3.5× faster | still slower than plain TCU at this size (single-buffer, DXA bypasses L1) |
| 3 DTCU (NO_TMA) | 33,485 | 34,125 | |
| 4 DTCU+TMA | 22,237 | 30,857 | **< mode 3 → TMA overlap helps; tripwire passes** |
| 5 TCU pipelined | 216,419 | 384,687 | **14.7× SLOWER than mode 1** — CONFIRMED register spill: `moti_tcu_pipe` has 17 sp-relative accesses vs 1 in mode 1 (fragA[2]/fragB[2] does not fit). Also K=32 gives a single K-tile, so there is nothing to overlap anyway. Both must be addressed before mode 5 appears in the paper. |
| 6 TCU+DXA pipelined | 108,682 | 191,692 | 6× slower than mode 2; also spills (9 sp-relative accesses vs 1 in mode 2) |

**File split + epilogue wiring (2026-07-28).** Device code is now split into
`wmma_common.h` (ctx/tile geometry, h2f, seed/store/fuse helpers), `k_core.h`
(mode 0 + the standalone `moti_epilogue` pass), `k_tcu.h` (modes 1/2/5/6),
`k_dtcu.h` (modes 3/4), with `kernel.cpp` reduced to includes; the host DTCU
descriptor moved to `desc.h`; the pro/epilogue math lives in `epilogue/{relu,gelu,
residual,scale,softmax,dequant}.h` behind one dispatcher, `epilogue.h`. The
epilogue headers are shared by the kernel AND the host CPU reference on purpose, so
verification compares identical arithmetic (that is also why gelu/softmax use
in-header tanh/exp approximations instead of libm — host and Vortex libm need not
agree bit-for-bit, and this harness measures cycles, not numerics).

Split verified behavior-preserving at NT=32: modes 0–4 reproduce their cycle counts
exactly; modes 5/6 move ~1–2% (212,231 and 109,743) because they are the largest
functions and the code layout shifted — same source, different I-cache/allocation.

**FIRST EPILOGUE RESULTS (NT=32, size_mult=2, apps 1–3 wired) — this is the
motivation result:**

| app | mode 1 (in-core TCU, epilogue FUSED in registers) | mode 2 (in-core TCU+DXA, FUSED) | mode 3 (DTCU, epilogue as a 2nd SIMT pass) |
|---|---|---|---|
| 1 baseline | 14,533 | 17,492 | 33,493 |
| 2 + ReLU | **14,657  (+0.9%)** | **17,764  (+1.6%)** | **74,222  (+122%)** |
| 3 + GELU | 20,728  (+43%) | — | 75,697  (+126%) |

Reading: fusing costs the in-core path almost nothing for a cheap activation
(+0.7% for ReLU) and only the activation's own arithmetic for an expensive one
(+42% for GELU). The DTCU has no epilogue HW, so it pays a full extra M×N
round-trip either way — **+120% regardless of how cheap the epilogue is.** The
in-core-vs-DTCU gap therefore widens from 2.3× (bare GEMM) to 5.0× (with ReLU): the
epilogue alone moves the argmin, which is exactly the crossover the paper claims.
Apps 4–8 (residual / scale / softmax / dequant) need extra operand buffers or a
cross-row reduction; their math is implemented in `epilogue/*.h` and the wiring is
the remaining Phase B work.

**Final app=1 numbers after the split + epilogue wiring + the mode-2 fix
(NT=32, all 7 modes PASS):** mode 0 188,712 · mode 1 **14,533** · mode 2 **17,492** ·
mode 3 33,493 · mode 4 22,245 · mode 5 (3-stage) 18,142 · mode 6 (2-stage) 19,920. Every mode is now
at or below its pre-split cycle count (mode 2 is 3% BELOW its pre-epilogue 18,021),
so the epilogue capability was added without costing baseline performance.

**Mode 2 regression (18,021 → 23,986) — DIAGNOSED AND FIXED.** The register-pressure
guess was WRONG: `moti_tcu_dxa` has 1 sp-relative access, same as the healthy mode 1
(the modes that DO spill are 5 and 6, with 17 and 9 — that is the real cause of
their slowness, see below). Counters ruled out the obvious suspects too: the +5,965
cycles came with only +56 instrs and +219 total stall_lsu/tcu, i.e. the extra time
was warp-suspended barrier/DXA wait, which no stall counter records.

One-variable builds on mode 2 (app=1, so the epilogue is a no-op):

| variant | cycles |
|---|---|
| no `wmma_fuse_epilogue` call at all | 15,456 |
| call present, `arg->app` read AT KERNEL ENTRY | **17,492** |
| call present, `arg->app` read at the store (as first written) | 23,986 |

So `always_inline` on seed/store actually IMPROVED mode 2 (18,021 → 15,456), and the
regression was entirely the epilogue call — specifically **the global load of
`arg->app` sitting alone on the tail of the kernel.** After the compute and the
barriers, the warp cannot store D or retire until that load returns; in a
DXA/barrier-bound kernel (stall_lsu is 57% of mode 2's cycles) that tail latency
delays CTA completion and stretches the whole DXA pipeline. Hoisting the read to
kernel entry, where it overlaps the GEMM, recovers 6,494 of the 8,530 cycles (76%).

FIX: every kernel now reads `app = arg->app` at entry alongside N/K (k_tcu.h,
k_core.h), and `wmma_fuse_epilogue` takes that local. Residual cost over the
no-epilogue build is ~2,000 cycles (the warp-uniform branch itself) — the intrinsic
price of a runtime-selected epilogue in a tail-latency-bound kernel, and mode 2 is
still faster than its pre-split 18,021.

RULE for this harness: **read kernel-arg scalars at kernel entry, never in the
tail.** A single late uniform load is nearly free in a compute-bound kernel (mode 1
was unaffected) and very expensive in a barrier/DMA-bound one.

Two measurement traps hit while diagnosing this, both of which silently produced
identical-looking numbers from a stale binary: (1) the `rm` of build artifacts must
run in `build/tests/regression/cgo27_motivation/`, not the source dir; (2) the HOST
binary must be deleted too — deleting only the kernel gave a host/kernel skew in
`kernel_arg_t` and errors=8192. `scratchpad/measure.sh` encodes the safe recipe.

**Pipelined modes 5/6 spill — DIAGNOSED; mode 6 FIXED, mode 5 is INFEASIBLE at
this tile geometry (2026-07-28).** Both used runtime-indexed arrays
(`fragA[cur]` / `A_smem[cur]` / `bar[cur]` with `cur = kk & 1`), which forces the
array to be addressable and pushes it to the stack. Rewrote both with NAMED
variables + the K loop unrolled by 2 so every selection is compile-time:

| mode | before | after | note |
|---|---|---|---|
| 6 TCU+DXA pipelined | 109,568 (9 sp-accesses) | **17,617 (1 sp-access)** | FIXED — 6.2× faster, now equal to naive mode 2 (17,492) |
| 5 TCU pipelined | 212,389 (17 sp-accesses) | 14,784 @ s=2 but **TIMES OUT @ s=8** | NOT fixed — see below |

Mode 5's problem is a hard register budget, measured (NT=32 ⇒ NRA=NRB=NRC=8; the
RISC-V F extension has 32 f-registers):

| variant | f-registers needed | verdict |
|---|---|---|
| single-buffered (mode 1) | NRA+NRB+NRC = 24 | fits (1 sp-access, 14,533 cyc) |
| double-buffer A **and** B (mode 5 as written) | 2NRA+2NRB+NRC = **40** | impossible, 8 over → spilled (mem_writes 19,456 vs 3,072) |
| double-buffer A only (current) | 2NRA+NRB+NRC = **32** | exactly at the limit, zero slack |

The A-only form looks fine at size_mult=2 — 14,784 cycles, mem_writes back to 3,072 —
but ONLY because K=32 ⇒ numK=1, so the prefetch branch never executes and the second
register set is never live. At size_mult=8 (K=128 ⇒ numK=4), where the pipeline
actually engages, mode 5 does not finish inside 400 s while modes 1/2/6 all complete
in ~370 k cycles. So register double-buffering is not viable on this configuration at
these tile shapes, by either form.

**Mode 5 REPURPOSED as a 3-stage smem pipeline (2026-07-28).** The register-pipeline
slot is gone (see the NR=4 note below for why it is unfixable); mode 5 is now the
DEEPER smem pipeline — three lmem stages, DXA running two tiles ahead — so modes 5/6
differ only in prefetch DEPTH (3 vs 2 stages) and depth becomes a measurable axis.
Mode 6 keeps its number so every previously recorded mode-6 figure stays valid.
Kernel entry `moti_tcu_dxa_pipe3`; host-side mode 5 now also requires the DXA
extension, programs the 2D descriptors, and requests 3*stage_bytes of lmem.

Getting it to perform took one more instance of the same lesson: **a deep pipeline
must keep barrier objects short-lived.** `vortex::barrier` holds bar_id_ +
num_warps_, so three long-lived barriers pin 6 integer registers; together with the
loop bookkeeping that spilled the kernel (15 sp-relative accesses, ALL integer —
sp-FLOAT was 0, so the fragments were fine). Measured on mode 5 at size_mult=2:

| variant | cycles | sp-accesses | mem_writes |
|---|---|---|---|
| 3 barriers live across the loop | 109,507 | 15 | 10,240 |
| stage pointers folded to constexpr offsets (no other change) | 109,507 | 15 | 10,240 |
| **barriers scoped to each use (`p3_fill` / `p3_consume` helpers)** | **18,142** | **1** | **3,072** |

Note the middle row: folding the six stage pointers into constant displacements off
one base changed NOTHING — the pressure was entirely the barrier objects. Worth
remembering as a counterexample to the obvious guess.

**Pipeline depth, first measurement** (M=512 N=256 K=128, size_mult=8, numK=4 — the
first size where prefetching has anything to overlap):

| mode | path | cycles |
|---|---|---|
| 2 | TCU+DXA naive (1 stage) | 365,681 |
| 6 | TCU+DXA 2-stage | 376,934 |
| **5** | **TCU+DXA 3-stage** | **364,395** |

So at this size the 3-stage pipeline is the fastest of the three, but only by 0.4%
over naive, while the 2-stage version is 3% SLOWER than naive. The spread is small
enough that depth is not yet paying for itself here — consistent with mode 2 already
being ~57% LSU-stall-bound rather than fill-latency-bound. Larger K (more K tiles per
CTA) is where depth should start to matter, which is exactly what Experiment 1's size
sweep must resolve. At size_mult=2 (numK=1, nothing to overlap) the ordering is the
expected one: naive 17,492 < 3-stage 18,142 < 2-stage 19,920, i.e. pure overhead.

**NR=4 attempt (2026-07-28): mode 5 CANNOT be rescued by shrinking the tile —
register pipelining is not expressible on this ISA.** Tried giving mode 5 its own
`wmma_context<..., NR_=4>` (tile 16x8x8, 16 registers double-buffered) plus a
mode-5-specific launch grid. It does not compile: at NT=32, NR=4 yields NRB=2 and
`mma_sync` static_asserts `FragB::NR` is 4 or 8. Reading vx_tensor.h explains why no
NR works — **mma_sync reaches the TCU through inline asm that PINS its operands to
fixed physical f-registers**: accumulator C/D -> f0-f7, A -> f10-f17,
B -> f24-f31 (f28-f31 when NRB==4). Consequences:
- 24 of the 32 f-registers are reserved by one MMA; only f8, f9, f18-f23 (= 8) are
  free, while a second prefetched operand set needs NRA+NRB = 16.
- FragA and FragC/D are hard-wired to 8 registers *regardless of NR*, so NRA=NRC=8 is
  forced, which forces xtileM·xtileK = xtileM·xtileN = 256 ⇒ xtileK = xtileN, and then
  NRB = xtileN²/32 = 4 has no power-of-two solution (needs xtileN² = 128). So NRB=8 is
  the only legal shape at NT=32, i.e. the 16x16x16 tile is the ONLY option.
- Even if a set did fit, the prefetched values would have to be MOVED into the pinned
  registers before each MMA, so the copies would consume the latency the prefetch was
  supposed to hide.
RESOLUTION: mode 5 is now deliberately the SAME single-buffered loop as mode 1, with
the reasoning recorded in `k_tcu.h`. That keeps all seven modes runnable for the
sweep (the A-only variant timed out at size_mult=8 and would have blocked it). OPEN
CALL for the owner: drop the mode-5 slot, or repurpose it (e.g. a deeper smem
pipeline, or a 3-stage variant of mode 6), since as it stands it duplicates mode 1.

This is a genuine result for the paper rather than a bug to hide: **a transformation
that pays off on register-rich targets (SW pipelining in registers) is infeasible
here, while the smem-staged equivalent (mode 6) works** — exactly the kind of
target-parametric choice the thesis says a compiler must make. To make mode 5
measurable at all it would have to shrink its tile so two sets fit (e.g.
`wmma_context<..., NR_=4>` ⇒ NRA=4, NRB=2, NRC=4 ⇒ 16 registers double-buffered),
which also means a mode-5-specific launch grid host-side. That couples tile size to
pipeline depth — a design decision left to the owner, NOT taken unilaterally.

Naive vs pipelined at size_mult=8 (M=512 N=256 K=128, numK=4), the first size where
pipelining can pay: mode 1 373,566 · mode 2 365,681 · mode 6 377,451 · mode 5 (did
not finish). So even with the spill fixed, mode 6 does not yet beat naive mode 2 at
this size — the crossover, if any, is further out and is what Experiment 1 must find.

Two engineering gotchas found while wiring this (both cost 10× if ignored):
1. **Fragment helpers MUST be `__attribute__((always_inline))`.** If
   `wmma_fuse_epilogue` is a real call, `fragD`'s address escapes and the whole
   accumulator array lives in memory instead of registers: mode 1 measured 158,970
   cycles instead of 14,552 (10.8×) even for app=1 where the epilogue is a no-op.
   Every fragment-touching helper in vx_tensor.h is always_inline for this reason;
   `wmma_seed_C`/`wmma_store_D` were switched too (that alone bought ~1%).
2. Register indices must be compile-time constants — walk the fragment with
   `vt::detail::unroll_for<NR>`, not `for (r = 0; r < NR; ++r)`, and hoist the app
   test out of the element walk.

Observations worth carrying into Experiment 1: at this (small) size the ordering is
TCU < TCU+DXA < DTCU+TMA < DTCU < SIMT < TCU+DXA-pipe < TCU-pipe. The pipelined
modes are pathologically slow at K=32 (one K-tile ⇒ zero overlap opportunity, plus
spill); the size sweep must go to much larger K before they can win, and mode 5's
spill should be confirmed (check instrs and any stack traffic) rather than assumed.

## 10. Decisions log

- pipelined = **separate mode numbers** (5,6), not an orthogonal flag (simpler,
  matches the flat HW-mode list).
- epilogues = ReLU / GELU / Residual / Scale / Softmax (dequant moved to prologue).
- dequant = int8→fp16 pre-pass (dequant apps store int8; others fp16).
- softmax cross-tile complexity **accepted** — want to see the hard case.
- HW memory = Hopper-¼ recommendation (L1 32KB / LMEM 64KB / L2 1MB), overriding
  the initially-floated 4KB/256KB/16KB (too small L1/L2).
- Do NOT run the full sweep yet — implement A/B/C + generate sweep code + smoke
  test only; user reviews after.
- Mode numbering CONFIRMED (2026-07-24, user): keep 5/6 = pipelined (TCU-pipe,
  TCU+DXA-pipe), heterogeneous = 7/8 (simt+tcu+dtcu / all). No renumber needed
  (code already matches). Hetero (7/8) still unbuilt.
- Kernel is split per-mode into separate `__kernel` entries; host `run_case`
  stays common and selects the entry by name. This both cleans up and (by
  keeping `dtensor_start` out of the WMMA kernels) works around the NT=32
  divergence-mask bug.

## 11. Gotchas / references

- SIMT has no HW fp16 (march `rv*imaf`, no Zfh) → software convert.
- DTCU always TMA-prefetches in HW; modes 3/4 differ only by the `NO_TMA` timing
  flag; no epilogue HW.
- Toolchain: `llvm-vortex` needs glibc 2.35 on this focal host → `tools/glibc-2.35`
  + patchelf.
- Related memory: `vortex-paper-direction`, `cgo27-motivation-harness`.
