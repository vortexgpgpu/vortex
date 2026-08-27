# FireSim integration — host-decoupled FPGA simulation of Vortex

This document describes how Vortex runs as a [FireSim](https://fires.im/)
target on a Xilinx Alveo U55C. It covers three things:

1. The **execution model** — what host-decoupled simulation means for a
   driver, and why the runtime *steps* the target rather than waiting on it.
2. The **integration architecture** — the Chisel target, the RTL wrapper, the
   transport layer, and how a kernel launch reaches the card.
3. The **build flow** — elaboration through place-and-route, and the
   invariants that make the difference between a design that fits and one
   that silently does not.

FireSim itself is an external dependency, developed in a clone of
[`vortexgpgpu/firesim`](https://github.com/vortexgpgpu/firesim) (branch
`vortex_3.x`, based on upstream 1.21.0) at `~/dev/firesim`, and consumed by
the Vortex build through `$(FIRESIM_PATH)`, which `config.mk` defaults to
`$(TOOLDIR)/firesim` — the same source/tool split POCL, Mesa and LLVM use.
Vortex's own changes to FireSim are small — 8 files, ~324 lines — and are bug
fixes and U55C platform plumbing rather than divergence; see §6.

For running kernels on hardware generally, see
[`fpga_setup.md`](../fpga_setup.md); for the other simulation vehicles, see
[`simulation.md`](../simulation.md).

---

## 1. Execution model

### 1.1 What "host-decoupled" means

FireSim does not emulate Vortex at wall-clock speed and then let the host
observe it. It compiles the design through **Golden Gate** (MIDAS) into a
FAME-1 transformed model in which *every* target clock edge is explicitly
scheduled by the host. The FPGA holds the transformed design; the host driver
decides how many target cycles to advance and when.

The practical consequence, and the thing that shapes the entire driver: **the
target does not run unless the host tells it to.** There is no free-running
clock to poll. A driver that issues an MMIO read and waits will wait forever,
because nothing advanced the design between the request and the read.

This is the single most important difference from the XRT flow, where the
kernel clock free-runs and the host merely observes.

### 1.2 Target cycles vs. host cycles

Two clocks matter and must not be conflated:

| | what it counts | where it comes from |
|---|---|---|
| **target cycles** | cycles the Vortex design advanced | `clockmodule_t::tcycle()`, and Vortex's own `PERF: cycles=` |
| **host cycles** | cycles the FPGA spent simulating them | `clockmodule_t::hcycle()` |

Their ratio is **FMR** (FPGA-to-Model Ratio) — how many host cycles it costs
to simulate one target cycle. FMR is the honest measure of simulation
efficiency; the effective target frequency is `host_freq / FMR`.

**Do not quote FireSim's `Target Cycles Emulated` as a kernel cycle count.**
It reports cycles the *driver stepped*, not cycles the kernel took, and the
driver steps in growing batches (§3.3). A kernel is only observed to have
finished at the first poll *after* it does, so the figure is quantized to the
batch schedule and can overstate a short kernel by up to 4×. Two workloads
that finish inside the same inter-poll interval report the identical number.
Vortex's `PERF: cycles=` counter is exact and is what every measurement in
this repository uses.

---

## 2. Integration architecture

### 2.1 Layering

```
  tests/regression/<app>            application
      │  vx_* API
  sw/runtime/firesim/vortex.cpp     runtime HAL       (driver ABI, XCLBIN_PATH)
      │  firesim_sim class
  sim/firesim/firesim_sim.cpp       transport         (request queue, stepping)
      │  peek_poke / loadmem bridges
  $(FIRESIM_PATH)/sim/…             FireSim driver    (simif_vitis, widgets)
      │  XRT
  ─────────────────────────────── PCIe ───────────────────────────────
  FireSim-generated.sv              FAME-1 model      (Golden Gate)
      │  ctrl_* / AXI
  hw/rtl/afu/firesim/VX_firesim_wrap.sv               AFU wrapper
      │
  Vortex                                              the design under test
```

Everything above the PCIe line except the FireSim driver lives in this
repository; that is deliberate, so the FireSim checkout stays close to
upstream.

### 2.2 The Chisel target — `hw/syn/firesim/src/main/scala/VortexTarget.scala`

`VortexDUT` instantiates Vortex as an opaque Verilog **BlackBox**. Golden Gate
cannot see inside it, which is the point: the FAME transform channelizes the
BlackBox's boundary signals and leaves its internals alone.

`VortexConfig` sets the memory interface the target presents:

```scala
NastiParameters(dataBits = 512, addrBits = 32, idBits = 4)
```

`idBits` is **paired with `PLATFORM_MEMORY_ID_WIDTH` in
`hw/syn/firesim/Makefile`** and neither may be changed alone — one shapes the
IO the BlackBox is instantiated with, the other shapes the module's own ports,
and Verilog pads or truncates a disagreement with no diagnostic.

It is also bounded from both sides, which is not obvious:

- **Too large and it overflows.** FASED sizes its read-response egress with
  `1 << idBits` in Scala `Int` arithmetic. At 32 that shift wraps to 1, which
  both selects the untranslated egress branch and gives it a single queue for
  every AXI id. Responses then leave carrying the requesting transaction's id
  and another transaction's data. Elaboration succeeds and nothing warns — the
  generated design is simply smaller and wrong.
- **Too large and it does not fit.** The response deinterleaver and the id-yank
  queues carry per-id state, so model area scales with the id space. At 8 that
  is 750,955 LUTs as memory against the 600,960 a U55C has. At 4 it is 51,678.

4 is what Vortex actually drives: `VX_mem_to_axi`'s tag buffer is engaged
because `UUID_WIDTH` puts the upstream tag past any AXI id width, so reads
carry `CLOG2(TAG_BUFFER_SIZE) = 4` bits. It also matches FASED's `maxReads`
of 16.

### 2.3 The RTL wrapper — `hw/rtl/afu/firesim/VX_firesim_wrap.sv`

A thin AFU presenting Vortex as:

- one AXI4 memory port (`m_axi_mem_0_*`),
- a DCR request/response pair for configuration,
- `start` / `busy` for launch and completion.

`VX_CFG_PLATFORM_MEMORY_NUM_BANKS=1` is set for this wrapper. Left at the
default of 2, Vortex builds two internal memory ports and `VX_mem_to_axi`
arbitrates them onto the one bank, recovering the originating port from the
*returned* AXI id (`rsp_xbar_sel_in = m_axi_rid[0 +: NUM_PORTS_IN_BITS]`).
That makes correctness depend on the memory model echoing `rid` bit-exact — an
assumption real AXI satisfies but a timing model need not. One port removes the
dependency entirely and matches what the wrapper presents.

### 2.4 Memory model

Target DRAM is served by **FASED**, FireSim's parameterized memory timing
model, backed by host DRAM through the LoadMem widget. The Vitis platform
binds `host_mem_0` across HBM[0:7] with a 4 GiB window — a 32-bit target spans
4 GB, with its program image at `0x80000000` and its stack just under 4 GB, so
a narrower binding puts kernel uploads outside the served window and the target
fetches memory nothing ever wrote, with no error anywhere.

---

## 3. The transport layer — `sim/firesim/firesim_sim.cpp`

The largest single piece of the integration, and where host-decoupling shows
up as real complexity.

### 3.1 Threading

FireSim's bridges are **not thread-safe** and must only be touched by the
thread running the simulation loop. The transport therefore owns a simulator
thread and marshals every operation onto it as a closure:

```
caller ──push labelled_request_t──> queue ──> simulator thread ──> bridges
       <──────── condition variable ────────
```

Each request carries a label so a trace attributes a stall to an operation
rather than showing a column of identical entries.

### 3.2 Memory access — the chunk-unit invariant

`access_mem` moves data through LoadMem one host beat at a time, with
read-modify-write on unaligned edges.

**LoadMem's chunk is counted in 32-bit words, not bytes.** `read_mem` and
`write_mem` push exactly that many *words* through the data register. Treating
it as a byte count uploads a quarter of each beat and advances the address by a
quarter of a beat — storing two live bytes per eight and dropping every write
that does not land on a beat boundary. A kernel image uploaded that way is
fetched as a mixture of written and stale memory, and nothing reports an error.

Whole-beat writes bypass the read-modify-write path.

### 3.3 Advancing the target — adaptive step batching

`is_busy()` must both catch the launch edge and give a long kernel throughput.
These pull in opposite directions: a small step never misses `busy` asserting
but costs an MMIO round trip per few cycles; a large step is efficient but can
step past a short kernel entirely.

The driver grows the batch: floor of 16 target cycles, ×4 per poll up to a
ceiling, dropping back to the floor when the target goes idle. Cumulative poll
boundaries therefore land at 16, 80, 336, 1360, 5456, 21840, …

This is also the origin of the `Target Cycles Emulated` quantization in §1.2 —
the counter is sampled at the poll that *observes* completion, not at the
busy→idle transition itself.

### 3.4 DCR reads must single-step

`dcr_rsp_valid` is a one-cycle pulse passed straight out of the core, so a
batched step lands past it and the read reports unanswered on a target that
replied. `dcr_read` therefore advances one target cycle per iteration, and its
bound is expressed in cycles rather than polls.

The cost is one MMIO round trip per target cycle — tenths of a second for a
cache flush on the card, minutes under emulation. Batched polling would need
the response **latched at the wrapper**, which is a target-side change and a
new bitstream; it is a known follow-up rather than a driver-side workaround.

### 3.5 Diagnostics

Vortex's own performance counters are the only view inside the core that does
not require re-elaborating the design, and they are read over the DCR path.
They are gated behind `PERF_ENABLE` (see `PERF_DEFINES` in
`hw/syn/firesim/Makefile`); **without it every counter reads zero, which is
indistinguishable from an idle core rather than an absent instrument.** The
driver says so explicitly rather than leaving it to be inferred.

---

## 4. Build flow

```
make -C hw/syn/firesim stage      copy Chisel/C++ targets into $(FIRESIM_PATH)
make -C hw/syn/firesim sources    Verilator source list (packages + wrapper)
make -C hw/syn/firesim vivado-sources
                                  flatten Vortex with macros inlined (-P)
cd $(FIRESIM_PATH)/sim && make replace-rtl PLATFORM=vitis DESIGN=VortexTarget
                                  Golden Gate elaboration
cd platforms/vitis/… && make bitstream DEVICE=… FREQUENCY=60.0 STRATEGY=TIMING
                                  synthesis + place-and-route
```

### 4.1 Two source lists, and why

Verilator resolves Vortex by **library search**, so `sources` names only the
packages and the wrapper — handing it the whole tree as a flat list makes it
stop treating the macro-computed width parameters as constants. Vivado has no
such search and must be handed every file, so `vivado-sources` flattens the
tree into one directory with macros already inlined. The constant-parameter
failure that rules flattening out for Verilator is an elaboration behaviour,
not a synthesis one.

### 4.2 Synthesis invariants

**`SYNTH_DEFINES := -DSYNTHESIS -DVIVADO -DNDEBUG` is not optional.**
`VX_dp_ram` and `VX_sp_ram` gate all block-RAM inference behind `` `ifdef
SYNTHESIS ``, and `gen_sources.sh -P` *preprocesses* into the flattened tree —
so without it the BRAM branch is not merely inactive, it is deleted before
Vivado sees it and every memory in the design lands in distributed RAM. The
symptom is a place-and-route failure hours later at ~750k LUTs-as-memory, with
nothing pointing at the cause.

**The Makefile is a prerequisite of the flatten rule.** Without it, changing
any define does not invalidate the flattened tree and the next build silently
reuses the previous elaboration. Note it must be `$(firstword $(MAKEFILE_LIST))`
— `lastword` resolves to `build/config.mk`.

**Guard the elaboration, not just the build.** `v++` runs synthesis into
place-and-route without stopping, so an over-utilized design surfaces only
after the placer has been tried. Two cheap checks catch the failure modes
above before hours are spent: that FASED elaborated a per-id response path
(`ReorderBuffer` present, multi-bit `MultiQueue` selectors), and that the
post-synthesis utilization report fits. Any such watchdog **must be
freshness-qualified** against the current run's timestamp — an unqualified
lookup reads the previous build's report and kills a healthy build.

---

## 5. Measured results

U55C, kernel clock constrained at 60 MHz, `STRATEGY=TIMING`. Configuration:
1 cluster / 1 core / 4 warps / 4 threads, SIMD 4, L1 only, no A or C
extension, no TCU/DXA/RTU/TEX/RASTER/OM, `FPU_USE_DSP=0`.

### 5.1 Utilization (kernel only)

| resource | used | available | % |
|---|---|---|---|
| CLB LUTs | 131,578 | 1,303,680 | 10.09 |
| — as logic | 79,900 | 1,303,680 | 6.13 |
| — as memory | 51,678 | 600,960 | 8.60 |
| CLB registers | 62,004 | 2,607,360 | 2.38 |
| Block RAM tiles | 109 | 2,016 | 5.41 |
| URAM | 8 | 960 | 0.83 |
| DSP | 24 | 9,024 | 0.27 |

DSP usage is near zero because this configuration builds the FPU from LUTs by
choice (`FPU_USE_DSP=0`), not for want of DSPs.

### 5.2 Timing

Kernel clock `clk_out1_firesim_clocking` closes at **WNS +5.283 ns** over
548,101 endpoints with zero failing, implying Fmax ≈ 87.8 MHz against the
60 MHz constraint. The design-level WNS of +0.003 ns belongs to the Vitis
shell's 450 MHz `hbm_aclk`, **not** to Vortex — reading the global number as
the kernel's inverts the conclusion.

### 5.3 Kernels

| app | args | instrs | cycles | IPC |
|---|---|---|---|---|
| demo | -n64 | 2,444 | 6,429 | 0.380 |
| vecadd | -n1024 | 6,160 | 18,837 | 0.327 |
| sgemm | -n32 | 48,400 | 115,490 | 0.419 |
| sgemm | -n1024 | 1,187,250,192 | 3,229,839,120 | 0.368 |

`sgemm -n1024` completes in **84.6 s** of kernel time. Instruction counts match
rtlsim exactly; cycle counts differ by −1.3% to +7.1%, which is the FASED
memory timing model rather than the core — FPGA and hw_emu cycle counts agree
exactly.

For scale: `sgemm -n32` takes 1228 ms on rtlsim and 8 ms on the card, ~153×.

---

## 6. Relationship to upstream FireSim

The FireSim checkout is a **git clone, not a vendored copy**. Vortex's changes
live on branch `vortex_3.x` of `vortexgpgpu/firesim`, based on upstream tag
1.21.0, and amount to 8 files:

| area | files | why |
|---|---|---|
| Vitis platform | `simif_vitis.cc`, `VitisShim.scala` | DRAM aperture across HBM[0:7] with a 4 GiB window; host-memory base address |
| driver lifetime | `entry.cc` | tear the platform down at the end of `entry()`; a global `unique_ptr` is destroyed during static destruction, after XRT's own statics, and segfaults in the device registry |
| cycle accounting | `simulation.{cc,h}` | `end_tcycle` is read absolute and never rebased, so it accumulates across runs and FMR divides a per-run quantity by a cumulative one |
| build | `cl_firesim/Makefile`, `gen_xo.tcl`, `package_kernel.tcl` | U55C build fixes; package the flattened Vortex sources into the kernel |

All are candidates for upstreaming: only the `HOST_MEM_SP` default is
U55C-specific, and it is parameterized rather than hardcoded. The branch should
shrink over time as changes are accepted upstream.

---

## 7. Design invariants

1. **`idBits` and `PLATFORM_MEMORY_ID_WIDTH` move together.** A disagreement is
   padded or truncated silently. Raising either requires re-checking
   utilization, not merely avoiding the `1 << idBits` overflow.
2. **`SYNTHESIS` must reach the flattened tree.** Its absence is invisible
   until place-and-route and costs hours.
3. **Vortex's `PERF:` counters are the cycle authority.** FireSim's
   `Target Cycles Emulated` is quantized by the step schedule.
4. **The target only advances when stepped.** Any new driver operation that
   waits on target state must step, and must know whether the signal it waits
   on is a pulse or a level.
5. **All-zero performance counters mean the instrument is absent**
   (`PERF_ENABLE` off), not that the core is idle.

---

## 8. Known limitations

- **Metasimulation is dead.** FireSim's Verilator metasim builds and starts,
  then deadlocks before target cycle 0 (65 threads in `futex_wait`, 0% CPU, no
  output past `entering simif.run`). It is deliberately not fixed: the card
  supersedes every purpose it served. Anyone reaching for it should expect this
  rather than read the silence as a slow build.
- **DCR reads are slow** by one MMIO round trip per target cycle (§3.4).
- **Regression coverage is partial.** The kernels in §5.3 plus `conv3`,
  `dogfood`, `dotproduct` and `mstress` pass on the card; the suite has not
  been swept exhaustively, and a large share of it requires extensions this
  configuration disables.
- **The conda environment is not relocatable** as built. It records its own
  prefix in every script shebang and in package metadata, so it must be
  packaged with `conda-pack` and unpacked with `conda-unpack`; see the
  packaging notes in `ci/toolchain_prebuilt.sh`.
