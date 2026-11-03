# Vortex Simulation Methods 

### RTL Simulation

[Verilator](https://www.veripool.org/projects/verilator/wiki) is a Verilog/SystemVerilog design simulator that converts the Verilog HDL to single- or mult-ithreaded C++/SystemC code to perform the design simulation. An installation guide for Verilator is located [here.](https://www.veripool.org/projects/verilator/wiki/Installing)

### Cycle-Approximate Simulation

SimX is a C++ cycle-level in-house simulator developed for Vortex. The relevant files are located in the `simx` folder. The [readme](README.md) has the most detailed instructions for building and running simX.

- To install on your own system, [follow this document](install_vortex.md).
- For the different Georgia Tech environments Vortex supports, [read this document](environment_setup.md).

### FGPA Simulation

The guide to build the fpga with specific configurations is located [here.](fpga_setup.md) You can find instructions for both Xilinx and Altera based FPGAs.

### FPGA-Accelerated Simulation (FireSim)

[FireSim](https://fires.im/) runs the Vortex RTL on an FPGA as a
*host-decoupled* simulation: the design is transformed so that every target
clock edge is scheduled by the host driver, and target DRAM is served by a
timing model (FASED) rather than by the raw FPGA memory. That decoupling is
what separates it from the plain FPGA flow — the results are cycle-accurate
against the RTL rather than dependent on the board's real memory timing, while
running orders of magnitude faster than Verilator.

For scale, `sgemm -n32` takes ~1228 ms on rtlsim and ~8 ms on an Alveo U55C,
and `sgemm -n1024` — over a billion instructions — completes in ~85 s.

FireSim is driven exactly like the card's native flow; only the driver name
differs. It takes the same targets as the card's native driver:

| card | targets | image built by |
|---|---|---|
| XRT (e.g. U55C) | `xrtsim` (default), `hw_emu`, `hw` | `make -C hw/syn/firesim PLATFORM=<xsa> TARGET=hw_emu\|hw` |
| aved (V80) | `avedsim` (default), `sim`, `hw` | `make -C hw/syn/firesim TARGET=sim\|hw` |

`xrtsim` and `avedsim` run the FireSim-transformed design in Verilator inside
the runtime and need no image or Vivado. The image builds take `PREFIX`,
`XLEN`, `CONFIGS` and `PERF=1` as `hw/syn/xilinx/{xrt,aved}` do, and leave the
image in `<PREFIX>_<xsa|aved>_<TARGET>/bin` together with `firesim.card`, which
tells the run which card it is for. From the build directory:

```bash
source ci/firesim_env.sh    # after the card's Xilinx tools; see the script

# no image needed
./ci/blackbox.sh --driver=firesim --target=avedsim --app=demo

# build an image, then run on it
CONFIGS="-DVX_CFG_NUM_WARPS=4 -DVX_CFG_NUM_THREADS=4" make -C hw/syn/firesim TARGET=hw PERF=1
FPGA_BIN_DIR=$PWD/hw/syn/firesim/build32_aved_hw/bin \
  ./ci/blackbox.sh --driver=firesim --target=hw --warps=4 --threads=4 --app=demo
```

On the V80, `TARGET=hw` runs `ci/v80_preflight.sh` first: it refuses to
program a card that is off the bus, unhealthy or still in use, and waits out a
cool-down since the last programming. A first `Ctrl-C` (or `SIGTERM`) stops a
FireSim run cleanly — the launch in flight is reported and completed, and the
application tears down normally — and a second one kills it. A stopped run
never reads as a pass: FireSim reports it `FAILED` and the process exits 130
(143 for `SIGTERM`, 124 when `VORTEX_FIRESIM_TIMEOUT_S` gave up).

Debugging knobs, all read from the environment by the runtime:

| variable | effect |
|---|---|
| `VORTEX_FIRESIM_TIMEOUT_S` | give up on a launch after this many seconds, with a report |
| `VORTEX_FIRESIM_STALL_CYCLES` | report (once) after this many busy target cycles; default 500000 |
| `VORTEX_FIRESIM_REPORT_ON_IDLE` | also report the counters when a launch completes |
| `VORTEX_FIRESIM_TRACE` | time-stamp every request to the simulator thread |
| `VORTEX_FIRESIM_EMUL_ARGS` | extra Verilator arguments for `xrtsim`/`avedsim`, e.g. `+verilator+rand+reset+2` |
| `FIRESIM_MM_REORDER`, `FIRESIM_MM_LATENCY` | `xrtsim`/`avedsim` host memory: reorder (1), also interleave (2) and delay reads |

The architecture, build flow and design invariants are documented in
[designs/firesim_integration.md](designs/firesim_integration.md); the
toolchain component is [section 12 of the toolchain
guide](building_toolchain.md#12-firesim).

One caveat worth knowing when reading its output: FireSim's
`Target Cycles Emulated` counts cycles the *driver stepped*, not cycles the
kernel took, and is quantized by the driver's polling schedule. Use Vortex's
own `PERF: cycles=` line, which is exact.

### Threaded FPGA Simulation (Foldsim)

Foldsim compiles Vortex into a simulator for the V80 with CIRCT: each class of
identical units (sockets, issue slices, ALUs, ...) is threaded through one
physical copy, so a GPU far larger than the card fits on it, still cycle-exact
against the RTL. Its kernel carries a small control block and a
latency-bandwidth memory model; the target runs free, and the runtime talks to
it like the native V80 driver does (AXI4-Lite registers, QDMA for memory).

| target | what runs | built by |
|---|---|---|
| `avedsim` (default) | the same kernel in Verilator, in-process | the run itself: the Foldsim compiler, then a Verilator build of the kernel, for the app's `CONFIGS` |
| `hw` | the image on the V80 | `make -C hw/syn/foldsim` |

```bash
# the classes to thread, and the units to give plain-port boundaries first
export FOLDSIM_CLASSES="VX_socket__lift VX_core__lift VX_compute_slice__lift"
export FOLDSIM_LIFT="VX_socket:drop=INSTANCE_ID VX_core:drop=INSTANCE_ID VX_compute_slice:drop=INSTANCE_ID,SLICE_STR"
CONFIGS="-DVX_CFG_NUM_CORES=2 -DVX_CFG_NUM_WARPS=4 -DVX_CFG_NUM_THREADS=4 -DVX_CFG_ISSUE_WIDTH=2"

# in Verilator, no image needed
CONFIGS="$CONFIGS" ./ci/blackbox.sh --driver=foldsim --app=sgemm

# build the image, then run on the V80
make -C hw/syn/foldsim CONFIGS="$CONFIGS"
FPGA_BIN_DIR=$PWD/hw/syn/foldsim/build32_foldsim_hw/bin \
  CONFIGS="$CONFIGS" ./ci/blackbox.sh --driver=foldsim --target=hw --app=sgemm
```

The flow needs CIRCT (`circt-verilog`, `circt-opt`, `firtool`) and Foldsim
(the `foldsim` driver and `foldsim-opt`) from the toolchain
(`./ci/toolchain_install.sh --circt --foldsim`; `CIRCT_BIN=<dir>` and
`FOLDSIM=<driver>` override them), and pyslang for the boundary lifting
(`LIFT_PYTHON`). The memory is timed as the configuration's DRAM
(`VX_CFG_PLATFORM_MEMORY_*`), a channel per memory bank, as `dram_sim` times
it; `VORTEX_DRAM_REFRESH=bank|all` adds refresh.
`VORTEX_FOLDSIM_TIMEOUT_S=<s>` fails a launch that runs longer; a command,
DCR read or reset the target leaves unanswered fails too once the target has
advanced no cycle for that long, or for 60 s when it is unset. At exit the runtime reports the target and kernel-clock cycles the simulator ran
and how busy each threaded class was. The design is
[designs/foldsim.md](designs/foldsim.md).

### How to Test (using `blackbox.sh`)

Running tests under specific drivers (rtlsim,simx,fpga) is done using the script named `blackbox.sh` located in the `ci` folder. Running command `./ci/blackbox.sh --help` from the Vortex root directory will display the following command line arguments for `blackbox.sh`:

- *Clusters* - used to specify the number of clusters (collection of processing elements) within a configuration.
- *Cores* - used to specify the number of cores (processing element containing multiple warps) within a configuration.
- *Warps* - used to specify the number of warps (collection of concurrent hardware threads) within a configuration.
- *Threads* - used to specify the number of threads (smallest unit of computation) within a configuration.
- *L2cache* - used to enable the shared l2cache among the Vortex cores.
- *L3cache* - used to enable the shared l3cache among the Vortex clusters.
- *Driver* - used to specify which driver to run the Vortex simulation (either rtlsim, opae, xrt, simx, firesim, foldsim).
- *Debug* - used to enable debug mode for the Vortex simulation.
- *Perf* - used to enable the detailed performance counters within the Vortex simulation.
- *App* - used to specify which test/benchmark to run in the Vortex simulation. The main choices are vecadd, sgemm, basic, demo, and dogfood. Other tests/benchmarks are located in the `/benchmarks/opencl` folder though not all of them work wit the current version of Vortex.
- *Args* - used to pass additional arguments to the application.

### Configuring Cache Sizes

By default, Vortex uses fixed cache sizes (L1 = 16KB, L2 = 1MB, L3 = 2MB).  
You can override these values at build time using the `CONFIGS` environment variable with `-D` flags:

| Cache Level |  Parameter | Default size | CONFIGS flag |
|-------------|-----------|---------|----------------|
| L1 Data     | `VX_CFG_DCACHE_SIZE` | 16384 | `-DVX_CFG_DCACHE_SIZE=16384` |
| L1 Instruction | `VX_CFG_ICACHE_SIZE` | 16384 | `-DVX_CFG_ICACHE_SIZE=16384` |
| L2 Cache    | `VX_CFG_L2_CACHE_SIZE` | 1048576 | `-DVX_CFG_L2_CACHE_SIZE=1048576` |
| L3 Cache    | `VX_CFG_L3_CACHE_SIZE` | 2097152 | `-DVX_CFG_L3_CACHE_SIZE=2097152` |

**Note:** All sizes must be specified in **bytes**.

**Example:** Running with custom cache sizes

```bash
$ CONFIGS="-DVX_CFG_DCACHE_SIZE=16384 -DVX_CFG_L2_CACHE_SIZE=1048576 -DVX_CFG_L3_CACHE_SIZE=2097152" \
./ci/blackbox.sh --driver=simx --clusters=1 --cores=4 --warps=4 --threads=4 --args="path-to-test-file"
```
Output from terminal:
```
open device connection
data type: float
matrix size: 32x32
allocate device memory
A_addr=0x10000
B_addr=0x11000
C_addr=0x12000
upload matrix A buffer
upload matrix B buffer
Upload kernel binary
upload kernel argument
start device
wait for completion
Elapsed time: 450 ms
download destination buffer
verify result
cleanup
PERF: core0: instrs=120689, cycles=150340, IPC=0.802774
PERF: core1: instrs=120689, cycles=150338, IPC=0.802784
PERF: core2: instrs=120689, cycles=150337, IPC=0.802790
PERF: core3: instrs=120689, cycles=150339, IPC=0.802779
PERF: instrs=482756, cycles=150340, IPC=3.211095
PASSED!
```

Example use of command line arguments: Run the sgemm benchmark using the opae driver with a Vortex configuration of 1 cluster, 4 cores, 4 warps, and 4 threads.

    $ ./ci/blackbox.sh --clusters=1 --cores=4 --warps=4 --threads=4 --driver=opae --app=sgemm

Output from terminal:
```
Create context
Create program from kernel source
Upload source buffers
Execute the kernel
Elapsed time: 2463 ms
Download destination buffer
Verify result
PASSED!
PERF: core0: instrs=90802, cycles=52776, IPC=1.720517
PERF: core1: instrs=90693, cycles=53108, IPC=1.707709
PERF: core2: instrs=90849, cycles=53107, IPC=1.710678
PERF: core3: instrs=90836, cycles=50347, IPC=1.804199
PERF: instrs=363180, cycles=53108, IPC=6.838518
```

> **Note on the `PERF:` numbers.** The instruction counts, cycle counts, and
> IPC shown above are **illustrative only** — your run will differ. They depend
> on the Vortex configuration (clusters/cores/warps/threads, cache sizes, etc.),
> the benchmark input size, and the simulator/microarchitecture revision, so the
> exact values are expected to change over time. Recent revisions also print a
> single aggregate line, e.g. `PERF: instrs=<N>, cycles=<M>, IPC=<X>`, rather
> than the per-core breakdown shown in these older examples. Use these figures
> to compare configurations relative to one another, not as fixed targets to
> reproduce.

### Hardware Profiles

A hardware profile is a named machine in `profiles/<name>.json`: a set of
`VX_CFG_*` values, grouped into `core`, `chip` and `platform` sections, with a
`sources` entry saying where each value comes from (optionally also a
`reference` block of measurements from the real part, and `extends` to build
on another profile). Every flow that includes `config.mk` builds the same
machine from it: SimX, rtlsim, the FPGA and ASIC flows, and the kernels.

| Profile | Machine |
|---|---|
| `tinygpu` | 1 core, 2 warps x 2 threads, shared memory |
| `gtx1080` | GeForce GTX 1080 (Pascal): 20 cores x 64 warps x 32 threads, 96 KB shared memory, 2 MB L2, GDDR5X as 16 GDDR6 channels; no tensor cores |
| `gtx1060` | GeForce GTX 1060 6 GB (Pascal), on `gtx1080`: 10 cores, 1.5 MB L2, 192 GB/s as 8 GDDR6 channels |
| `rtx3060` | GeForce RTX 3060 12 GB (Ampere GA106): 28 cores x 48 warps, 100 KB shared memory, 3 MB L2, GDDR6 at 360 GB/s; FP16/TF32/INT8/INT4 tensor cores with sparsity; no RTU (it needs full-warp ALU lanes, GA106 has 16 INT32 per block) |
| `rx6600` | Radeon RX 6600 (RDNA 2): 28 CUs x 40 waves x 32 lanes, 64 KB LDS, 2 MB L2, 32 MB Infinity Cache as the L3, GDDR6 at 224 GB/s; ray accelerators as RTU |
| `rx580` | Radeon RX 580 (Polaris, GCN 4): 36 CUs x 40 waves x 64 lanes on a 16-lane datapath (`SIMD_WIDTH` 16, a wave instruction every 4 clocks as on GCN), 64 KB LDS, 2 MB L2, GDDR5 at 256 GB/s as GDDR6 channels; no matrix units |
| `mi250x` | Instinct MI250X (CDNA 2): 2 dies x 110 CUs x 32 waves x 64 lanes, 64 KB LDS, 8 MB L2, HBM2E at 3.28 TB/s; INT8 matrix cores |
| `mi300x` | Instinct MI300X (CDNA 3): 8 XCDs x 38 CUs x 32 waves x 64 lanes, 64 KB LDS, 4 MB L2 per XCD, 256 MB Infinity Cache as the L3, HBM3 at 5.3 TB/s; TF32/FP8/INT8 matrix cores with sparsity |
| `rx9070xt` | Radeon RX 9070 XT (RDNA 4): 64 CUs x 32 waves x 32 lanes, 64 KB LDS, 8 MB L2, 64 MB Infinity Cache as the L3, GDDR6 at 640 GB/s; FP16/FP8/INT8/INT4 matrix units with sparsity, ray accelerators as RTU |
| `v100` | Tesla V100 (Volta): 80 cores x 64 warps, 96 KB shared memory, 6 MB L2, 32 HBM2 channels at 1752 MT/s; FP16 tensor cores |
| `a100` | A100 (Ampere): 108 cores x 64 warps, 164 KB shared memory, 192 KB L1, 40 MB L2, HBM2 at 1555 GB/s; FP16/TF32/INT8/INT4 tensor cores with 2:4 sparsity |
| `h100` | H100 SXM5 (Hopper): 132 cores x 64 warps, 228 KB shared memory, 50 MB L2, HBM3 at 3.35 TB/s; FP16/TF32/FP8/INT8 tensor cores with sparsity and wgmma, TMA as DXA |
| `rtx5080` | GeForce RTX 5080 (Blackwell): 84 cores x 48 warps, 100 KB shared memory, 64 MB L2, GDDR7 bandwidth on the GDDR6 model; FP16/TF32/FP8/FP4/MX/INT8 tensor cores with sparsity, TMA as DXA, RT cores as RTU |
| `titanv` | TITAN V (Volta), on `v100`: 4.5 MB L2, HBM2 at 653 GB/s |
| `qv100` | Quadro GV100 (Volta), on `v100`: HBM2 at 870 GB/s, 1132 MHz |
| `rtx2060` | GeForce RTX 2060 (Turing TU106): 30 cores x 32 warps, 64 KB shared memory, 96 KB L1, 3 MB L2, GDDR6 at 336 GB/s; FP16/INT8/INT4 tensor cores |
| `rtx2060s` | GeForce RTX 2060 SUPER (Turing), on `rtx2060`: 34 cores, 4 MB L2, GDDR6 at 448 GB/s |
| `t4` | Tesla T4 (Turing TU104), on `rtx2060`: 40 cores, 4 MB L2, GDDR6 at 320 GB/s |
| `rtx3070` | GeForce RTX 3070 (Ampere GA104), on `rtx3060`: 46 cores, 4 MB L2, GDDR6 at 448 GB/s |
| `h200` | H200 SXM5 (Hopper), on `h100`: HBM3e at 4.8 TB/s |
| `r9nano` | Radeon R9 Nano (Fiji, GCN 3), on `rx580`: 64 CUs, 2 MB L2, HBM at 512 GB/s |
| `rx5500xt` | Radeon RX 5500 XT (RDNA 1), on `rx6600`: 22 CUs, 1 MB L2, no Infinity Cache, no ray accelerators |
| `w6800` | Radeon Pro W6800 (RDNA 2), on `rx6600`: 60 CUs, 4 MB L2, 128 MB Infinity Cache, GDDR6 at 512 GB/s |

Every GPU profile also turns on virtual memory, the A extension with Zacas,
Zfhmin, MUFU at the part's special-function throughput, the part's INT32,
FP32 and load/store lanes per processing block (`NUM_ALU_LANES`,
`NUM_FPU_LANES`, `NUM_LSU_LANES`; a warp instruction takes `NUM_THREADS` /
lanes cycles, as on the part), and shared memory at the part's 128 B per clock
(16 banks: a Vortex bank is XLEN wide). The real parts are 64-bit machines, so these profiles set `"xlen": 64`:
`configure --profile=<name>` picks XLEN from the profile (an explicit,
conflicting `--xlen` is an error), and a `build32` tree rejects them. A profile
without an `xlen` field, like `tinygpu`, builds at either width.

Select one per build with `../configure --profile=<name>`, or per run:

```bash
$ ./ci/blackbox.sh --driver=simx --profile=rtx5080 --app=sgemm --args="-n256"
```

`CONFIGS` and blackbox's own `--clusters/--cores/--warps/--threads/--l2cache/--l3cache`
flags win over the profile: the profile only supplies the knobs they leave
unset, and everything else falls back to `VX_config.toml`.

```bash
$ CONFIGS="-DVX_CFG_L2_SIZE=8388608" ./ci/blackbox.sh --driver=simx --profile=a100 --warps=32 --app=sgemm
```

The other profiles (10 to 304 cores) are SimX-scale machines for architecture
studies, not FPGA builds; `tinygpu` is the small one, sized for FPGA
bring-up and fast simulation.

A profile's `reference` block holds the part's published or measured numbers,
and `ci/profile_check.py` compares the model against them from a build tree
of the profile's XLEN:

| Key | Checked against |
|---|---|
| `fp32_lanes`, `dram_bandwidth_gbs`, `dram_peak_bytes_per_cycle`, `clock_mhz` | the resolved configuration |
| `l1_hit_cycles`, `l2_hit_cycles`, `dram_cycles` | `tests/regression/chase` on one core: a random pointer chase at a 128 B stride (DRAM: a cold chain over a buffer never uploaded) |
| `stream_cycles` (`"<n>MiB"`) | `vecadd` with n MiB per buffer, whole chip |
| `sgemm_cycles` (`"<n>"`) | `sgemm -n<n>`, whole chip |

```bash
$ cd build64 && python3 ../ci/profile_check.py --profile=a100 --metrics=peak,latency
```

It exits 1 when a metric is off by more than `--tolerance` (10% by default),
and 2 when a run fails.

The two chase latencies are calibrated in that order. `l2_hit_cycles` sets
`VX_CFG_L2_LINK_LATENCY`, the link between the sockets and the L2 (cycles
each way, so a hit gains twice it): on the parts most of an L2 hit is the
SM-to-L2 network. A DRAM access crosses the link too, so
`VX_CFG_PLATFORM_MEMORY_LATENCY` then takes what `dram_cycles` still needs.
Both latencies move the measured ones one for one (the link two for one).

### Configuring the DRAM Model

The simulators (simx, rtlsim, opaesim, xrtsim) all model the device memory
the same way: one independent DRAM channel (controller and device) per
platform memory bank, timed by
[Ramulator](https://github.com/CMU-SAFARI/ramulator2) with its JEDEC device
models. Each channel maps addresses the way GPU memory controllers do: four
bursts per row visit, then the next bank, with the bank permuted by the row so
strided streams spread over the banks. Refresh is not modelled: GPU
controllers hide it with per-bank refresh, for which Ramulator has no manager. A channel is described the way a data sheet describes it; these
`VX_config.toml` keys select it, overridable through `CONFIGS` or a profile,
and the RTL never reads them:

| Key | Default | Meaning |
|---|---|---|
| `VX_CFG_PLATFORM_MEMORY_TYPE` | `HBM2` | JEDEC standard: `HBM2` (also HBM2E), `HBM3` or `GDDR6` (x16 channels) |
| `VX_CFG_PLATFORM_MEMORY_NUM_BANKS` | 2 | memory banks, and so DRAM channels (HBM: 128-bit channels) |
| `VX_CFG_PLATFORM_MEMORY_DATA_RATE` | 2 x `PLATFORM_CLOCK_RATE` | transfer rate per pin in MT/s; the default clocks the DRAM with the core |
| `VX_CFG_PLATFORM_CLOCK_RATE` | 400 | core clock in MHz; with the data rate it sets how many DRAM cycles a core cycle spans |
| `VX_CFG_PLATFORM_MEMORY_LATENCY` | 0 | fixed read latency added for the PHY, controller and interconnect, in core cycles |
| `VX_CFG_PLATFORM_MEMORY_QUEUE_SIZE` | 0 | requests a channel holds in flight (0 = unbounded) |

The model's peak bandwidth follows from the keys: channels x channel width x
data rate (HBM 16 B, GDDR6 2 B per transfer); `VX_CFG_PLATFORM_MEMORY_PEAK_BW`
separately states the platform's, which the runtime reports. The device's
array timings are held in nanoseconds, so a different data rate changes the
bandwidth but not the latency. The default machine clocks the DRAM with the
core: an idle read takes 14 cycles on an open row and 26 on a row conflict,
and the memory delivers up to one 64 B block per cycle.

```bash
$ CONFIGS="-DVX_CFG_PLATFORM_MEMORY_TYPE=GDDR6 -DVX_CFG_PLATFORM_MEMORY_NUM_BANKS=16 -DVX_CFG_PLATFORM_MEMORY_DATA_RATE=14000" \
  ./ci/blackbox.sh --driver=simx --app=vecadd
```

## Additional Quick Start Scenarios

Running Vortex simulators with different configurations and drivers is supported. For example:

- Run basic driver test with rtlsim driver and Vortex config of 2 clusters, 2 cores, 2 warps, 4 threads

    $ ./ci/blackbox.sh --driver=rtlsim --clusters=2 --cores=2 --warps=2 --threads=4  --app=basic

- Run demo driver test with opae driver and Vortex config of 1 clusters, 4 cores, 4 warps, 2 threads

    $ ./ci/blackbox.sh --driver=opae --clusters=1 --cores=4 --warps=4 --threads=2 --app=demo

- Run dogfood driver test with simx driver and Vortex config of 4 cluster, 4 cores, 8 warps, 6 threads

    $ ./ci/blackbox.sh --driver=simx --clusters=4 --cores=4 --warps=8 --threads=6  --app=dogfood