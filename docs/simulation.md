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

```bash
export TARGET=hw
export FPGA_BIN_DIR=<directory containing firesim.xclbin>
./ci/blackbox.sh --driver=firesim --app=sgemm --args="-n1024"
```

The architecture, build flow and design invariants are documented in
[designs/firesim_integration.md](designs/firesim_integration.md); the
toolchain component is [section 12 of the toolchain
guide](building_toolchain.md#12-firesim).

One caveat worth knowing when reading its output: FireSim's
`Target Cycles Emulated` counts cycles the *driver stepped*, not cycles the
kernel took, and is quantized by the driver's polling schedule. Use Vortex's
own `PERF: cycles=` line, which is exact.

### How to Test (using `blackbox.sh`)

Running tests under specific drivers (rtlsim,simx,fpga) is done using the script named `blackbox.sh` located in the `ci` folder. Running command `./ci/blackbox.sh --help` from the Vortex root directory will display the following command line arguments for `blackbox.sh`:

- *Clusters* - used to specify the number of clusters (collection of processing elements) within a configuration.
- *Cores* - used to specify the number of cores (processing element containing multiple warps) within a configuration.
- *Warps* - used to specify the number of warps (collection of concurrent hardware threads) within a configuration.
- *Threads* - used to specify the number of threads (smallest unit of computation) within a configuration.
- *L2cache* - used to enable the shared l2cache among the Vortex cores.
- *L3cache* - used to enable the shared l3cache among the Vortex clusters.
- *Driver* - used to specify which driver to run the Vortex simulation (either rtlsim, opae, xrt, simx, firesim).
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
| `rtx1080` | GeForce GTX 1080 (Pascal): 20 cores x 64 warps x 32 threads, 2 MB L2, GDDR5X as 16 GDDR6 channels |
| `v100` | Tesla V100 (Volta): 80 cores, 4 MB L2, 32 HBM2 channels at 877 MHz |
| `a100` | A100 (Ampere): 108 cores, 32 MB L2, GPGPU-Sim's A100 HBM timing |
| `h100` | H100 SXM5 (Hopper): 132 cores, 32 MB L2, HBM3 bandwidth on the HBM2 model |
| `rtx5080` | GeForce RTX 5080 (Blackwell): 84 cores x 48 warps, 64 MB L2, GDDR7 bandwidth on the GDDR6 model |

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

The other profiles (20 to 132 cores) are SimX-scale machines for architecture
studies, not FPGA builds; `tinygpu` is the small one, sized for FPGA
bring-up and fast simulation.

### Configuring the DRAM Model

The simulators (simx, rtlsim, opaesim, xrtsim) all model the device memory
the same way: one independent DRAM channel per platform memory bank, timed by
[Ramulator](https://github.com/CMU-SAFARI/ramulator2), with its refresh. These
`VX_config.toml` keys select it, overridable through `CONFIGS` or a profile:

| Key | Default | Meaning |
|---|---|---|
| `VX_CFG_PLATFORM_MEMORY_TYPE` | `HBM2` | `HBM2` (JEDEC 2 Gbps), `HBM2_A100` (GPGPU-Sim's A100 timing) or `GDDR6` (x16 channels, 32 B bursts) |
| `VX_CFG_PLATFORM_MEMORY_NUM_BANKS` | 2 | memory banks, and so DRAM channels |
| `VX_CFG_PLATFORM_MEMORY_CLOCK_RATE` | 1000 | DRAM clock in MHz (GDDR6: the data clock, half the transfer rate) |
| `VX_CFG_PLATFORM_CLOCK_RATE` | 400 | core clock in MHz; with the DRAM clock it sets how many DRAM cycles a core cycle spans |
| `VX_CFG_PLATFORM_MEMORY_LATENCY` | 0 | fixed read latency added for the PHY, controller and interconnect, in core cycles |
| `VX_CFG_PLATFORM_MEMORY_WINDOW` | 0 | requests a channel holds in flight (0 = unbounded) |

```bash
$ CONFIGS="-DVX_CFG_PLATFORM_MEMORY_TYPE=GDDR6 -DVX_CFG_PLATFORM_MEMORY_NUM_BANKS=16 -DVX_CFG_PLATFORM_MEMORY_CLOCK_RATE=8000" \
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