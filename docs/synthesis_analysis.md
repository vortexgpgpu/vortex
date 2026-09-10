# Synthesis and Power Analysis

This guide explains how to run synthesis, timing analysis, area analysis, and power analysis for Vortex across all supported back-ends: Xilinx (Vivado), Altera (Quartus), Yosys (open-source), and Synopsys Design Compiler.

---

## Table of Contents

- [Design Configuration](#design-configuration)
- [Generating SAIF Files](#generating-saif-files)
- [Specifying SAIF_INST](#specifying-saif_inst)
- [DUT Sub-Component Evaluation](#dut-sub-component-evaluation)
- [Xilinx (Vivado)](#xilinx-vivado)
- [Altera (Quartus)](#altera-quartus)
- [Yosys (Open-Source)](#yosys-open-source)
- [Synopsys Design Compiler](#synopsys-design-compiler)
- [Understanding Power Reports](#understanding-power-reports)

---

## Design Configuration

All synthesis flows accept a `CONFIGS` variable to customize the hardware design at build time. `CONFIGS` is a string of preprocessor macro definitions (`-D` flags) that control core count, cache hierarchy, extensions, and other parameters.

Common configuration flags (all parameters live in the `VX_CFG_*` namespace; see `VX_config.toml` at the repo root for the full list):

| Flag | Description |
|------|-------------|
| `-DVX_CFG_NUM_CLUSTERS=N` | Number of clusters |
| `-DVX_CFG_NUM_CORES=N` | Number of cores per cluster |
| `-DVX_CFG_NUM_WARPS=N` | Number of warps per core |
| `-DVX_CFG_NUM_THREADS=N` | Number of threads per warp |
| `-DVX_CFG_L2_ENABLE` | Enable shared L2 cache |
| `-DVX_CFG_L3_ENABLE` | Enable shared L3 cache |
| `-DVX_CFG_EXT_TCU_ENABLE` | Enable Tensor Core Unit |
| `-DVX_CFG_EXT_DXA_ENABLE` | Enable DXA extension |
| `-DVX_CFG_DCACHE_SIZE=N` | Set data cache size in bytes |

Example:

```bash
CONFIGS="-DVX_CFG_NUM_CLUSTERS=1 -DVX_CFG_NUM_CORES=4 -DVX_CFG_L2_ENABLE -DVX_CFG_EXT_TCU_ENABLE"
```

### Overriding top-module parameters (`-G`)

`CONFIGS` may also carry Verilog **parameter overrides** as `-G<NAME>=<value>` alongside the `-D` macros. `gen_sources.sh` forwards these to `repl_params.py`, which rewrites the parameter's default in the top module's *per-build copy* (the source tree is untouched, so concurrent builds with different overrides stay isolated). `gen_config` ignores `-G` tokens, so no separate Makefile variable is needed.

This is mainly for the DUT unittest wrappers, whose knobs are Verilog parameters rather than `VX_CFG_*` macros — e.g. the cache wrapper's `AMO_ENABLE` and `IS_LLC`:

```bash
# LLC cache DUT, AMO disabled vs enabled (NT=NW=32)
CONFIGS="-DVX_CFG_NUM_THREADS=32 -DVX_CFG_NUM_WARPS=32 -GAMO_ENABLE=0 -GIS_LLC=1" PREFIX=amo0 make cache
CONFIGS="-DVX_CFG_NUM_THREADS=32 -DVX_CFG_NUM_WARPS=32 -GAMO_ENABLE=1 -GIS_LLC=1" PREFIX=amo1 make cache
```

All flows also support the `NUM_CORES` Makefile shorthand which auto-selects a pre-defined cluster/core/L2 configuration:

```bash
NUM_CORES=4   # equivalent to -DVX_CFG_NUM_CLUSTERS=1 -DVX_CFG_NUM_CORES=4 -DVX_CFG_L2_ENABLE
NUM_CORES=16  # equivalent to -DVX_CFG_NUM_CLUSTERS=1 -DVX_CFG_NUM_CORES=16 -DVX_CFG_L2_ENABLE
NUM_CORES=32  # equivalent to -DVX_CFG_NUM_CLUSTERS=2 -DVX_CFG_NUM_CORES=16 -DVX_CFG_L2_ENABLE
```

### Using PREFIX for Isolated Builds

Use `PREFIX=<unique_build_dir>` to keep builds separate. Each flow creates a build directory derived from `PREFIX` so that multiple configurations can coexist without overwriting each other:

```bash
# Xilinx XRT: creates build_4c_<platform>_<target>/
PREFIX=build_4c NUM_CORES=4 make -C hw/syn/xilinx/xrt

# Synopsys: creates my_test_Vortex/
PREFIX=my_test make -C hw/syn/synopsys synthesis
```

### The `tinygpu` Configuration for Fast Turnaround

When the component under investigation is *outside* the GPU core — the AFU shell, the command processor, the runtime, or the FPGA driver — a full-size Vortex build is wasted synthesis time. Most of the hours go into cores and caches that the bug does not live in, and every debug iteration pays for them again.

For those cases, use a deliberately minimal GPU: one core, two warps of two threads, all L1 caches off, and shared memory off.

```bash
CONFIGS="-DVX_CFG_NUM_CLUSTERS=1 -DVX_CFG_NUM_CORES=1 \
         -DVX_CFG_NUM_WARPS=2 -DVX_CFG_NUM_THREADS=2 \
         -DVX_CFG_ICACHE_DISABLE -DVX_CFG_DCACHE_DISABLE -DVX_CFG_LMEM_DISABLE"
```

Note the `_DISABLE` spelling: knobs that default to `true` in `VX_config.toml` are turned off by defining their `_DISABLE` guard, not by assigning `=0`.

This strips the design down to a single pipeline talking straight to the memory interface. Synthesis and place-and-route finish in a small fraction of the time a production configuration takes, and the resulting bitstream still exercises the complete external path: host to driver, driver to shell, shell to command processor, command processor to core, and the memory traffic back out. Timing closure is generally uneventful at this size, so a failure to close is itself a signal that the problem is in the surrounding logic rather than in core density.

Pair it with the `sgemm` benchmark. `sgemm` is well understood, self-checking, and touches every part of the external path — kernel launch, argument passing, bulk DMA in both directions, and completion signalling — while staying small enough to run quickly. A `sgemm` failure on `tinygpu` isolates the defect to the surrounding infrastructure, because the core configuration is too small to be hiding a microarchitectural corner case. Conversely, `sgemm` passing on `tinygpu` but failing at full size points back at the core, cache hierarchy, or a concurrency effect that only appears with more warps in flight.

Treat this as the first move when debugging external components, not a fallback after a long build fails. Once `tinygpu` is green, scale back up to the target configuration to confirm the fix under real conditions.

---

## Generating SAIF Files

SAIF (Switching Activity Interchange Format) files capture signal toggle rates during simulation and are used to produce accurate power estimates. Vortex supports SAIF generation through its RTL simulators: `rtlsim`, `opaesim`, and `xrtsim`.

### Method 1: Build the Simulator Directly

Build the simulator with SAIF tracing enabled, then run a workload:

```bash
# Build rtlsim with SAIF support
make -C sim/rtlsim SAIF=1

# Run a test application
make -C tests/regression/sgemm run-rtlsim
```

The SAIF file is written to `trace.saif` in the application directory.

### Method 2: Use the Blackbox Test Driver

The `ci/blackbox.sh` script provides a convenient wrapper:

```bash
./ci/blackbox.sh --driver=rtlsim --app=sgemm --cores=4 --l2cache --saif
```

When `--saif` is passed, blackbox.sh:
1. Builds the simulator with `SAIF=1`
2. Runs the application
3. Copies the resulting `trace.saif` to the current directory

`--saif` composes with `--debug` on every RTL driver, so a run can emit both the
`run.log` trace and the SAIF. It cannot be combined with `--vcd`: a model emits one
waveform format or the other.

Available drivers for SAIF generation:

| Driver | Simulator | Use Case |
|--------|-----------|----------|
| `rtlsim` | Verilator RTL sim | General-purpose RTL power analysis |
| `opaesim` | OPAE AFU simulator | Intel/Altera platform-specific analysis |
| `xrtsim` | XRT simulator | Xilinx platform-specific analysis |

---

## Specifying SAIF_INST

When reading a SAIF file, the tool must strip the testbench hierarchy prefix from signal names so they align with the synthesized netlist. `SAIF_INST` specifies this prefix.

Typical values:

| Flow | SAIF_INST |
|------|-----------|
| Xilinx DUT | `TOP.rtlsim_shim.vortex` |
| Xilinx XRT | `TOP.vortex_afu_shim.vortex_afu` |
| Synopsys / Yosys | Instance path matching your simulation hierarchy |

The path does not have to be absolute. A module instance name works if the tool can resolve it unambiguously (find-first semantics).

If the SAIF root scope already matches the top module, leave `SAIF_INST` empty.

---

## DUT Sub-Component Evaluation

Both Xilinx and Altera provide DUT (Device Under Test) flows for synthesizing and analyzing sub-components in isolation, without the full platform wrapper. This is useful for evaluating individual units such as the TCU, FPU, cache, or a single core.

### Xilinx DUT Targets

Located in `hw/syn/xilinx/dut/`. Available sub-component targets:

| Target | Module | Description |
|--------|--------|-------------|
| `unittest` | Unit tests | Basic block tests |
| `scope` | Scope analyzer | Debug scope |
| `mem_unit` | Memory unit | Memory subsystem |
| `lmem` | Local memory | Local/shared memory |
| `cache` | Cache | Cache subsystem |
| `fpu` | FPU | Floating-point unit |
| `tcu` | TCU | Tensor Core Unit |
| `dxa` | DXA | DXA extension |
| `core` | Core | Single core |
| `issue` | Issue unit | Instruction issue |
| `vortex` | Vortex | Full processor (no AFU wrapper) |
| `top` | Top | Full design with AFU |

```bash
cd hw/syn/xilinx/dut

# Synthesize the TCU in isolation
CONFIGS="-DVX_CFG_EXT_TCU_ENABLE" make tcu

# Synthesize a 4-core Vortex without the platform wrapper
CONFIGS="-DVX_CFG_NUM_CORES=4 -DVX_CFG_L2_ENABLE" make vortex

# Run power analysis on an existing tcu
make tcu-power SAIF_FILE=/path/to/trace.saif SAIF_INST=*.tensor_unit
```

Each target creates its build under `<target>/<BUILD_DIR>/` (e.g., `tcu/build/`).

### Altera DUT Targets

Located in `hw/syn/altera/dut/`. Same set of sub-component targets as Xilinx. Requires the `DEVICE_FAMILY` variable and IP cache generation:

```bash
cd hw/syn/altera/dut

# Generate IP cache first (required for fpu, vortex, top)
make ip-gen

# Synthesize TCU for Arria 10
DEVICE_FAMILY=arria10 CONFIGS="-DVX_CFG_EXT_TCU_ENABLE" make tcu

# Synthesize a single core for Stratix 10
DEVICE_FAMILY=stratix10 make core
```

Build directories include the device family: `<target>/build_<device_family>/`.

---

## Xilinx (Vivado)

### XRT Full-Platform Flow

Located in `hw/syn/xilinx/xrt/`. Builds a complete Vitis xclbin for deployment on Xilinx FPGAs.

Supported platforms: Alveo U50, U55C, U200, U250, U280, Versal VCK5000.

#### Running Synthesis

```bash
cd hw/syn/xilinx/xrt

# Build a 4-core design for U280
PREFIX=build_4c NUM_CORES=4 TARGET=hw \
  PLATFORM=xilinx_u280_gen3x16_xdma_1_202310_1 \
  CONFIGS="-DVX_CFG_L2_ENABLE -DVX_CFG_DCACHE_SIZE=8192" \
  make > build.log 2>&1 &
```

Key variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `PREFIX` | `build$(XLEN)` | Build directory prefix |
| `TARGET` | `hw` | `hw` for hardware, `hw_emu` for emulation |
| `PLATFORM` | (required) | Xilinx platform identifier |
| `NUM_CORES` | - | Shorthand for core configuration |
| `CONFIGS` | - | Additional design macros |
| `MAX_JOBS` | 8 | Parallel Vivado jobs |

#### Power Analysis

```bash
make power SAIF_FILE=/path/to/trace.saif SAIF_INST=TOP.vortex_afu_shim.vortex_afu BUILD_DIR=<build_dir>
```

The script (`hw/scripts/xilinx_power_analysis.tcl`) resolves the post-implementation checkpoint from `BUILD_DIR` automatically.

#### Where to Find Reports

**XRT flow** (under `<BUILD_DIR>/`):

| Report | Location | Content |
|--------|----------|---------|
| Utilization | `<BUILD_DIR>/bin/utilization.rpt` | LUTs, FFs, BRAM, DSP |
| Timing | `<BUILD_DIR>/bin/timing.rpt` | Worst setup paths |
| Power (vectorless) | `power_vectorless.rpt` | Baseline power estimate |
| Power (SAIF) | `power_saif.rpt` | Activity-annotated power |

**DUT flow** (under `<target>/<BUILD_DIR>/`):

| Report | Location | Content |
|--------|----------|---------|
| Post-synth utilization | `post_synth_util.rpt` | Hierarchical resource usage |
| Post-impl utilization | `post_impl_util.rpt` | Hierarchical resource usage after P&R |
| Timing | `timing.rpt` | 100 worst setup paths |
| Methodology | `methodology.rpt` | Design rule checks |
| Clock utilization | `clock_utilization.rpt` | Clock tree and register usage |
| RAM utilization | `ram_utilization.rpt` | Detailed RAM/BRAM usage |
| Power (vectorless) | `power_vectorless.rpt` | Baseline power |
| Power (VCD) | `power_vcd.rpt` | VCD-annotated power (if VCD_FILE set) |
| Power (SAIF) | `power_saif.rpt` | SAIF-annotated power (via `make power`) |
| DRC | `drc.rpt` | Design rule violations |
| High fanout nets | `high_fanout_nets.rpt` | Nets with >100 fanout |

#### Finding Key Metrics

- **Fmax**: Look in `timing.rpt` for the worst negative slack (WNS). Fmax = 1 / (clock_period - WNS).
- **Total LUTs**: In `post_impl_util.rpt`, find the row for `CLB LUTs` or `Slice LUTs`.
- **Total DSPs**: In `post_impl_util.rpt`, find the row for `DSPs` or `DSP48E2`.
- **Total BRAM**: In `post_impl_util.rpt`, find the row for `Block RAM Tile` or `RAMB36/RAMB18`.

---

## Altera (Quartus)

### OPAE Full-Platform Flow

Located in `hw/syn/altera/opae/`. Builds AFU images for Intel OPAE platforms (Arria 10, Stratix 10).

#### Running Synthesis

```bash
cd hw/syn/altera/opae

# Full build: IP generation, setup, and synthesis
DEVICE_FAMILY=arria10 PREFIX=build_4c NUM_CORES=4 TARGET=fpga make

# For ASE simulation build
DEVICE_FAMILY=stratix10 TARGET=asesim make
```

Key variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `DEVICE_FAMILY` | `arria10` | `arria10` or `stratix10` |
| `PREFIX` | `build$(XLEN)` | Build directory prefix |
| `TARGET` | `fpga` | `fpga`, `asesim`, or `ase` |
| `NUM_CORES` | - | Shorthand for core configuration |
| `CONFIGS` | - | Additional design macros |

Build directory: `<PREFIX>_<device_family>_<target>_<num_cores>c/`.

#### Power Analysis

Altera power analysis uses Quartus PowerPlay with VCD-based toggle annotation:

```bash
# Located in hw/syn/altera/power_play.sh
quartus_pow --input_vcd=trace.vcd \
  --vcd_filter_glitches=on \
  --default_input_io_toggle_rate=10000transitions/s \
  $ProjectName
```

#### Where to Find Reports

Reports are generated in the synthesis build directory by the Quartus report scripts.

**Area reports** (from `report_area.tcl`):

| Report | Content |
|--------|---------|
| `*.syn.area.resource_summary.csv` | Synthesis resource summary |
| `*.syn.area.resource_breakdown.csv` | Resource breakdown by entity |
| `*.syn.area.ram_summary.csv` | Synthesis RAM summary |
| `*.syn.area.stats.csv` | Post-synthesis netlist statistics |
| `*.fit.area.resource_summary.csv` | Fitter resource summary (post-P&R) |
| `*.fit.area.resource_breakdown.csv` | Fitter resource breakdown by entity |
| `*.fit.area.ram_summary.csv` | Fitter RAM summary |
| `*.fit.area.routing_summary.csv` | Routing utilization |
| `*.fit.area.routing_global.csv` | Global signal routing |
| `*.fit.area.routing_high_fanout.csv` | High fanout signal routing |

**Timing reports** (from `analyze_timing.tcl`):

| Report | Content |
|--------|---------|
| `*.fit.timing.summary.txt` | Summary with Fmax, setup/hold, clock summary |
| `*.fit.timing.setup.html` | Top 200 setup violation paths (with routing) |
| `*.fit.timing.hold.html` | Top 200 hold violation paths |
| `*.fit.timing.recovery.html` | Recovery timing paths |
| `*.fit.timing.removal.html` | Removal timing paths |
| `*.fit.timing.check_errors.html` | Timing DRC (no clock, multiple clock, loops) |
| `*.fit.timing.check_metastability.html` | Metastability report |
| `*.fit.timing_histogram.*.setup.html` | Per-clock setup slack histograms |
| `*.fit.timing.setup.bottlenecks.txt` | Bottleneck analysis (TNS, fanout, fanin) |
| `*.fit.timing.summary.fmax.csv` | Fmax summary (CSV) |
| `*.fit.timing.summary.setup.csv` | Setup summary (CSV) |
| `*.fit.timing.summary.hold.csv` | Hold summary (CSV) |
| `*.fit.timing.summary.multicorner.csv` | Multi-corner timing summary |

#### Finding Key Metrics

- **Fmax**: Open `*.fit.timing.summary.txt` or `*.fit.timing.summary.fmax.csv`. The Fmax summary reports the restricted Fmax for each clock domain.
- **Total ALMs/LUTs**: In `*.fit.area.resource_summary.csv`, look for `ALMs needed` (Stratix 10) or `Logic utilization` (Arria 10).
- **Total DSPs**: In `*.fit.area.resource_summary.csv`, look for the `DSP` row.
- **Total BRAM (M20K/M10K)**: In `*.fit.area.resource_summary.csv`, look for `M20K blocks` or `M10K blocks`. Also see `*.fit.area.ram_summary.csv` for detailed RAM usage by entity.

---

## Yosys (Open-Source)

Located in `hw/syn/yosys/`. Uses Yosys/ABC for technology mapping and OpenSTA for pre-layout timing and power analysis. ASAP7 7.5-track v28 is the default PDK; NanGate45 remains available as a legacy option.

### Running Synthesis

```bash
# First use installs the selected, pinned ASAP7 Liberty files automatically.
# Synthesis only (generic gates)
make -C build/hw/syn/yosys PREFIX=test NUM_CORES=1 synthesis

# Explicit legacy library selection
make -C build/hw/syn/yosys PREFIX=test NUM_CORES=1 PDK=nangate45 timing

# Synthesis + technology mapping
make -C build/hw/syn/yosys PREFIX=test NUM_CORES=1 techmap

# Full flow: synthesis + mapping + STA + power
make -C build/hw/syn/yosys PREFIX=test NUM_CORES=1 SAIF_FILE=/path/to/trace.saif SAIF_INST=<inst> timing
```

Key variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `PREFIX` | `build` | Build directory prefix |
| `TOP_LEVEL_ENTITY` | `Vortex` | Top module name |
| `NUM_CORES` | - | Shorthand for core configuration |
| `CONFIGS` | - | Additional design macros |
| `CLOCK_FREQ` | 800 | Target clock frequency in MHz |
| `DELAY_UNC` | 0.02 | Clock uncertainty (fraction of period) |
| `DELAY_IO` | 0.05 | I/O delay (fraction of period) |
| `ABC_DRIVER_CELL` | ASAP7 RVT: `BUFx4_ASAP7_75t_R` | Input driver cell used by ABC for buffering and sizing |
| `ABC_LOAD` | ASAP7: `5.0` | ABC primary-output load in fF |
| `YOSYS_FLATTEN` | 1 | Flatten hierarchy before technology mapping for better cross-module optimization |
| `YOSYS_SHARE` | 1 | Enable Yosys resource sharing before mapping; set to 0 to disable |
| `PDK` | `asap7` | `asap7` or the legacy `nangate45` library |
| `ASAP7_VT` | `rvt` | ASAP7 threshold-voltage library: `rvt` or `lvt` |
| `CORNER` | `tt` | ASAP7 `tt`, `ss`, or `ff` NLDM corner |
| `LIB_TGT` | selected by `PDK` | Explicit custom Liberty override |
| `SAIF_FILE` | - | SAIF file for power annotation |
| `SAIF_INST` | top module | OpenSTA SAIF scope override |
| `DUT_FILELIST` | - | VCS-style filelist for an arbitrary DUT |
| `SDC_FILE` | `project.sdc` | Clock, reset, and I/O constraints |

Example synthesizing a TFR FEDP unit with FP8 and MX enabled with ASAP7 RVT at the default 800 MHz:
```bash
make -C build/hw/syn/yosys timing TOP_LEVEL_ENTITY=VX_tcu_fedp_tfr PREFIX=tfr_fedp CORNER=tt EXTRA_CONFIGS='-DVX_CFG_EXT_TCU_ENABLE -DVX_CFG_TCU_TYPE_TFR -DVX_CFG_NUM_THREADS=32 -DVX_CFG_TCU_FEDP_FP8_ENABLE -DVX_CFG_TCU_MX_ENABLE'
```
Output reports are written to `build/hw/syn/yosys/tfr_fedp_VX_tcu_fedp_tfr/reports/`

Build directory: `<PREFIX>_<TOP_LEVEL_ENTITY>/` (e.g., `test_Vortex/`).

The flow uses `sv2v` to convert SystemVerilog sources to Verilog before feeding them to Yosys.

#### Per-DUT synthesis

The DUTs above are declared once, in `hw/syn/yosys/dut/catalog.mk` — a top module, an include path and a define set each — and driven through one dispatcher, so a hand run and the `asic_gate` (below) build the same thing:

```bash
make -C build/hw/syn/yosys/dut list                 # what is available
make -C build/hw/syn/yosys/dut om                   # synthesis + STA (TARGET=timing)
make -C build/hw/syn/yosys/dut tcu TARGET=synthesis # synthesis only
make -C build/hw/syn/yosys/dut om CLOCK_FREQ=500    # override the target clock
```

`PREFIX` defaults to the DUT name, so each DUT gets its own tree. That is not cosmetic: the flow caches `$(BUILD_DIR)/src` and does **not** regenerate it when `EXTRA_INCLUDE` changes, so two DUTs sharing one tree silently synthesize the first one's sources. Override `PREFIX` to keep an automated sweep away from a hand-run build of the same DUT.

#### Synthesis-regression gate (`asic_gate`)

`ci/asic_gate.py` runs that catalog against checked-in goldens in `ci/baselines/synthesis/yosys/` and fails on a Fmax or cell-area move beyond ±5%. It is the ASIC sibling of `ci/fpga_gate.py` and shares its implementation (`ci/synth_gate.py`).

```bash
ci/asic_gate.py --list             # builds and their recorded baselines
ci/asic_gate.py -b om -b tex       # gate two builds
ci/asic_gate.py --update-baseline  # re-record (human-reviewed, never in CI)
```

Two things about Fmax on this flow are worth knowing before reading a number. ABC maps to the *target* period and stops, so a design that closes does so with picoseconds of margin and its Fmax sits just above `CLOCK_FREQ` by construction — **cell area is the sensitive metric**, and Fmax is mostly a met/missed signal. And `report_wns` is worst *negative* slack, clamped at zero: `run_sta.tcl` uses `report_worst_slack` (signed) for exactly this reason, which is what `worst_slack.rpt` holds.

See [docs/designs/continuous_integration.md](designs/continuous_integration.md) §3.5 and §4.5.

The first ASAP7 invocation runs the generated `build/hw/syn/libs/asap7/install.sh` installer. It downloads and SHA-256 verifies only the selected VT's five logical groups (`INVBUF`, `SIMPLE`, `AO`, `OA`, and `SEQ`) at TT, SS, and FF, then writes the merged Liberty files to `build/hw/syn/libs/asap7/lib/`. Gate-level simulation additionally installs the selected functional Verilog models in `build/hw/syn/libs/asap7/verilog/`. Later invocations verify the installed files and skip the download and preparation. The pin manifest, installer, preparation script, and upstream license are versioned in `hw/syn/libs/asap7/`, alongside the other standard-cell collateral; `configure` instantiates only `install.sh.in` into the build tree, and it reads the manifest and preparation script from the source tree in place. No ASAP7 collateral is kept under `third_party/`, and the flow requires no OpenROAD executable or physical collateral.

### Threshold-Voltage Selection

RVT is the default because it is the appropriate general-purpose implementation library. Use LVT to establish a timing-focused bound or when a design does not meet its target with RVT:

```bash
make -C build/hw/syn/yosys timing ASAP7_VT=lvt CLOCK_FREQ=1000
```

### Arbitrary DUT and Gate-Level SAIF

An arbitrary Verilog DUT can bypass Vortex source generation with its own top, filelist, and SDC. Filelists may contain RTL files, `-f`, `+incdir+`, and `+define+` entries.

```bash
make -C build/hw/syn/yosys timing TOP_LEVEL_ENTITY=my_dut \
  DUT_FILELIST=/absolute/path/to/dut.f \
  SDC_FILE=/absolute/path/to/dut.sdc PREFIX=my_dut
```

For gate-level simulation, provide a Verilator-compatible testbench filelist. The testbench must instantiate the mapped DUT (the default instance name is `dut`) and call `$dumpfile("gate_raw.saif")` plus `$dumpvars`. `SIM_ARGS`, `GATE_SIM_FLAGS`, `TB_TOP`, and `GATE_DUT_INSTANCE` are available for testbench-specific needs.

```bash
make -C build/hw/syn/yosys gate-saif TOP_LEVEL_ENTITY=my_dut \
  DUT_FILELIST=/absolute/path/to/dut.f \
  SDC_FILE=/absolute/path/to/dut.sdc \
  TB_FILELIST=/absolute/path/to/tb.f SIM_ARGS="+seed=1"

make -C build/hw/syn/yosys timing TOP_LEVEL_ENTITY=my_dut \
  DUT_FILELIST=/absolute/path/to/dut.f \
  SDC_FILE=/absolute/path/to/dut.sdc \
  SAIF_FILE=$PWD/build_my_dut/gate.saif
```

The gate simulation traces Yosys-generated underscore nets and retains the complete mapped-cell hierarchy before re-rooting the DUT SAIF scope. OpenSTA writes annotated and unannotated pin reports and treats a requested SAIF with zero coverage as an error.

### SRAM Area Estimation

Yosys uses blackbox modules (`VX_dp_ram_asic`, `VX_sp_ram_asic`) for SRAM. The `sram_cost.py` script estimates SRAM area from the Yosys JSON netlist by inferring width and depth from port connectivity:

```
Area = (width x depth x SRAM_BIT_AREA) + SRAM_OVERHEAD
```

Defaults: `SRAM_BIT_AREA=0.1` um^2/bit, `SRAM_OVERHEAD=100.0` um^2. These can be overridden via environment variables.

### Where to Find Reports

All reports are under `<BUILD_DIR>/reports/`:

| Report | Content |
|--------|---------|
| `yosys.log` | Full Yosys synthesis log |
| `stat_lib.rpt` | Cell count and area (post-mapping, by liberty cell type) |
| `sram_area.rpt` | Estimated SRAM area breakdown |
| `sta.log` | OpenSTA timing log |
| `setup.rpt` / `hold.rpt` | Detailed setup and hold paths |
| `wns.rpt` / `tns.rpt` | Worst and total negative slack (clamped at 0 when timing closes) |
| `worst_slack.rpt` | Worst slack, **signed** — positive when the design closes with margin |
| `synth_summary.csv` | One-row machine-readable summary of all of the above (`synth_summary.py`) |
| `power.rpt` | Power estimate (vectorless or SAIF-annotated) |
| `power_hier.rpt` | Hierarchical power breakdown |
| `saif_annotated.rpt` | Pins covered by SAIF |
| `saif_unannotated.rpt` | Signals not covered by SAIF |

Netlists are written to `<BUILD_DIR>/out/`:

| File | Content |
|------|---------|
| `<TOP>_syn.v` | Post-synthesis generic netlist |
| `<TOP>_mapped.v` | Post-mapping technology netlist |
| `<TOP>.json` | Yosys JSON netlist (used by `sram_cost.py`) |

### Finding Key Metrics

- **Total area**: In `stat_lib.rpt`, look for `Chip area for top module`. Add the estimated SRAM area from `sram_area.rpt` for the total.
- **Cell count**: In `stat_lib.rpt`, the per-cell-type breakdown shows gate counts.
- **Timing (WNS/TNS)**: In `sta.log`, look for the `report_wns` and `report_tns` outputs. Fmax = 1 / (target_period - WNS).
- **Power**: In `power.rpt`, look for total power. `power_hier.rpt` breaks it down by hierarchy.

---

## Synopsys Design Compiler

Located in `hw/syn/synopsys/`. Uses Synopsys DC for ASIC synthesis with support for multiple technology libraries.

### Supported Libraries

| LIB_TYPE | Technology | Path |
|----------|-----------|------|
| `DEFAULT` | NanGate 15nm OCL | Bundled in `hw/syn/libs/` |
| `ASAP7` | ASAP7 7nm | `/mnt/nas0/eda.libs/asap7/asap7sc7p5t_28/LIB/NLDM` |
| `SAED14` | SAED 14nm SLVT | `/mnt/nas0/eda.libs/saed14/EDK_03_2025` |

### Running Synthesis

```bash
cd hw/syn/synopsys

# Default library, 1 core
PREFIX=test make synthesis

# ASAP7 library, 4 cores, with SAIF power
PREFIX=test NUM_CORES=4 LIB_TYPE=ASAP7 \
  SAIF_FILE=/path/to/trace.saif SAIF_INST=<inst> \
  make synthesis

# Synthesis without SRAM macros (blackbox)
PREFIX=test make synthesis-nosram

# Synthesis with estimated SRAM area
PREFIX=test make synthesis-estsram
```

Key variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `PREFIX` | `build` | Build directory prefix |
| `TOP_LEVEL_ENTITY` | `Vortex` | Top module name |
| `NUM_CORES` | - | Shorthand for core configuration |
| `CONFIGS` | - | Additional design macros |
| `CLOCK_FREQ` | 800 | Target frequency in MHz |
| `DELAY_UNC` | 0.02 | Clock uncertainty (fraction of period) |
| `DELAY_IO` | 0.05 | I/O delay (fraction of period) |
| `LIB_TYPE` | `DEFAULT` | Technology library selection |
| `SAIF_FILE` | - | SAIF file for power annotation |
| `SAIF_INST` | - | Instance path prefix in SAIF |

Build directory: `<PREFIX>_<TOP_LEVEL_ENTITY>/` (e.g., `test_Vortex/`).

### Synthesis Variants

- **`synthesis`**: Full synthesis with generated SRAM wrappers from the technology library's SRAM `.db` files. Requires the library to provide SRAM models.
- **`synthesis-nosram`**: Synthesis without any SRAM logic. RAM modules are inferred by DC.
- **`synthesis-estsram`**: Blackboxes `VX_dp_ram_asic` and `VX_sp_ram_asic`, then estimates their area from port dimensions (same approach as Yosys).

### Where to Find Reports

All reports are under `<BUILD_DIR>/reports/`:

| Report | Content |
|--------|---------|
| `area.rpt` | Hierarchical area breakdown |
| `qor.rpt` | Quality of Results summary (area, timing, utilization) |
| `timing_max.rpt` | Setup timing (50 worst paths, with nets/transitions/capacitance) |
| `timing_min.rpt` | Hold timing (50 worst paths) |
| `clock_skew.rpt` | Clock skew analysis |
| `constraints_violators.rpt` | All constraint violations |
| `check_design.rpt` | Pre-synthesis design checks |
| `power_active.rpt` | SAIF-annotated hierarchical power (if SAIF_FILE provided) |
| `power_vectorless.rpt` | Vectorless power estimate (if no SAIF_FILE) |
| `saif_annotation_coverage.rpt` | SAIF annotation coverage statistics |

Outputs are under `<BUILD_DIR>/out/`:

| File | Content |
|------|---------|
| `<TOP>.mapped.ddc` | Synopsys binary netlist |
| `<TOP>.mapped.v` | Mapped gate-level Verilog |
| `<TOP>.mapped.sdf` | Standard Delay Format for back-annotation |
| `<TOP>.post_compile.sdc` | Post-compile timing constraints |

### Finding Key Metrics

- **Total area**: In `area.rpt`, look for the top-level `Total cell area`. The SRAM estimated area (if using `synthesis-estsram`) is printed in the build log as `Total Estimated SRAM Area`.
- **Timing / Fmax**: In `timing_max.rpt`, the slack of the first path gives the worst negative slack (WNS). Fmax = 1 / (target_period - WNS). Also check `qor.rpt` for a summary.
- **Power**: In `power_active.rpt` (with SAIF) or `power_vectorless.rpt` (without), the hierarchical breakdown shows internal, switching, and leakage power per module.
- **Gate count**: In `qor.rpt`, look for `Design Area` and `Number of cells`.

---

## Understanding Power Reports

Power reports across all flows break down total power into similar categories. Understanding these helps identify optimization targets.

### Power Components

| Component | Description |
|-----------|-------------|
| **Dynamic power** | Power consumed by signal switching activity |
| &nbsp;&nbsp;Internal | Short-circuit current during output transitions within cells |
| &nbsp;&nbsp;Switching | Charging/discharging of interconnect and load capacitances |
| **Static (leakage) power** | Power consumed even when signals are not switching; due to sub-threshold and gate leakage currents |

**Total Power = Dynamic (Internal + Switching) + Static (Leakage)**

### Vectorless vs. Activity-Annotated

- **Vectorless**: The tool assumes a default toggle rate (typically 12.5%) and static probability (0.5) for all signals. Provides a rough baseline but can significantly over- or under-estimate actual power.
- **SAIF/VCD-annotated**: Uses real switching activity captured during simulation. Much more accurate for the specific workload simulated. Signals not covered by the SAIF/VCD fall back to the default toggle rate.

Always compare the vectorless and annotated reports to understand which modules differ most from the default assumption.

### Tips

- Run a representative workload when generating SAIF files. Short or trivial tests will underestimate steady-state power.
- Check SAIF annotation coverage reports (`saif_annotation_coverage.rpt` in Synopsys, `saif_unannotated.rpt` in Yosys) to ensure good signal coverage.
- For Xilinx, the power report includes device-specific contributions (clocking, I/O, BRAM, DSP power) that are not present in ASIC flows.
- For hierarchical analysis, look at per-module power breakdowns to identify the most power-hungry blocks (e.g., caches, FPU, TCU).
