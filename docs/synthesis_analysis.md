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

Common configuration flags:

| Flag | Description |
|------|-------------|
| `-DNUM_CLUSTERS=N` | Number of clusters |
| `-DNUM_CORES=N` | Number of cores per cluster |
| `-DNUM_WARPS=N` | Number of warps per core |
| `-DNUM_THREADS=N` | Number of threads per warp |
| `-DL2_ENABLE` | Enable shared L2 cache |
| `-DL3_ENABLE` | Enable shared L3 cache |
| `-DEXT_TCU_ENABLE` | Enable Tensor Core Unit |
| `-DEXT_DXA_ENABLE` | Enable DXA extension |
| `-DEXT_V_ENABLE` | Enable Vector extension |
| `-DDCACHE_SIZE=N` | Set data cache size in bytes |

Example:

```bash
CONFIGS="-DNUM_CLUSTERS=1 -DNUM_CORES=4 -DL2_ENABLE -DEXT_TCU_ENABLE"
```

All flows also support the `NUM_CORES` shorthand which auto-selects a pre-defined cluster/core/L2 configuration:

```bash
NUM_CORES=4   # equivalent to -DNUM_CLUSTERS=1 -DNUM_CORES=4 -DL2_ENABLE
NUM_CORES=16  # equivalent to -DNUM_CLUSTERS=1 -DNUM_CORES=16 -DL2_ENABLE
NUM_CORES=32  # equivalent to -DNUM_CLUSTERS=2 -DNUM_CORES=16 -DL2_ENABLE
```

### Using PREFIX for Isolated Builds

Use `PREFIX=<unique_build_dir>` to keep builds separate. Each flow creates a build directory derived from `PREFIX` so that multiple configurations can coexist without overwriting each other:

```bash
# Xilinx XRT: creates build_4c_<platform>_<target>/
PREFIX=build_4c NUM_CORES=4 make -C hw/syn/xilinx/xrt

# Synopsys: creates my_test_Vortex/
PREFIX=my_test make -C hw/syn/synopsys synthesis
```

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
CONFIGS="-DEXT_TCU_ENABLE" make tcu

# Synthesize a 4-core Vortex without the platform wrapper
CONFIGS="-DNUM_CORES=4 -DL2_ENABLE" make vortex

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
DEVICE_FAMILY=arria10 CONFIGS="-DEXT_TCU_ENABLE" make tcu

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
  CONFIGS="-DL2_ENABLE -DDCACHE_SIZE=8192" \
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

Located in `hw/syn/yosys/`. Uses Yosys for synthesis, optional ABC for technology mapping, and OpenSTA for static timing analysis.

### Running Synthesis

```bash
cd hw/syn/yosys

# Synthesis only (generic gates)
PREFIX=test NUM_CORES=1 make synthesis

# Synthesis + technology mapping
PREFIX=test NUM_CORES=1 make techmap

# Full flow: synthesis + mapping + STA + power
PREFIX=test NUM_CORES=1 SAIF_FILE=/path/to/trace.saif SAIF_INST=<inst> make timing
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
| `LIB_TGT` | NanGate 15nm OCL | Liberty file for technology mapping |
| `SAIF_FILE` | - | SAIF file for power annotation |
| `SAIF_INST` | - | Instance path prefix to strip |

Build directory: `<PREFIX>_<TOP_LEVEL_ENTITY>/` (e.g., `test_Vortex/`).

The flow uses `sv2v` to convert SystemVerilog sources to Verilog before feeding them to Yosys.

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
| `power.rpt` | Power estimate (vectorless or SAIF-annotated) |
| `power_hier.rpt` | Hierarchical power breakdown |
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
