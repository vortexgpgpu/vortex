# Vortex Codebase

The directory/file layout of the Vortex codebase is as follows:

- `VX_config.toml` / `VX_types.toml`: hardware configuration system — the single
  source of truth for all build-time parameters (see the
  [build configuration system](designs/build_configuration_system.md)).
- `ci`: continuous integration scripts
  - `testcases`: declarative test catalog (YAML), executed by the pytest harness
  - `baselines/perf`: golden performance baselines for the perf-regression gate
  - `baselines/synthesis/xilinx`: golden FPGA synthesis baselines for the fpga_gate
  - `blackbox.sh`: universal application launcher for all drivers
  - `regression.sh`: local entry point into the CI catalog
- `docs`: documentation
  - `designs`: detailed design documents for Vortex subsystems
- `hw`:
  - `rtl`: hardware RTL sources
    - `core`: core pipeline (fetch, decode, issue, execute, LSU, commit)
    - `cache`: cache subsystem (banks, MSHR, AMO engine, flush)
    - `mem`: memory subsystem (arbiters, adapters, local memory)
    - `fpu`: floating point unit
    - `cp`: command processor
    - `tcu`: tensor core unit (WGMMA, structured sparsity)
    - `dxa`: asynchronous data-transfer accelerator (DMA/multicast)
    - `rtu`: ray-tracing unit (BVH traversal, intersection)
    - `raster`, `tex`, `om`, `gfx`: graphics fixed-function pipeline
      (rasterizer, texture units, output merger, shared graphics logic)
    - `afu`: FPGA accelerator functional unit shells (OPAE, XRT)
    - `interfaces`: SystemVerilog interfaces for inter-module communication
    - `libs`: general-purpose RTL modules (queues, arbiters, crossbars, encoders)
  - `dpi`: DPI models shared by the RTL simulators
  - `syn`: synthesis flows
    - `altera`: Altera Quartus synthesis scripts
    - `xilinx`: Xilinx Vivado/Vitis synthesis scripts
    - `synopsys`: Synopsys ASIC synthesis scripts
    - `yosys`: Yosys open-source synthesis scripts
  - `unittest`: Verilator-based unit tests for individual hardware components
  - `scripts`: RTL build and preprocessing utilities
- `sw`: software stack
  - `kernel`: device-side kernel API
    - `include`: public kernel headers (installed)
    - `src`: kernel runtime implementation (startup, scheduling intrinsics)
    - `linker`: linker scripts for kernel binaries
  - `runtime`: host-side runtime and drivers
    - `include`: public driver headers (installed)
    - `stub`: driver stub library (dynamic backend dispatch)
    - `simx`: driver backend for the SimX simulator
    - `rtlsim`: driver backend for the RTL simulator
    - `opae`: Intel OPAE FPGA driver (targets: fpga | asesim | opaesim)
    - `xrt`: Xilinx XRT FPGA driver (targets: hw | hw_emu | sw_emu)
    - `gem5`: driver backend for gem5 full-system integration
  - `common`: vortex-internal shared layer (on-wire ABI structs, host-side
    hardware models, shared helpers) — never installed
  - `gfx`: graphics fixed-function software emitters
- `sim`: simulators
  - `simx`: cycle-approximate C++ simulator
  - `rtlsim`: Verilator-based processor RTL simulator
  - `opaesim`: Intel OPAE AFU RTL simulator
  - `xrtsim`: Xilinx XRT AFU RTL simulator
  - `common`: shared simulator infrastructure (command processor model,
    DRAM model, ELF loader, virtual memory)
- `tests`: test suites
  - `riscv`: RISC-V conformance tests
  - `kernel`: device kernel tests
  - `regression`: host + kernel regression tests
  - `unittest`: host-side unit tests
  - `opencl`: OpenCL benchmarks and tests
  - `vulkan`: Vulkan tests (via the mesa-vortex driver)
  - `hip`: HIP tests (via chipStar)
  - `graphics`: graphics pipeline tests
  - `raytracing`: ray-tracing unit tests
  - `runtime`: driver API tests
  - `mpi`: multi-process tests
- `third_party`: external library submodules (cvfpu, softfloat, hardfloat,
  ramulator, cocogfx)
- `perf`: performance analysis resources
- `miscs`: miscellaneous resources
