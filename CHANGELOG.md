# Changelog

All notable changes to Vortex are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project
follows the version pins recorded in [VERSION](VERSION) (`VORTEX_VERSION`,
`TOOLCHAIN_REV`, `GEM5_REV`).

## [3.0] — Unreleased

The 3.0 release introduces a fixed-function graphics stack (rasterizer, texture units, and output mergers), tensor core structured sparsity (2:4), warpgroup-level matrix multiplication (WGMMA), global-to-local data transfer acceleration (DXA), a new hardware kernel scheduler (KMU) and Command Processor (CP) architecture, a new asynchronous runtime API (`vortex2.h`), asynchronous barriers with arrive/wait/event semantics, compressed instruction set (RVC) support, partial support for hardware atomics, an MMU/SV32 virtual memory stack, a Mesa/lavapipe Vulkan backend (`vortexpipe`), HIP via chipStar, gem5 integration, a SimX v3 TLM architecture with fixed-size handshake channels, productized Synopsys and Yosys ASIC synthesis flows, and a refreshed toolchain (LLVM 20, POCL 7.0). Build and configuration infrastructure was reworked: TOML-driven HW configuration ([VX_config.toml](VX_config.toml) + [VX_types.toml](VX_types.toml)) decoupling SimX/runtime from the RTL source tree, a `VX_CFG_` macro namespace that resolves toolchain preprocessor collisions, retirement of the global `toolchain_env.sh` to enable parallel multi-version Vortex worktrees on the same shell, consolidation of `kernel/`/`runtime/` under a shared `sw/` root, a single-source [VERSION](VERSION) file driving CI toolchain pinning, Perfetto trace export ([ci/perfetto.py](ci/perfetto.py)), and new top-level [AGENTS.md](AGENTS.md) + [CONTRIBUTING.md](CONTRIBUTING.md) for AI-agent and contributor workflows.

### Added

- **Tensor Core Unit (TCU).** Re-architected into a block-level RTL pipeline under `hw/rtl/tcu/` with swappable arithmetic backends (`tfr`, `dpi`, `dsp`, `bhf`); supports fp32 / fp16 / bf16 / fp8 / tf32 / i8 / u8 / i4 / u4 gated by `VX_CFG_TCU_*_ENABLE`.
- **Tensor-core structured sparsity (2:4).** `VX_tcu_sp_mux` + `VX_tcu_meta` datapath plus host `compress_2to4_matrix` / `prune_2to4_matrix` helpers; gated by `VX_CFG_TCU_SPARSE_ENABLE`.
- **Warpgroup-level MMA (WGMMA).** Per-warp `NRA=4` / variable-`NRC` fragment layout, S/R source modes, smem descriptor path; gated by `VX_CFG_TCU_WGMMA_ENABLE`.
- **Data-transfer Acceleration (DXA).** Async global→local DMA engine for tile staging (`hw/rtl/dxa/` + `sim/simx/dxa/`).
- **Hardware Kernel Management Unit (KMU).** New scheduler block (`hw/rtl/VX_kmu.sv` + `sim/simx/kmu/`) that owns CTA dispatch from the CP launch path.
- **Command Processor (CP) v3.** New `hw/rtl/cp/` block + host-resident command ring (`CMD_LAUNCH`, `CMD_MEM_*`, `CMD_DCR_*`, `CMD_CACHE_FLUSH`, `CMD_EVENT_*`); integrated end-to-end across xrt, opae, simx, rtlsim behind `VORTEX_USE_CP`.
- **Asynchronous `vortex2.h` runtime API.** Queues, events, modules, kernels, UVA raw-pointer kernel args, per-queue worker thread; legacy `vortex.h` retained as a thin wrapper.
- **C++ software CP model** (`sim/common/cmd_processor.cpp`) shared by simx and rtlsim.
- **Graphics stack (RASTER / TEX / OM).** Fixed-function 3D pipeline: `hw/rtl/{raster,tex,om}/` + `VX_graphics.sv` + matching SimX models; `--graphics` regression group.
- **Vulkan support** via a new Mesa Gallium driver `vortexpipe` selected through the `lavapipe` ICD; `tests/vulkan/` suite (compute, draw3d, depth, textured, raytrace); Mesa shipped via the prebuilt toolchain; rv64 path enabled.
- **HIP support on rv64** via chipStar (`vx_llvm` Clang HIP path + `libhip_vortex`).
- **Hardware atomics (partial).** RISC-V `A`-extension via cache-resident `VX_amo_unit` + `LR`/`SC` reservation table; gated by `VX_CFG_EXT_A_ENABLE`. Currently restricted to L1-as-LLC — see [Known limitations](#known-limitations).
- **Compressed instruction set (RVC).** New `VX_decompressor` block in the fetch stage ([hw/rtl/core/VX_decompressor.sv](hw/rtl/core/VX_decompressor.sv) + [sim/simx/decompressor.cpp](sim/simx/decompressor.cpp)); gated by `VX_CFG_EXT_C_ENABLE`. v2.x shipped the test binaries but had no decompressor.
- **Asynchronous barriers** with `arrive` / `wait` / `expect_tx` semantics. `VX_bar_unit` + `vortex::barrier` host API; `expect_tx` is the hook DXA multicast uses to declare expected bytes.
- **MMU / virtual memory** (SV32, rv32-only). Host-shadow page table + `DeviceMemIO` refactor + `--vm` regression group.
- **gem5 integration.** `VortexGPGPU` SimObject + x86/aarch64 host runtimes; `ci/regression.sh --gem5` + `VORTEX_GEM5_ARM=1`.
- **SimX v3 TLM architecture.** Transaction-level memory packets (`MemReq`/`MemRsp` with `shared_ptr<mem_block_t>` payloads) and reusable TLM cache / switch / coalescer modules across the L1/L2/L3 hierarchy, tcache/ocache/rcache, DXA, and CP DMA paths.
- **ASIC synthesis flows.** `hw/syn/{synopsys,yosys}/` productized: shared `hw/syn/common.mk`, bundled `NanGate_15nm_OCL.db` standard cells, standardized `OPT_LEVEL`; legacy `hw/syn/modelsim` flow retired.
- **Preemption groundwork.** Synchronous RISC-V trap path + native `riscv-tests` support.
- **Toolchain refresh.** LLVM 20, POCL 7.0, chipStar.
- **Versioned toolchain pipeline for CI.** [VERSION](VERSION) is the single source of truth (`VORTEX_VERSION`, `TOOLCHAIN_REV`, `GEM5_REV`); CI cache keys + installer scripts both honour it; bumping a pin rolls the CI cache.
- **Perfetto trace integration.** [ci/perfetto.py](ci/perfetto.py) renders RTL and SimX traces into Chrome Trace JSON (auto-detects flavour); see [docs/perfetto_analysis.md](docs/perfetto_analysis.md).
- **AI-agent integration via [AGENTS.md](AGENTS.md).** Canonical entry point for AI agents and human contributors — foundation rules, documentation map, build/test/design invariants.
- **Test groups.** New `--vulkan`, `--gem5`, `--hip`, `--amo`, `--tensor`, `--tensor_sp`, `--tensor_wg`, `--cupbop`, `--dtm`, `--mpi`, `--rvc`, `--vm`, `--graphics` runners in `ci/regression.sh`; matrix expanded to `rv32` + `rv64`.
- **[CONTRIBUTING.md](CONTRIBUTING.md)** and this changelog at the repo root.

### Changed

- **Source-tree consolidation.** `kernel/` + `runtime/` → `sw/{kernel,runtime,common}`; new `sw/common/` holds code shared by device and host (rvfloats, softfloat_ext, tensor_cfg, graphics utils, mem_alloc).
- **Configuration moved to TOML.** [VX_config.toml](VX_config.toml) + [VX_types.toml](VX_types.toml) replace the hand-written `hw/rtl/VX_config.vh`; [ci/gen_config.py](ci/gen_config.py) emits per-target headers (`build/hw/VX_config.vh`, `build/sw/VX_config.h`) and `-D` overrides from one source. Adds `expr:` / `[[enum]]` / `[[builtin]]` / `[[param]]` semantics the flat `\`define` block could not express, and decouples SimX/runtime from the RTL source tree.
- **Configuration namespace.** All HW config macros now carry the `VX_CFG_` prefix to resolve preprocessor collisions with LLVM/Clang and Verilator/SV `\`define`s; HW/SW layering split — `VX_config.h` is HW/sim-private.
- **No more global `toolchain_env.sh`.** Each `build/` carries its own resolved tool paths, enabling parallel multi-version Vortex worktrees on one shell.
- **Build system.** Tool-path env vars unified on the `_PATH` suffix; shared `hw/syn/common.mk`; standardized `OPT_LEVEL` across synthesis backends; LLVM default target fixed; dead `hw/config` call dropped.
- **SimObject channels — explicit fixed-size handshaking.** Unbounded `std::queue` + blocking `push`/`pop` replaced by fixed-capacity channels + non-blocking `try_send` / `try_pop` (RTL ready/valid analog); `[[nodiscard]]` forces producers to handle backpressure at the call site.

### Known limitations

- **Hardware atomics require L1-as-LLC.** No inter-cache coherence yet, so with L2/L3 enabled the LLC bank cannot observe upstream-cached values and `LR`/`SC` + `AMO*` RMW semantics break. Unblock: future MSI/MESI directory on the cache bus.
- **`--hip` is `rv64`-only.** chipStar emits SPIR-V with `OpMemoryModel Physical64`; POCL rejects it on a 32-bit Vortex device.
- **`--vm` is SimX-only.** Host-shadow PT is modelled in SimX C++ memory; rtlsim's `DeviceMemIO` is not yet wired through the same shim.

## [2.3] — 2026-05-11

Last v2.x maintenance release. See `git log v2.2..v2.3`.

## Earlier releases

Tags `v0.2.0` through `v2.3` predate this changelog. Use `git log`
and the GitHub releases page for history.
