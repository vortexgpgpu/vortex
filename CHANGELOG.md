# Changelog

All notable changes to Vortex are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project
follows the version pins recorded in [VERSION](VERSION) (`VORTEX_VERSION`,
`TOOLCHAIN_REV`, `GEM5_REV`).

## [3.0] — Unreleased

The 3.0 release introduces a fixed-function graphics stack (rasterizer, texture units, and output mergers), tensor core structured sparsity (2:4), warpgroup-level matrix multiplication (WGMMA), global-to-local data transfer acceleration (DXA), a new hardware kernel scheduler (KMU) and Command Processor (CP) architecture, a new asynchronous runtime API (`vortex2.h`), asynchronous barriers with arrive/wait/event semantics, compressed instruction set (RVC) support, partial support for hardware atomics, an MMU/SV32 virtual memory stack, a Mesa/lavapipe Vulkan backend (`vortexpipe`), HIP via chipStar, gem5 integration, a SimX v3 TLM architecture with fixed-size handshake channels, productized Synopsys and Yosys ASIC synthesis flows, and a refreshed toolchain (LLVM 20, POCL 7.0). Build and configuration infrastructure was reworked: TOML-driven HW configuration ([VX_config.toml](VX_config.toml) + [VX_types.toml](VX_types.toml)) decoupling SimX/runtime from the RTL source tree, a `VX_CFG_` macro namespace that resolves toolchain preprocessor collisions, retirement of the global `toolchain_env.sh` to enable parallel multi-version Vortex worktrees on the same shell, consolidation of `kernel/`/`runtime/` under a shared `sw/` root, a single-source [VERSION](VERSION) file driving CI toolchain pinning, Perfetto trace export ([ci/perfetto.py](ci/perfetto.py)), and new top-level [AGENTS.md](AGENTS.md) + [CONTRIBUTING.md](CONTRIBUTING.md) for AI-agent and contributor workflows.

### Added

- **Tensor Core Unit (TCU).** Re-architected into a block-level RTL pipeline under `hw/rtl/tcu/` with swappable arithmetic backends (`tfr`, `dpi`, `dsp`, `bhf`); supports fp32 / fp16 / bf16 / fp8 / tf32 / i8 / u8 / i4 / u4 gated by `VX_CFG_TCU_*_ENABLE`.
- **Unified int + float TCU pipeline.** v2.x's split `VX_tcu_fp` / `VX_tcu_fedp_int` paths collapsed into one FEDP per backend that dispatches both integer and floating-point ops through a single case statement. *Why:* halves the TCU instantiation area and lets sparse / WGMMA / DXA wiring reuse a single datapath instead of being duplicated.
- **New TCU numeric formats.** Adds **FP8 (e4m3)**, **BF8 (e5m2)**, and **TF32** on top of the v2.x set (fp32 / fp16 / bf16 / i32 / i8 / u8 / i4 / u4). Each is gated by its own `VX_CFG_TCU_{FP8,BF16,TF32}_ENABLE`; format dispatch is unified across all four FEDP backends.
- **WMMA register-file bank-conflict-free mapping.** TCU micro-op generation in `TcuUopGen` (and the SimX [sim/simx/tcu/tcu_unit.cpp:1332](sim/simx/tcu/tcu_unit.cpp#L1332) bank-conflict-free formulas) permutes A / B / C operand offsets so every uop's three RF reads land in different GPR banks; separate formula classes cover sparse, dense `NT∈{4,16,64}`, and dense `NT∈{8,32}`. *Why:* drops issue-stage stall cycles to zero on the TCU MMA loop — v2.x used a naive `(step % sub_blocks) * block_size` offset that incurred bank collisions on every other uop.
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
- **In-house 32-bit IEEE-754 FPU (`VX_fpu_std`).** Fully RV-compliant scalar FPU built from Vortex-owned blocks — `VX_fma_unit` (6-cycle fused multiply-add, F32/F64), `VX_fdivsqrt_unit` (17-cycle fused radix-2 non-restoring FDIV + FSQRT, single-lane pipelined), `VX_fcvt_unit`, `VX_fncp_unit` — selected via `VX_CFG_FPU_TYPE_STD`. *Why:* removes the FPNEW dependency from the ASIC/Yosys/Synopsys and FPGA flows; the new units deliver higher fmax, lower latency, and smaller area than the FPNEW path in the same configurations, and the in-tree source unblocks block-level tuning that vendoring made impractical.
- **RISC-V `Zicond` (conditional ops).** `CZERO.EQZ` / `CZERO.NEZ` integrated end-to-end (decode in `VX_decode.sv`, ALU in `VX_alu_int.sv`); gated by `VX_CFG_EXT_ZICOND_ENABLE`. Adds an ISA-level branchless-select primitive used by LLVM 20's codegen.
- **Pack-load intrinsics (`vx_packlb_f` / `vx_packlh_f`).** Single-instruction strided loads that fold 4×byte (`PACKLB`) or 2×halfword (`PACKLH`) loads into one front-end issue, expanded by `VX_uop_packld` into N back-to-back LSU uops with `eff_rs1 = rs1 + rs2 × uop_idx`. Used heavily by [sw/kernel/include/vx_tensor.h](sw/kernel/include/vx_tensor.h) for TCU tile-row packing.
- **RTL library expansion under `hw/rtl/libs/`.** New arithmetic primitives (CSA 3:2 / 4:2 / mod-4 / generic-tree, Wallace + folded multipliers, Kogge-Stone parallel-prefix adder) shared by `VX_fma_unit` and the TFR TCU backend; new `VX_stream_dispatch` / `VX_stream_fork` / `VX_stream_join` replacing ad-hoc fork/join code in DCR/DXA/GBAR paths; inference-based ICG cell (`VX_clockgate`) instantiated for per-core gating in `VX_socket`.
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

- **Source-tree consolidation.** `kernel/` + `runtime/` → `sw/{kernel,runtime,common}`; new `sw/common/` holds code shared by device and host (rvfloats, softfloat_ext, tensor_cfg, graphics utils, mem_alloc). *Why:* removes include-path duplication and mirrors the `sim/{simx,common,...}` layout.
- **riscv-tests `.bin` blobs removed from the source tree.** Pre-built `tests/riscv/isa/*.bin` (`c56562fe`) and `tests/riscv/benchmarks_{32,64}/*.bin` (`cd1656cf`) collapsed into a single on-demand build under [tests/riscv/common.mk](tests/riscv/common.mk) that clones upstream `riscv-tests` at a pinned commit and builds per-XLEN behind a stamp file. *Why:* drops binary blobs from version control and pins behaviour to a single upstream commit instead of stale checked-in artifacts.
- **Configuration moved to TOML.** [VX_config.toml](VX_config.toml) + [VX_types.toml](VX_types.toml) replace `hw/rtl/VX_config.vh`; [ci/gen_config.py](ci/gen_config.py) emits per-target headers (`build/hw/VX_config.vh`, `build/sw/VX_config.h`) and `-D` overrides from one source, with `expr:` / `[[enum]]` / `[[builtin]]` / `[[param]]` semantics. *Why:* gives the config typed scalars / cross-key expressions / typed enums the flat `\`define` block could not express, and decouples SimX/runtime from the RTL source tree (no more `-I$(ROOT_DIR)/hw`).
- **Configuration namespace.** All HW config macros now carry the `VX_CFG_` prefix; HW/SW layering split — `VX_config.h` is HW/sim-private. *Why:* resolves preprocessor collisions with LLVM/Clang and Verilator/SV `\`define`s, and stops HW config from leaking into kernel-side toolchain invocations.
- **No more global `toolchain_env.sh`.** Each `build/` carries its own resolved tool paths. *Why:* enables parallel multi-version Vortex worktrees on the same shell (the old `source ci/toolchain_env.sh` hijacked `$PATH`).
- **Build system.** Tool-path env vars unified on the `_PATH` suffix; shared `hw/syn/common.mk`; standardized `OPT_LEVEL` across synthesis backends; LLVM default target fixed; dead `hw/config` call dropped. *Why:* normalizes the build/synthesis surface so cross-backend changes touch one place.
- **SimObject channels — explicit fixed-size handshaking.** Unbounded `std::queue` + blocking `push`/`pop` replaced by fixed-capacity channels + non-blocking `try_send` / `try_pop` (RTL ready/valid analog); `[[nodiscard]]` forces producers to handle backpressure at the call site. *Why:* makes SimX model true RTL backpressure (1:1 with ready/valid) so buffering bugs surface in C++ instead of only at RTL bring-up.
- **Scoreboard issue arbiter — GTO (Greedy-Then-Oldest).** `VX_scoreboard` now drives a dedicated [VX_gto_arbiter](hw/rtl/libs/VX_gto_arbiter.sv) with a `suppress` mask for FU-stalled warps (they keep aging but are skipped for selection until the FU drains). v2.x used `VX_stream_arb` (round-robin). *Why:* GTO matches mainstream GPU warp-issue policy — finish the currently-running warp before switching — yielding better ILP/cache locality than naive RR.
- **Warp scheduler — per-warp ibuffer-capacity gate.** `VX_scheduler` tracks per-warp `ibuf_full` and computes `schedule_warps = ready_warps & ~ibuf_full`, with an `all_ibuf_full ? ready_warps : preferred_warps` fallback to keep pipelines absorbing transient stalls. v2.x scheduled solely on `active_warps & ~stalled_warps`. *Why:* prevents the scheduler from issuing a warp whose decoded uops will only block on a full ibuffer downstream — wasting fetch/decode bandwidth and pushing back-pressure into the front end.

### Fixed

- **`CMD_CACHE_FLUSH` coherence — simx + rtlsim.** Cache-flush command now waits for in-flight LSU stores to drain before signalling done (commit `e9f33598`). *Symptom before:* `vx_queue_destroy` returned while writebacks were still pending in the LSU mem-scheduler, occasionally racing with the next launch's reads; `lsu_queue_empty` is now plumbed up into `Core.busy` and `VX_cache_flush` drains MSHRs before completion.

### Known limitations

- **Hardware atomics require L1-as-LLC.** No inter-cache coherence yet, so with L2/L3 enabled the LLC bank cannot observe upstream-cached values and `LR`/`SC` + `AMO*` RMW semantics break. Unblock: future MSI/MESI directory on the cache bus.
- **`--hip` is `rv64`-only.** chipStar emits SPIR-V with `OpMemoryModel Physical64`; POCL rejects it on a 32-bit Vortex device.
- **`--vm` is SimX-only.** Host-shadow PT is modelled in SimX C++ memory; rtlsim's `DeviceMemIO` is not yet wired through the same shim.

## [2.3] — 2026-05-11

Last v2.x maintenance release. See `git log v2.2..v2.3`.

## Earlier releases

Tags `v0.2.0` through `v2.3` predate this changelog. Use `git log`
and the GitHub releases page for history.
