# v3.0 RTL Audit — v2.3 → HEAD

**Purpose:** review-grade catalogue of every `hw/rtl/*` change between
`v2.3` (commit `d425a56e`) and the current `tinebp-patch-2` tip. Each row
is one logical change. The **In CHANGELOG?** column marks whether the
change is already covered in [CHANGELOG.md](../../CHANGELOG.md) `[3.0]
Unreleased`:

- **Yes** — covered by an existing Added/Changed bullet.
- **Implicit** — falls under an umbrella bullet (e.g. all `hw/rtl/cp/`
  files implied by "CP v3"); not individually named.
- **No** — not in the changelog yet; review whether to promote.

File deltas (`hw/rtl/` only): **151 added · 21 deleted · 111 modified · 5 renamed**.

---

## Table

### A. Brand-new RTL subsystems

| Area | Files | Kind | Description | In CHANGELOG? |
|---|---|---|---|---|
| Command Processor | `hw/rtl/cp/` (18 SV files) | Feature | CP v3 block: ring-fetch, arbiter, engine, launch, DCR proxy, unpack, DMA, event unit, profiling; AXI-Lite regfile + AXI4 master | **Yes** |
| Kernel Management Unit | `hw/rtl/VX_kmu.sv` + `hw/rtl/core/VX_kmu_arb.sv`, `VX_cta_dispatch.sv` | Feature | HW scheduler that owns CTA dispatch from CP launch path; per-core CTA dispatcher consumes KMU descriptors | **Yes** (CTA dispatcher itself is implicit) |
| Data-transfer Acceleration | `hw/rtl/dxa/` (16 SV files) | Feature | Async global→local DMA engine for tile staging: core, desc-table, worker, setup/dispatch, addr-gen, watchdog, completion | **Yes** |
| Graphics stack — RASTER | `hw/rtl/raster/` (7 SV files) | Feature | Tile/quad rasterizer with CSR-programmable viewport, multi-prim batching, mem-fifo + quad-fifo | **Yes** |
| Graphics stack — TEX | `hw/rtl/tex/` (16 SV files) | Feature | Texture units: addr-gen, sampler, format decode, LERP, wrap/sat/stride, TCACHE backing | **Yes** |
| Graphics stack — OM | `hw/rtl/om/` (17 SV files) | Feature | Output mergers: blending (func / min-max / multadd), compare, depth/stencil, logic op, OCACHE backing | **Yes** |
| Graphics top-level | `hw/rtl/VX_graphics.sv` | Feature | Wraps RASTER + TEX + OM into a single instantiable block | **Yes** (implicit) |
| MMU subsystem | `hw/rtl/mem/VX_mmu.sv`, `VX_mmu_tlb.sv`, `VX_mmu_ptw.sv` | Feature | SV32 MMU with TLB + hardware page-table walker; gated by `VX_CFG_VM_ENABLE` | **Yes** |
| Compressed-ISA decoder | `hw/rtl/core/VX_decompressor.sv` | Feature | Expands `rv*c` 16-bit encodings to 32-bit base ISA before decode | **Yes** |
| Async barrier unit | `hw/rtl/core/VX_bar_unit.sv` | Feature | Backs `vortex::barrier` with generation-phase counter + `expect_tx` semantics | **Yes** |
| AMO / hardware atomics | `hw/rtl/cache/VX_amo_unit.sv`, `VX_amo_alu.sv` | Feature | RISC-V `A`-extension at the LLC cache bank + `LR`/`SC` reservation table | **Yes** (L1-as-LLC limitation called out) |

### B. New RTL primitives in `hw/rtl/libs/`

| Module | Kind | Description | In CHANGELOG? |
|---|---|---|---|
| `VX_gto_arbiter.sv` | Feature | Greedy-Then-Oldest arbiter with `suppress` mask; used by scoreboard issue + memory scheduler | **Yes** (scoreboard GTO bullet) |
| `VX_clockgate.sv` | Feature | Library-style clock-gate cell for synthesizable power gating | **No** |
| `VX_csa_32.sv`, `VX_csa_42.sv`, `VX_csa_mod4.sv`, `VX_csa_tree.sv` | Feature | Carry-save-adder building blocks (3:2 / 4:2 / mod-4 / generic tree) used by TFR TCU + Wallace multiplier | **No** (implicit under TCU re-arch) |
| `VX_ks_adder.sv` | Feature | Kogge-Stone parallel-prefix adder used in TFR TCU and Wallace fold | **No** |
| `VX_wallace_mul.sv`, `VX_fold_mul.sv` | Feature | Wallace-tree multiplier + folded-radix multiplier — used by TFR TCU FEDP path | **No** |
| `VX_axi_arb2.sv` | Feature | 2-input AXI arbiter; used by CP AXI xbar | **No** (implicit under CP) |
| `VX_stream_dispatch.sv`, `VX_stream_fork.sv`, `VX_stream_join.sv` | Refactor | Reusable stream split/join primitives (replaces ad-hoc fork/join inlined across the pipeline) | **No** |

### C. Core pipeline — modified or replaced

| File | Kind | Description | In CHANGELOG? |
|---|---|---|---|
| `VX_scheduler.sv` (replaces `VX_schedule.sv`, 446→726 lines) | Feature + refactor | Renamed; adds per-warp `ibuf_full` tracking + `schedule_warps = ready_warps & ~ibuf_full`; CTA-aware active-warp mask | **Yes** (ibuf-capacity gate bullet) |
| `VX_scoreboard.sv` (+126/-25 lines) | Feature | Switched issue arbiter from RR `VX_stream_arb` to `VX_gto_arbiter` with `suppress` mask | **Yes** (GTO bullet) |
| `VX_fetch.sv` | Feature/refactor | Decompressor wiring; ibuffer-occupancy hand-off comments | **Partial** (RVC bullet) |
| `VX_decode.sv` (+436 lines!) | Feature + bugfix | Adds decode for: RVC, AMO, TCU full op family (WMMA / WGMMA / META_STORE), DXA descriptors, KMU intrinsics, CP intrinsics, vx_packl[bh], barrier intrinsics; CUSTOM-0/1/2 opcode dispatch | **Partial** (each feature is in changelog but the decode-side wiring is implicit) |
| `VX_ibuffer.sv` (+48 lines) | Refactor | Per-warp ready-in routing; cleaner pop accounting matching the new scheduler ibuf_full gate | **No** (implicit under ibuf gate) |
| `VX_issue.sv`, `VX_issue_slice.sv` | Refactor | Re-pipelined to feed the GTO scoreboard + new dispatcher | **No** |
| `VX_opc_unit.sv` | Refactor | Operand-collector reshape for variable-NRC TCU + sparse meta operand sources | **No** (implicit under TCU re-arch) |
| `VX_operands.sv` | Refactor | Bank-conflict-free GPR offset formulas for TCU A/B/C uops; sparse-friendly | **No** |
| `VX_dispatcher.sv` (new) + `VX_dispatch.sv` (deleted) | Refactor | Renamed from `VX_dispatch.sv` and re-architected as a per-EX-unit dispatcher with perf counters | **No** |
| `VX_lane_dispatch.sv` (renamed from `VX_dispatch_unit.sv`) | Refactor | Renamed for clarity; lane-fan-out path | **No** |
| `VX_lane_gather.sv` (renamed from `VX_gather_unit.sv`) | Refactor | Renamed for clarity; lane-merge path | **No** |
| `VX_dcr_arb.sv`, `VX_dcr_flush.sv` | Feature | Per-core DCR arbitration + flush handshake (consumed by CP DCR proxy) | **No** (implicit under CP) |
| `VX_txbar_arb.sv` | Feature | Cross-bar arbiter used by the txbar bus | **No** |
| `VX_uop_packld.sv` | Feature | Sequencer for `vx_packlb_f` / `vx_packlh_f` pack-load intrinsics (4×byte or 2×halfword strided loads → single instruction) | **No** |
| `VX_uop_sequencer.sv` (+ revert of stateless TCU uops) | Bugfix | Restored `uop_data` register after `feature_cp` regression (commit `10e53e81`) | **No** |
| `VX_csr_unit.sv` (+157/-25 lines) | Feature | New CSRs: KMU descriptors, CP launch, MMU `satp`, perfetto-trace markers, FPU-flag race-condition fix | **Partial** (MMU/CP implicit; FPU CSR fix unmentioned) |
| `VX_csr_data.sv` | Feature | New `csr_data` storage for KMU / CP / MMU / vx_serial CSRs | **No** |
| `VX_decode.sv` includes for `VX_CUSTOM0..3` opcode handling | Feature | New custom opcode dispatch for TCU + CP + DXA + KMU intrinsics | **No** (implicit) |
| `VX_alu_unit.sv`, `VX_alu_int.sv`, `VX_alu_muldiv.sv` | Refactor | Re-pipelined; muldiv split out as a sibling unit for FPGA timing; Zicond integration | **No** (Zicond not in changelog) |
| `VX_sfu_unit.sv` (+212 lines) | Feature | Adds barrier-arrive/wait, CTA-launch ack, KMU descriptor read, perfetto markers, CP-event signal/wait | **Partial** (barriers mentioned; CTA/KMU/CP-event SFU paths not) |
| `VX_wctl_unit.sv` (+204 lines) | Feature | Warp-control rewrites for KMU CTA fork/join, async barrier interactions, preemption trap path | **Partial** (preemption mentioned; KMU CTA fork/join wctl impl not) |
| `VX_lsu_unit.sv`, `VX_lsu_slice.sv` (+274 lines combined) | Feature + refactor | Adds LSU-drain signal (`lsu_queue_empty` plumbed to `Core.busy`) for CP `CMD_CACHE_FLUSH` coherence; MMU integration | **Yes** (CMD_CACHE_FLUSH bullet, but in the older fixed list now trimmed — currently *not* in changelog after the trim pass) |
| `VX_mem_unit.sv` | Refactor | Per-LSU-block mem scheduling; MMU dcache_bus_if hookup | **No** |
| `VX_commit.sv` | Refactor | Per-issue commit accounting; works with the new GTO arbiter | **No** |
| `VX_execute.sv` (+108 lines) | Feature | New EX paths for TCU WMMA/WGMMA, KMU, CP, MMU faults | **Partial** (implicit under feature bullets) |
| `VX_split_join.sv` | Refactor | Cleaned up SIMT divergence stack semantics | **No** |
| `VX_uuid_gen.sv` | Feature | Hierarchical UUID generation for perfetto/trace correlation across micro-op expansion | **No** (implicit under perfetto) |
| `VX_ipdom_stack.sv` | Bugfix | IPDOM stack overflow guard + reset clear (commit history) | **No** |
| `VX_pe_switch.sv` | Refactor | Per-issue priority-encoder switch matrix | **No** |
| `VX_core.sv` | Feature | Top-level reorg to wire KMU, CP, MMU, DXA, async barriers, AMO, TCU into the core | **No** (implicit) |

### D. FPU — re-architecture

| File | Kind | Description | In CHANGELOG? |
|---|---|---|---|
| `VX_fpu_fma.sv` → `VX_fma_unit.sv` (+other rebalancing) | Refactor | FPU subunits renamed under the `*_unit.sv` convention and rewrapped under a shared `VX_fpu_std.sv` selector | **No** |
| `VX_fpu_div.sv`, `VX_fpu_sqrt.sv` → `VX_fdivsqrt_unit.sv` | Refactor | Div + sqrt collapsed into a shared `fdivsqrt` unit (saves one DSP block on FPGA) | **No** |
| `VX_fpu_cvt.sv` → `VX_fcvt_unit.sv` | Refactor | Renamed; new MXFP path (since removed with MX strip) | **No** |
| `VX_fpu_ncp.sv` → `VX_fncp_unit.sv` | Refactor | Renamed; FCLASS / FMV / FSGN consolidated | **No** |
| `VX_fpu_std.sv` (new) | Refactor | New "standard" FPU dispatcher mux (DPI / DSP / FPNEW / STD); replaces inline `VX_fpu_unit.sv` mux | **No** |
| `VX_fpu_dpi.sv` | Refactor | DPI-FPU now routes through `VX_fcvt_unit` instead of the deleted `VX_fpu_cvt.sv` | **No** |
| `VX_fpu_dsp.sv`, `VX_fpu_fpnew.sv` | Refactor | Updated to match new subunit boundaries | **No** |

### E. TCU — re-architecture (table omits already-changelogged pieces)

| File | Kind | Description | In CHANGELOG? |
|---|---|---|---|
| `hw/rtl/tcu/VX_tcu_{abuf,bbuf,mbuf,tbuf}.sv` | Feature | A/B/M/T fragment buffers (new for v3.0 TCU pipeline) | **Yes** (implicit under TCU re-arch) |
| `hw/rtl/tcu/VX_tcu_core.sv` | Feature | Top-level TCU pipeline replacing v2.x `VX_tcu_top.sv` | **Yes** |
| `hw/rtl/tcu/VX_tcu_unit.sv` | Refactor | Per-block TCU wrapping (multi-block fanout) | **Yes** (implicit) |
| `hw/rtl/tcu/VX_tcu_uops.sv` | Refactor | Micro-op expansion (WMMA / WGMMA / META_STORE) | **Yes** (implicit) |
| `hw/rtl/tcu/VX_tcu_meta.sv`, `VX_tcu_sp_mux.sv` | Feature | Sparse metadata store + sparse operand mux | **Yes** (sparsity bullet) |
| `hw/rtl/tcu/tfr/*` (14 files) | Feature | TFR ("transfer-function") backend: per-format multiplier, classifier, max-exp reducer, normalizer, accumulator | **Yes** (implicit under TCU re-arch) |
| `hw/rtl/tcu/dpi/VX_tcu_fedp_dpi.sv` | Refactor | Re-targeted DPI golden FEDP under the new `dpi/` subdir | **Yes** (implicit) |
| `hw/rtl/tcu/bhf/VX_tcu_bhf_fp8mul.sv` | Feature | BHF backend fp8 multiplier | **Yes** (implicit) |
| `hw/rtl/tcu/dsp/VX_tcu_fedp_dsp.sv` (renamed) | Refactor | Moved under `dsp/` subdir | **Yes** (implicit) |

### F. Cache + memory subsystem

| File | Kind | Description | In CHANGELOG? |
|---|---|---|---|
| `VX_cache.sv`, `VX_cache_bank.sv` | Refactor | AMO-aware datapath (`AMO_ENABLE` parameter); LLC detection wired in | **Yes** (AMO bullet) |
| `VX_cache_flush.sv` | Bugfix | `CMD_CACHE_FLUSH` coherence: the bank now drains in-flight MSHRs before signalling done | **No** (was in the old fixed list, dropped by the trim pass) |
| `VX_cache_mshr.sv`, `VX_cache_repl.sv` | Refactor | TLM-style ready/valid; replacement policy parameterized | **No** |
| `VX_cache_bypass.sv`, `VX_cache_init.sv`, `VX_cache_cluster.sv`, `VX_cache_wrap.sv` | Refactor | Aligns with new TLM cache surface | **No** |
| `hw/rtl/mem/VX_mem_xbar.sv` (new) | Feature | Generic mem xbar used by clusters, DXA gmem, CP DMA | **No** |
| `VX_lsu_adapter.sv`, `VX_lsu_mem_arb.sv`, `VX_lsu_mem_if.sv` | Refactor | LSU↔mem boundary refactored for TLM + MMU + AMO | **Partial** (MMU implicit) |
| `VX_local_mem.sv`, `VX_lmem_switch.sv` | Refactor | Multi-bank local memory + LMEM switch redesign | **No** |
| `VX_gbar_*.sv` | Refactor | Global-barrier net updated for KMU CTA semantics | **No** |
| `VX_mem_switch.sv`, `VX_mem_arb.sv`, `VX_mem_bus_if.sv` | Refactor | Bus refactoring around new TLM surface | **No** |

### G. Interfaces

| File | Kind | Description | In CHANGELOG? |
|---|---|---|---|
| `VX_cache_flush_if.sv` (new) | Feature | Handshake interface for CP-driven cache flush | **No** |
| `VX_dcr_csr_if.sv`, `VX_dcr_flush_if.sv` (new) | Feature | DCR↔CSR + DCR-flush interfaces | **No** |
| `VX_kmu_bus_if.sv` (new) | Feature | KMU↔core CTA dispatch handshake | **Yes** (implicit) |
| `VX_sfu_csr_if.sv` (new) | Feature | SFU↔CSR interface for barrier/CTA-ack/perfetto | **No** |
| `VX_txbar_bus_if.sv` (renamed from `VX_commit_csr_if.sv`) | Refactor | Renamed to reflect new txbar bus semantics | **No** |
| `VX_sched_csr_if.sv`, `VX_schedule_if.sv`, `VX_warp_ctl_if.sv` | Refactor | Widened to carry CTA id, ibuf-empty, KMU/CP control | **No** |

### H. AFU integration (Xilinx XRT + Intel OPAE)

| File | Kind | Description | In CHANGELOG? |
|---|---|---|---|
| `hw/rtl/afu/xrt/VX_afu_wrap.sv`, `VX_afu_ctrl.sv`, `vortex_afu.v`, `vortex_afu.vh` | Feature | XRT AFU wrapper updated to expose CP AXI4 master + CP AXI-Lite slave; `VORTEX_USE_CP=1` path | **Yes** (implicit under CP integration) |
| `hw/rtl/afu/opae/vortex_afu.sv`, `ccip_*` | Feature + bugfix | OPAE AFU updated for CP integration end-to-end | **Yes** (implicit) |

### I. Build-system + config (RTL-side)

| File | Kind | Description | In CHANGELOG? |
|---|---|---|---|
| `hw/rtl/VX_config.vh` (deleted) | Refactor | Replaced by TOML-driven `build/hw/VX_config.vh` | **Yes** (TOML bullet) |
| `hw/rtl/VX_types.vh` (deleted) | Refactor | Replaced by TOML-driven `build/hw/VX_types.vh` | **Yes** |
| `hw/rtl/{Vortex,Vortex_axi,VX_socket,VX_cluster,VX_gpu_pkg,VX_define,VX_platform,VX_trace_pkg}.sv` | Refactor | Top-level reorg for KMU/CP/MMU/AMO/graphics + `VX_CFG_*` macro namespace migration | **Yes** (implicit) |
| 6 deleted `*_top.sv` files | Refactor | `VX_cache_top`, `VX_core_top`, `VX_issue_top`, `VX_mem_unit_top`, `VX_local_mem_top`, `VX_tcu_top`, `VX_schedule` removed; functionality folded into the renamed/replacement modules | **No** |

---

## Recommended promotions to CHANGELOG

Items currently **No** that I'd flag as worth a one-line bullet (rest can stay implicit):

1. **`CMD_CACHE_FLUSH` coherence + LSU drain fix** — substantive correctness fix tied to a specific shipped failure. Should land under a new **Fixed** section.
2. **FPU subunit reorganization + shared `fdivsqrt`** — saves an FPGA DSP block, user-visible in area reports.
3. **Pack-load intrinsics (`vx_packlb_f` / `vx_packlh_f`) + `VX_uop_packld`** — new compute intrinsic surface visible to kernel authors.
4. **`VX_clockgate.sv` + standard-cell synthesis** — productized clock gating is meaningful for ASIC users.
5. **CSA / Wallace / Kogge-Stone primitives** — new arithmetic library worth one line.
6. **TLM cache surface + reusable switch/coalescer/xbar** — already mentioned in SimX context, but the RTL-side equivalent (cache bank / lsu adapter / mem xbar / gbar / lmem refactor) is a separate landing worth one line.
7. **Stream split/join primitives (`VX_stream_dispatch/fork/join`)** — replaces ad-hoc fork/join code in the pipeline.
8. **Zicond integration in ALU** — RISC-V `Zicond` extension is a real ISA feature gained.

Items I'd **leave implicit**: every per-block buffer / arbiter / interface that's only ever exercised through an already-named umbrella feature (KMU, CP, DXA, TCU, graphics, MMU). The changelog is for readers, not for completeness.
