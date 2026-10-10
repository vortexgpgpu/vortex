# FourShadow ASIC sweep: one full GPU core, Synopsys DC, ASAP7 LVT/TT (14 runs)

Three experiments on one complete Vortex core: a thread-count sweep, an issue-width sweep, and a TCU feature ladder.
Every run gives timing, area and power. Expt 3 also gives a per-subcomponent area breakdown.

Use repo branch `master`.

## Setup common to all 14 runs

### Top module

**`Vortex`** with `NUM_CLUSTERS=1, NUM_CORES=1` and L2/L3 off. This is the synopsys Makefile's default top, and it has flat ports. It contains:

- the core: `g_clusters[0].cluster / g_sockets[0].socket / g_cores[0].core`
- the L1 I$ and D$ (`socket/icache`, `socket/dcache`), at 16 KB each by default
- the DXA engine (`socket/dxa_core`, present only when DXA is enabled). DXA is socket-resident, so it is outside `VX_core`, and this is why the top is `Vortex` and not `VX_core`.
- a thin uncore: KMU, DCR/KMU/gbar arbiters, and the passthrough L2/L3 wrappers

### Library and corner

- **PDK:** ASAP7 7.5-track (`asap7sc7p5t_28`), **LVT only, TT corner**, NLDM `.db`.
- **SRAM:** ASAP7 `srambank_*_6t122` macros, using the real-macro flow (`make synthesis`). See the SRAM audit section below. It is mandatory.
- **Clock:** a fixed **1 GHz** target (`CLOCK_FREQ=1000`). Keep the default 2% uncertainty and 5% I/O delay.
- **Effort:** `OPT_LEVEL=3` (`compile_ultra -retime`), with hierarchy preserved (see the flow changes below).
- **Power:** vectorless (no SAIF), using DC's default switching activity. Say so in the CSV.

### Defines common to all runs

```
-DVX_CFG_NUM_CLUSTERS=1 -DVX_CFG_NUM_CORES=1
-DVX_CFG_EXT_TCU_ENABLE
-DVX_CFG_TCU_TF32_ENABLE -DVX_CFG_TCU_FP8_ENABLE -DVX_CFG_TCU_INT8_ENABLE -DVX_CFG_TCU_INT4_ENABLE
-DVX_CFG_LMEM_LOG_SIZE=14
```

- **Formats:** these defines give tf32, fp16, bf16, fp8 (e4m3), bf8 (e5m2), int8 and int4. fp16/bf16 come from the default-on `VX_CFG_TCU_FP16_ENABLE`. MX, FP4, MXFP4 and NVFP4 are default-off, so leave them alone. Do **not** add `TCU_MX_ENABLE` or any FP4 flag.
- **LMEM = 16 KB:** the toml derives LMEM size from the lane count (`max(8 KB, NT × 2 KB)`), which would give 8 to 64 KB across these runs. `LMEM_LOG_SIZE=14` pins it to 16 KB. The bank count stays `NUM_LSU_LANES = NT`.
- **Left at their defaults, which the toml resolves for ASIC:**
  - `TCU_TYPE=TFR`
  - `TCU_USE_DSP=0` (Wallace-tree multipliers)
  - `TCU_LATENCY=4`
  - `FPU_TYPE=STD`, with the scalar RV32F FPU kept in because this is a full core
  - `XLEN=32`
  - I$/D$ at 16 KB, 4-way
  - `NUM_TCU_BLOCKS = ISSUE_WIDTH`
  - `NUM_OPCS = up(NW / (4 × IW))`

`-DSYNTHESIS -DASIC -DSYNOPSYS` are appended by the Makefile itself. **Pass `CONFIGS` as an environment variable, not as a make argument.** A command-line `make CONFIGS=...` overrides the Makefile's `CONFIGS += -DSYNTHESIS -DASIC -DSYNOPSYS`. That silently produces an FPGA-flavoured elaboration with DSP multipliers and no ASIC RAM wrappers.

## The 14 runs

Columns `TC_K`, `OPCS` and `LMEM banks` are derived values. Use them to sanity-check each run's elaboration log.

### Expt 1: thread-count sweep (WMMA dense)

NW=16, IW=1, TCU = WMMA only: no WGMMA, no sparsity, no DXA, no FEDP2K.

| Run | NT | NW | IW | TC_K | OPCS | LMEM banks × bytes | Extra defines |
|---|---|---|---|---|---|---|---|
| E1-NT4  | 4  | 16 | 1 | 2 | 4 | 4 × 4 KB   | `-DVX_CFG_NUM_THREADS=4 -DVX_CFG_NUM_WARPS=16 -DVX_CFG_ISSUE_WIDTH=1` |
| E1-NT8  | 8  | 16 | 1 | 2 | 4 | 8 × 2 KB   | `-DVX_CFG_NUM_THREADS=8 -DVX_CFG_NUM_WARPS=16 -DVX_CFG_ISSUE_WIDTH=1` |
| E1-NT16 | 16 | 16 | 1 | 4 | 4 | 16 × 1 KB  | `-DVX_CFG_NUM_THREADS=16 -DVX_CFG_NUM_WARPS=16 -DVX_CFG_ISSUE_WIDTH=1` |
| E1-NT32 | 32 | 16 | 1 | 4 | 4 | 32 × 512 B | `-DVX_CFG_NUM_THREADS=32 -DVX_CFG_NUM_WARPS=16 -DVX_CFG_ISSUE_WIDTH=1` |

### Expt 2: issue-width sweep (WGMMA dense, no FEDP2K, no DXA)

NT=32, NW=32, TCU = WGMMA dense: `-DVX_CFG_TCU_WGMMA_ENABLE` only. There is no `EXT_DXA`, no sparsity and no FEDP2K. WGMMA is additive, so WMMA stays supported.

| Run | NT | NW | IW | TCU blocks | OPCS | Extra defines |
|---|---|---|---|---|---|---|
| E2-IW1 | 32 | 32 | 1 | 1 | 8 | `-DVX_CFG_NUM_THREADS=32 -DVX_CFG_NUM_WARPS=32 -DVX_CFG_ISSUE_WIDTH=1 -DVX_CFG_TCU_WGMMA_ENABLE` |
| E2-IW2 | 32 | 32 | 2 | 2 | 4 | `-DVX_CFG_NUM_THREADS=32 -DVX_CFG_NUM_WARPS=32 -DVX_CFG_ISSUE_WIDTH=2 -DVX_CFG_TCU_WGMMA_ENABLE` |
| E2-IW4 | 32 | 32 | 4 | 4 | 2 | `-DVX_CFG_NUM_THREADS=32 -DVX_CFG_NUM_WARPS=32 -DVX_CFG_ISSUE_WIDTH=4 -DVX_CFG_TCU_WGMMA_ENABLE` |
| E2-IW8 | 32 | 32 | 8 | 8 | 1 | `-DVX_CFG_NUM_THREADS=32 -DVX_CFG_NUM_WARPS=32 -DVX_CFG_ISSUE_WIDTH=8 -DVX_CFG_TCU_WGMMA_ENABLE` |

### Expt 3: TCU feature ladder

NT=32, NW=32, IW=4 (4 TCU blocks, OPCS=2). Base defines for every Expt 3 run: `-DVX_CFG_NUM_THREADS=32 -DVX_CFG_NUM_WARPS=32 -DVX_CFG_ISSUE_WIDTH=4`.

| Run | Label | WGMMA | DXA | 2:4 sparse | FEDP2K | Extra defines (on top of the Expt 3 base) |
|---|---|---|---|---|---|---|
| E3-A | WMMA                      | – | – | – | – | *(none)* |
| E3-B | WMMA + 2:4                | – | – | ✓ | – | `-DVX_CFG_TCU_SPARSE_ENABLE` |
| E3-C | WGMMA + DXA               | ✓ | ✓ | – | – | `-DVX_CFG_TCU_WGMMA_ENABLE -DVX_CFG_EXT_DXA_ENABLE` |
| E3-D | WGMMA + DXA + FEDP2K      | ✓ | ✓ | – | ✓ | `-DVX_CFG_TCU_WGMMA_ENABLE -DVX_CFG_EXT_DXA_ENABLE -DVX_CFG_TCU_FEDP2K` |
| E3-E | WGMMA + DXA + 2:4         | ✓ | ✓ | ✓ | – | `-DVX_CFG_TCU_WGMMA_ENABLE -DVX_CFG_EXT_DXA_ENABLE -DVX_CFG_TCU_SPARSE_ENABLE` |
| E3-F | WGMMA + DXA + 2:4 + FEDP2K | ✓ | ✓ | ✓ | ✓ | `-DVX_CFG_TCU_WGMMA_ENABLE -DVX_CFG_EXT_DXA_ENABLE -DVX_CFG_TCU_SPARSE_ENABLE -DVX_CFG_TCU_FEDP2K` |

Notes:
- FEDP2K widens only the dense WGMMA FEDP (`fedpK = 2·tcK`), so it only appears in WGMMA rows.
- `SPARSE_ENABLE` turns on both `WMMA_SP` and, when WGMMA is on, `WGMMA_SP`.

## Running

Run from a build tree configured with `--xlen=32`. Re-run `../configure` first, as AGENTS.md requires.

```bash
# one-time: an LVT/TT-only library dir (see the flow traps below)
LVT_DIR=$PWD/asap7_lvt_tt
mkdir -p $LVT_DIR
ln -sf /mnt/nas0/eda.libs/asap7/asap7sc7p5t_28/LIB/NLDM/asap7sc7p5t_*_LVT_TT_nldm.db $LVT_DIR/
ls $LVT_DIR      # expect exactly 5: AO, INVBUF, OA, SEQ, SIMPLE

# per run (example: E3-F)
COMMON="-DVX_CFG_NUM_CLUSTERS=1 -DVX_CFG_NUM_CORES=1 -DVX_CFG_EXT_TCU_ENABLE \
        -DVX_CFG_TCU_TF32_ENABLE -DVX_CFG_TCU_FP8_ENABLE -DVX_CFG_TCU_INT8_ENABLE -DVX_CFG_TCU_INT4_ENABLE \
        -DVX_CFG_LMEM_LOG_SIZE=14"
RUN="-DVX_CFG_NUM_THREADS=32 -DVX_CFG_NUM_WARPS=32 -DVX_CFG_ISSUE_WIDTH=4 \
     -DVX_CFG_TCU_WGMMA_ENABLE -DVX_CFG_EXT_DXA_ENABLE -DVX_CFG_TCU_SPARSE_ENABLE -DVX_CFG_TCU_FEDP2K"

CONFIGS="$COMMON $RUN" make -C hw/syn/synopsys synthesis \
    LIB_TYPE=ASAP7 LIB_ROOT=$LVT_DIR \
    SRAM_DB='/mnt/nas0/eda.libs/asap7/ASAP7_SRAM_0p0/generated/LIB/srambank_*_6t122.db' \
    CLOCK_FREQ=1000 OPT_LEVEL=3 \
    TOP_LEVEL_ENTITY=Vortex PREFIX=fs_e3f
```

The 14 runs are independent, so run them in parallel if licences allow.

## Flow traps (read before the first run)

1. **`CONFIGS` must come from the environment**, as described above. `LIB_ROOT`, `SRAM_DB`, `CLOCK_FREQ` and the other knobs go on the make command line.
2. **Use a unique `PREFIX` per run.** The build dir is `<PREFIX>_Vortex`, and `sources.txt` inside it is a file target. Reusing a prefix with different `CONFIGS` silently resynthesizes the *old* elaboration.
3. **Preserve the hierarchy.** `project.tcl` runs `compile_ultra -retime`, which auto-ungroups small modules, and that destroys the Expt 3 breakdown (dont enable it for this expt only)
4. **Hierarchical power.** The vectorless branch of `project.tcl` only writes a flat `report_power`. Have it also write `report_power -hierarchy`, for the per-component power in the Expt 3 breakdown.
5. **SRAM audit.**
   - The ASAP7 `srambank` macros are 1RW only. `gen_sram_wrappers` emits a `VX_sp_ram_asic` wrapper that matches exact `DATAW` and `SIZE ≤ depth`, and emits **no** `VX_dp_ram_asic` wrapper for ASAP7.
   - A RAM that `VX_sp_ram`/`VX_dp_ram` routes to an `*_asic` wrapper (`FORCE_BRAM` and a registered read) has two failure modes. If it is dual-port, or if it is single-port with an unmatched geometry, it either falls through to the `hw/syn/libs/no_mem` blackbox or gets an empty generate branch. Either way it can end up with **zero area and its fan-out optimized away**. RAMs not routed to `*_asic` become flops.
   - Before trusting any number, list every `*_ram_asic` instance with (`DATAW`, `SIZE`, port type, and what it bound to: macro / blackbox / empty), plus the macro count and macro area from `report_area`.
   - If any memory (GPR banks, I$/D$ data/tag, LMEM banks, WGMMA tile buffers, sparse metadata) is unmapped or has zero area, **stop and report back with the audit** before choosing a fallback (blackbox + area estimate vs flops). That choice affects every area percentage.
6. **Area units.** Report µm² as the `.lib` gives them. The 4× factor in the ASAP7 LEFs is a physical-database scale; do not apply it to synthesis area.

## Deliverables

All CSVs go in `evaluation/`, one row per run, with consistent column names across experiments

### 1. Per-experiment results: `fourshadow_asic_expt1.csv`, `fourshadow_asic_expt2.csv`, `fourshadow_asic_expt3.csv`

| Column | Source / definition |
|---|---|
| `run_id`, `label` | E1-NT4 … E3-F, and the label from the tables above |
| `num_threads`, `num_warps`, `issue_width`, `num_tcu_blocks`, `tc_k`, `num_opcs` | config (confirm from the elaboration log) |
| `wgmma`, `dxa`, `sparse`, `fedp2k` | 0/1 |
| `lmem_kb`, `icache_kb`, `dcache_kb` | 16 / 16 / 16 |
| `clock_target_mhz` | 1000 |
| `wns_ps` | worst slack over all paths, signed (positive = met) |
| `wns_reg2reg_ps` | worst register-to-register slack (`report_timing -from [all_registers] -to [all_registers]`). Top-level I/O paths carry an artificial 5% budget. |
| `tns_ps`, `num_violating_paths` | from `qor.rpt` |
| `timing_met` | `wns_ps >= 0` |
| `fmax_est_mhz` | `1e6 / (1000 − wns_reg2reg_ps)`. This is an estimate: when slack is positive, DC stopped optimizing at 1 GHz, so the true Fmax may be higher. |
| `crit_path_start`, `crit_path_end`, `crit_path_unit` | endpoints of the worst reg2reg path, plus the owning unit (e.g. `tcu_core/fedp`, `operands`, `dcache`) |
| `total_cell_area_um2` | `report_area` "Total cell area" (includes macros) |
| `comb_area_um2`, `buf_inv_area_um2`, `seq_area_um2`, `macro_area_um2` | the `report_area` breakdown. Buf/inv is a subset of combinational; keep DC's definitions. |
| `num_cells`, `num_sram_macros` | |
| `core_area_um2`, `tcu_area_um2`, `dxa_area_um2` | hierarchical area of `core`, `core/execute/tcu_unit` and `socket/dxa_core` (0 when absent) |
| `leakage_mw` | static power |
| `internal_mw`, `switching_mw`, `dynamic_mw` | dynamic = internal + switching |
| `total_mw` | |
| `power_mode` | `vectorless` |
| `sram_audit` | `ok`, or a short description of what was unmapped |
| `git_commit`, `dc_version`, `lib` | the commit synthesized, `dc_shell -version`, `asap7sc7p5t_28 LVT TT` |

### 2. Expt 3 subcomponent breakdown: `fourshadow_asic_expt3_breakdown.csv`

Long format, one row per (run, component). Columns:
- `run_id`, `component`, `level` (`leaf` / `rollup`), `hier_paths` (the DC instance paths summed)
- `area_um2`, `pct_of_top`, `pct_of_core`
- `leakage_mw`, `dynamic_mw`

**Leaf rows must partition the `Vortex` top exactly.** They sum to 100% of `total_cell_area_um2`, and the agent should check that the residual is under 0.5%. Components absent from a config (e.g. DXA in E3-A) get a 0 row, so all six runs have the same row set.

Leaf components (paths are relative to `core` = `g_clusters[0].cluster/g_sockets[0].socket/g_cores[0].core` unless they start with `socket/`; DC flattens generate names, e.g. `g_blocks_0__tcu_core`):

| Component | Hierarchy |
|---|---|
| TCU – FEDP datapath | `execute/tcu_unit/g_blocks[*].tcu_core/**/fedp` (all `VX_tcu_fedp_tfr` instances) |
| TCU – sparse metadata + gather | `…/tcu_core/tcu_meta`, `…/tcu_core/**/tcu_sp_mux`, `dual_sparse_mask` |
| TCU – block glue | the rest of each `tcu_core`: operand staging, landing queue, credits, pipe regs |
| TCU – WGMMA tile buffers | `execute/tcu_unit/wgmma/tbuf` (abuf ×blocks + bbuf + LMEM arb). Also give abuf and bbuf as sub-rows if they are separable. |
| TCU – WGMMA control | the rest of `execute/tcu_unit/wgmma` (lockstep, request build) |
| TCU – AGU + lane dispatch/gather | `execute/tcu_unit/agu`, `lane_dispatch`, `lane_gather`, and the remaining `tcu_unit` glue |
| FPU | `execute/fpu_unit` |
| ALU | `execute/alu_unit` |
| SFU | `execute/sfu_unit` |
| LSU | `execute/lsu_unit` + `lsu_scheduler` |
| Register file + operand collectors | `issue/g_slices[*].issue_slice/operands` |
| Issue (ibuffer, scoreboard, dispatcher) | the rest of `issue` |
| Frontend + control | `scheduler`, `fetch`, `decode`, `commit`, `dcr_data`, CSR, `icache_dcr_flush`, and the rest of `core` |
| LMEM (16 KB) | `mem_unit/local_mem` |
| Memory unit (other) | the rest of `mem_unit`: coalescer, lmem switch/adapter, DMA arb, DXA completion detect |
| L1 I$ | `socket/icache` |
| L1 D$ | `socket/dcache` |
| DXA | `socket/dxa_core` |
| Uncore | everything else in `Vortex`: KMU, arbiters, gbar, passthrough L2/L3, socket glue |

Rollup rows: **TCU total** (all TCU leaves), **Core total** (everything under `core`), **L1 caches**, and **memories** (macro area only, summed across the design).

## Sanity checks before reporting

- The elaboration log matches the derived `TC_K`, `OPCS`, `NUM_TCU_BLOCKS` and LMEM bank geometry in the tables.
- There are no GTECH/unmapped cells and no unresolved references (`project.tcl` already dies on GTECH).
- Area is monotonic: E1 increases with NT; E2 increases with IW; E3-A < E3-B; E3-C < E3-D; E3-C < E3-E < E3-F.
