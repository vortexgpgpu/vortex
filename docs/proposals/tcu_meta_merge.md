# Proposal: Merge `VX_tcu_sp_meta` + `VX_tcu_mx_meta` into a unified `VX_tcu_meta`

Status: investigation / pre-implementation
Scope: `hw/rtl/tcu/` (RTL), `sw/kernel/include/vx_tensor.h` (contract note only)
Target: Alveo U55C @ 300 MHz

## 1. Motivation

`VX_tcu_core` instantiates two independent metadata SRAM modules per block:

- `VX_tcu_sp_meta` — sparse (2:4) lane-validity metadata for `WMMA_SP` /
  `WGMMA_SP`. One `VX_dp_ram`, width `PACKED_WIDTH`, per-32b-column write
  enable (`WRENW = TOTAL_COLS`).
- `VX_tcu_mx_meta` — MX scale-factor metadata for MX formats. **Two**
  `VX_dp_ram`s (`meta_a`, `meta_b`), each `TCU_BLOCK_CAP*32` wide, `WRENW = 1`.

That is **three** `VX_dp_ram` instances per `VX_tcu_core`, and `VX_tcu_core` is
itself replicated `BLOCK_SIZE = VX_CFG_NUM_TCU_BLOCKS` times inside
`VX_tcu_unit`. All three RAMs:

- are addressed by warp id (`SIZE = VX_CFG_NUM_WARPS`, identical depth),
- share one broadcast write port from `VX_tcu_agu` (`ext_meta_wr_*`),
- use `OUT_REG=0, RADDR_REG=1` (async-read hint, address registered upstream).

Because they are depth-identical, single-write-port, and already driven by a
single unified write bus, they are natural candidates for a single merged RAM.
This document evaluates merging them into one `VX_tcu_meta` backed by **one**
`VX_dp_ram` with `OUT_REG=1`, and estimates the net U55C cost impact.

## 2. Current structure (as built)

### 2.1 Write side — already unified

`VX_tcu_agu` drives a single broadcast write bus to every block:
`{meta_wr_en, meta_wr_wid, meta_wr_idx[4:0], meta_wr_data}`. In
`VX_tcu_core` the 5-bit slot index is already demuxed into the two modules by
its MSB:

```
sparse_meta_wr_en = ext_meta_wr_en && !ext_meta_wr_idx[4];   // SP namespace
mx_wr_en          = ext_meta_wr_en &&  ext_meta_wr_idx[4];    // MX namespace
mx_wr_axis        = ext_meta_wr_idx[0];                        // MX A vs B
```

`vx_tensor.h` already encodes this in the destination register of the
`TCU_LD` micro-op: `x0/x1` → SP slots (`idx[4]=0`), `x16` → MX-A
(`idx[4]=1, idx[0]=0`), `x17` → MX-B (`idx[4]=1, idx[0]=1`). The namespace
partition is therefore an existing software/hardware contract, not something
this merge introduces.

### 2.2 Read side — independent, same address

Both modules read at `rd_wid = execute_if.data.header.wid`:

- SP: registers `rd_wid`, reads the packed row, then a combinational
  bank mux `vld_block = bank_rdata[bank_sel]` selects the `{step_m,step_k}`
  bank.
- MX: registers `rd_wid`, returns the full `meta_a` / `meta_b` rows; the
  scale extraction (`mx_scale_at`) happens downstream in `VX_tcu_core`.

A single op can read **both** regions in the same cycle: a sparse-MX op
(`WMMA_SP` with an MX format, `mx_is_sparse`) consumes the SP lane mask *and*
the MX scales. Both come from the same `rd_wid`, so a single read port that
returns the full merged width serves every case.

### 2.3 Synthesis path today (Vivado / U55C)

Under `VIVADO` + `!SIMULATION`, `ASYNC_BRAM_PATCH` is defined
(`VX_platform.vh:196`). With `OUT_REG=0, RADDR_REG=1` every one of the three
RAMs is built through `VX_async_ram_patch`. With `RADDR_REG=1` that patch
reduces to a synchronous BRAM block (`SYNC_RAM_WF[_WREN]_BLOCK`) **plus** two
`(* keep="true" *)` `VX_placeholder` buffers on the read-address / read-enable
path that exist purely to pin the registered-address optimization. Those
placeholders and the patch wrapper are instantiated **3 × BLOCK_SIZE** times.

## 3. Proposed design: `VX_tcu_meta`

One module, one `VX_dp_ram`, address = `rd_wid` / `wr_wid`, width = the
concatenation of the three regions, region-granular write enables, `OUT_REG=1`.

```
module VX_tcu_meta import VX_gpu_pkg::*, VX_tcu_pkg::*; #(
    parameter `STRING INSTANCE_ID = ""
) (
    input  wire                                       clk, reset,
    // Unified broadcast write port (from VX_tcu_agu)
    input  wire                                       wr_en,
    input  wire [NW_WIDTH-1:0]                        wr_wid,
    input  wire [4:0]                                 wr_idx,   // [4]=NS, low=slot/axis
    input  wire [TCU_BLOCK_CAP-1:0][`VX_CFG_XLEN-1:0] wr_data,
    // Unified read port
    input  wire [NW_WIDTH-1:0]                        rd_wid,
`ifdef VX_CFG_TCU_SPARSE_ENABLE
    input  wire [3:0]                                 step_m, step_k,
    output wire [TCU_MAX_META_BLOCK_WIDTH-1:0]        vld_block,
`endif
`ifdef VX_CFG_TCU_MX_ENABLE
    output wire [TCU_BLOCK_CAP-1:0][31:0]             meta_a, meta_b,
`endif
);
```

### 3.1 Merged layout

`SIZE = VX_CFG_NUM_WARPS` (unchanged). Data width is the concatenation of the
enabled regions:

| region | present when | width (bits) | write-col granularity |
|--------|--------------|--------------|-----------------------|
| SP packed banks | `TCU_SPARSE_ENABLE` | `TCU_META_PER_WARP_DEPTH * TCU_MAX_META_BLOCK_WIDTH` | 32b column (`TOTAL_COLS` cols) |
| MX-A scales | `TCU_MX_ENABLE` | `TCU_BLOCK_CAP * 32` | 32b column (`TCU_BLOCK_CAP` cols) |
| MX-B scales | `TCU_MX_ENABLE` | `TCU_BLOCK_CAP * 32` | 32b column (`TCU_BLOCK_CAP` cols) |

`DATAW = sum of enabled regions`, uniform `WRENW` at 32-bit column
granularity (`= TOTAL_COLS + 2*TCU_BLOCK_CAP` when both enabled). 32-bit
columns align to the BRAM byte-write-enable lanes, so this is BRAM-friendly.

Per `TCU_LD`, exactly one region is written (`wr_idx[4]` picks SP vs MX, and
within MX `wr_idx[0]` picks A/B); the merged `wren` vector activates only that
region's columns. SP keeps its existing per-column decode
(`packed_wren`); the MX regions assert all of their columns together.

### 3.2 `OUT_REG` and read timing — correction + decision

**Correction to an earlier draft of this doc:** `OUT_REG=1` is **not**
latency-neutral here. The current `OUT_REG=0/RADDR_REG=1` design reads the RAM
**combinationally in the issue cycle**: the read address is the
already-registered `execute_if.data.header.wid`, and in rtlsim it is literally
`assign rdata = ram[raddr]` (0 added latency); under Vivado the
`async_bram_patch` + blackbox `VX_placeholder` absorbs the upstream dispatch
register into the BRAM address register, hitting the same 0 *added* latency on
a real BRAM. The sparse mux / MX-scale extraction consume that data in the
same issue cycle, before `pipe_fedp`.

`OUT_REG=1` registers the address **inside** the RAM, so data arrives one cycle
**after** issue (in both rtlsim and synth). With the operands still consumed at
issue, the metadata is off-by-one and functionally wrong unless compensated.
There are three ways to handle it:

1. **Keep `OUT_REG=0/RADDR_REG=1` (chosen for the first increment).** Merge the
   three RAMs into one while preserving the exact current read timing. Latency-
   neutral, guaranteed rtlsim parity, delivers the BRAM/glue consolidation,
   measurable via the compare synth. Keeps the async patch.
2. `OUT_REG=1`, absorb the read latency by moving the sparse-mux / MX-scale /
   DSM application **after** the existing `pipe_fedp` register (0 added TCU
   latency, native BRAM output register, but a substantial FEDP-input
   restructure across the sparse/MX/WGMMA paths).
3. `OUT_REG=1` with an explicit +1 operand-delay stage and `PIPE_LATENCY+1`
   (simplest correct RTL, native BRAM output register, but +1 TCU latency →
   SimX↔RTL cycle-parity re-validation required).

**Decision:** implement option 1 now. It is the smallest validatable increment
and is the part of the user's idea (the *merge*) that is independent of
`OUT_REG`. The native-BRAM-output-register `OUT_REG=1` variants (options 2/3)
are deferred to a follow-up that includes the pipeline change and SimX
re-validation. No `vx_tensor.h` change is required in either case; the
slot→region encoding is already what the merge consumes (documented there only
as a comment).

## 4. U55C cost analysis

Honest framing up front: the metadata RAMs are **not** the dominant TCU LUT
consumer. From `perf/results/tcu/tcu_synth.csv` (300 MHz):

| design | LUT | FF | WNS (ns) |
|--------|-----|-----|---------|
| tcu | 192,154 | 33,977 | -1.184 |
| tcu_sp | 198,217 | 28,101 | -1.851 |
| tcu_mx | 240,713 | 32,572 | -1.799 |
| tcu_sp_mx | 242,051 | 30,610 | -1.840 |

The `+6k` LUT for SP (`tcu_sp − tcu`) and the `+48k` for MX (`tcu_mx − tcu`)
are dominated by the **datapath** added alongside each feature — the
`VX_tcu_sp_mux` lane selection and, for MX, the `mx_scale_at` extraction +
`FEDP_SF` scale-factor pipeline registers across the `TCU_TC_M × TCU_TC_N`
FEDP grid. The merge does **not** touch any of that. So the LUT savings from
this change are real but bounded, and come from three glue-level sources,
each multiplied by `BLOCK_SIZE`:

1. **Async-patch elimination.** 3 → 1 `VX_async_ram_patch` instances per
   block removed (the merged RAM uses `OUT_REG=1`'s native path), deleting
   their `(* keep *)` `VX_placeholder` buffers and patch glue. ×`BLOCK_SIZE`.
2. **Write-decode / read-output consolidation.** One write decoder and one
   registered read stage instead of three; one set of output nets instead of
   three RAM read ports feeding three muxes.
3. **Partial-BRAM defragmentation.** Three wide-shallow RAMs each round their
   width up to a BRAM-primitive multiple independently; one merged RAM rounds
   once. Worked example (NT=8, sparse int4 — widest meta):
   - `TCU_MAX_ELT_RATIO=8`, `TCU_MAX_META_BLOCK_WIDTH=128`, `NUM_COLS=4`
   - `M_STEPS=2, K_STEPS=4 → PER_WARP_DEPTH=4 → TOTAL_COLS=16 →
     PACKED_WIDTH=512b`
   - MX-A = MX-B = `NT*32 = 256b`
   - Separate: ⌈512/72⌉ + ⌈256/72⌉ + ⌈256/72⌉ = 8+4+4 = **16 RAMB36**
   - Merged: ⌈(512+256+256)/72⌉ = ⌈1024/72⌉ = **15 RAMB36**
   - Saves ~1–2 RAMB36 **per block** (BRAM, not LUT) from rounding recovery.

Net expectation: a **modest** LUT reduction (low thousands, scaling with
`BLOCK_SIZE`), a small BRAM reduction, and — the strongest argument — a
**timing** improvement. Every TCU config above currently **fails** 300 MHz
(WNS −1.18 to −1.85 ns). `OUT_REG=1` puts the read behind the BRAM's native
(free, in-tile) output register and removes the placeholder-pinned async path,
shortening the read-data combinational cone before the SP bank-mux / MX
extraction. This change should be judged primarily on WNS, with LUT/BRAM as
secondary wins.

Do **not** over-claim a large LUT saving in the commit message; the
measurement plan (§6) decides the actual number.

## 5. Why not LUTRAM, and the bigger opportunity

- These RAMs are depth-`NUM_WARPS` (typ. 8–32) — very shallow. `LUTRAM=1`
  would move them to distributed RAM, which **costs** LUTs (≈ DATAW LUTs per
  region per block), the opposite of the goal. Keeping `LUTRAM=0` (BRAM) is
  correct for LUT minimization; the merge keeps BRAM.
- **Cross-block replication is the larger prize, out of scope here.** All
  `BLOCK_SIZE` copies receive *identical* broadcast writes; only the read
  port (per-block `rd_wid`) differs. A single shared `VX_tcu_meta` with
  `BLOCK_SIZE` independent read ports (true multi-read, or read-port banking)
  would cut metadata BRAM/glue by up to `BLOCK_SIZE×`, far exceeding the SP/MX
  merge. The SP/MX merge proposed here is the natural first step and de-risks
  that follow-on by establishing the unified module. Flagged as future work.

## 6. Validation / measurement plan

1. Implement `VX_tcu_meta`; replace the two instances in `VX_tcu_core`.
   Generate-guard the SP and MX regions so each of the four configs
   (`tcu`, `tcu_sp`, `tcu_mx`, `tcu_sp_mx`) builds with only its regions.
2. **rtlsim parity first** (per project doctrine — defer synth until
   rtlsim-green): run `sgemm_tcu_sp`, `sgemm_tcu_mx`, `sgemm_tcu_sp_mx`,
   `sgemm_tcu_wg_sp` to confirm the merged read timing/alignment is
   bit-identical to today.
3. Re-synthesize the `hw/syn/xilinx/dut/tcu` DUT for all four configs with a
   **unique PREFIX per run**; diff LUT/FF/BRAM/WNS against the four-row
   `tcu_synth.csv` baseline above. The merge is justified if WNS improves
   (or holds) with non-increasing LUT and non-increasing BRAM. If LUT or WNS
   regress, do not land it.
4. Update `tcu_synth.csv` with post-merge numbers in the same commit.

## 6a. Measured results (option 1 landed)

Implemented `VX_tcu_meta` (option 1: `OUT_REG=0/RADDR_REG=1`, latency-neutral);
removed `VX_tcu_sp_meta` / `VX_tcu_mx_meta`.

**Functional (rtlsim, DPI FEDP).** 8/8 configs PASS across every geometry axis
the merged RAM touches:
- combined `sp_mx` NT=4 mxfp8 (fp) and NT=8 mxint8 (int) — both regions in one RAM
- `sp` NT=2/8 int4 (widest meta + partial-bank), NT=16 int8 (SYM_SPARSE)
- `mx` NT=4/8 mxfp8/mxint8, NT=16 nvfp4 (multi-scale-factor, `-k64`)

**DUT synthesis (`tcu` target, `-DVX_CFG_TCU_TYPE_DSP -DVX_CFG_TCU_SPARSE_ENABLE`,
U55C @ 300 MHz):**

| metric | baseline (separate) | merged | Δ |
|--------|--------------------:|-------:|--:|
| WNS    | +0.095 ns | **+0.180 ns** | +0.085 ns (better) |
| Top LUT | 16410 | 16407 | −3 |
| Top FF | 824 | 824 | 0 |
| RAMB36 | 5 | 5 | 0 |
| DSP    | 64 | 64 | 0 |
| meta block LUT | 433 | 429 | −4 |

No regression; timing improves slightly even without `OUT_REG=1`. LUT/BRAM are
flat because the SP-only synth exercises a single region — the 3→1 RAM
consolidation benefit only appears in a combined `sp_mx` synth (not run here;
the user scoped the synth comparison to `tcu+dsp+sp`). A follow-up `sp_mx` DUT
synth would quantify the multi-region BRAM/glue saving.

## 7. Recommendation

Proceed to implementation. The change is low-risk (write bus and namespace
encoding are already unified; no software/ABI change), directly targets the
metric that is actually failing (300 MHz timing) via `OUT_REG=1` on the
native BRAM output register, and consolidates three RAM instances + their
async-patch wrappers into one. Treat the LUT saving as modest-and-to-be-
measured rather than the headline, and use this as the stepping stone to the
higher-value cross-block metadata sharing.
