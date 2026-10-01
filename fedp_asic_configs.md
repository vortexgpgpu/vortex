# FEDP ASIC synthesis configurations (6 runs)

## Top module

**`VX_tcu_fedp_tfr`** (file `hw/rtl/tcu/tfr/VX_tcu_fedp_tfr.sv`). This is a single TFR tensor-core
fused dot-product unit (one FEDP).

Use repo branch `feature_mx`

Using ASAP7 LVT

### Top-level parameters (set them explicitly for every run)

| Parameter | Meaning |
|---|---|
| `N` | Number of 32-bit operand words per FEDP input. `N = TCU_TC_K`, or `2*TCU_TC_K` with FEDP2K. |
| `SF` | Number of microscaling block-scale slots per FEDP. It only matters when `VX_CFG_TCU_MX_ENABLE` is defined; otherwise use 1. |

Leave `LATENCY`, `LANE_MASK`, `USE_DSP`, `W` and `INSTANCE_ID` at their defaults (0, 0, 0, 25, "").
`LATENCY = 0` means "derive from `VX_CFG_TCU_LATENCY`" (4 stages here).

### Ports

`clk, reset, enable, vld_mask[N*8-1:0], fmt_s[4:0], fmt_d[4:0], a_row[N-1:0][31:0], b_col[N-1:0][31:0], c_val[31:0]` -> `d_val[31:0]`.
The ports `sf_a[SF-1:0][7:0]` and `sf_b[SF-1:0][7:0]` exist only when `VX_CFG_TCU_MX_ENABLE` is defined.

## Defines common to all 6 runs

```
+define+SYNTHESIS +define+ASIC
+define+VX_CFG_XLEN=32 +define+VX_CFG_XLEN_32
+define+VX_CFG_EXT_TCU_ENABLE
+define+VX_CFG_TCU_TYPE_TFR
+define+VX_CFG_TCU_USE_DSP=0        (Wallace-tree multipliers; no DSP mapping)
+define+VX_CFG_TCU_LATENCY=4        (4-stage FEDP pipeline)
```

As compiler flags these are `-DSYNTHESIS -DASIC -DVX_CFG_XLEN=32 -DVX_CFG_XLEN_32 -DVX_CFG_EXT_TCU_ENABLE -DVX_CFG_TCU_TYPE_TFR -DVX_CFG_TCU_USE_DSP=0 -DVX_CFG_TCU_LATENCY=4`.

### Format-set defines

- **FP 8-bit group** (fp8, bf8, mxfp8, mxbf8): `-DVX_CFG_TCU_FP16_DISABLE -DVX_CFG_TCU_FP8_ENABLE -DVX_CFG_TCU_MX_ENABLE`
  - fp16/bf16 is enabled by default, so it has to be disabled explicitly. `FP8_ENABLE` gives fp8 (e4m3) and bf8 (e5m2); adding `MX_ENABLE` gives mxfp8 and mxbf8.
- **All formats** (tf32, fp16, bf16, fp8, bf8, mxfp8, mxbf8, mxfp4, nvfp4, int8, int4):
  `-DVX_CFG_TCU_TF32_ENABLE -DVX_CFG_TCU_FP8_ENABLE -DVX_CFG_TCU_MX_ENABLE -DVX_CFG_TCU_FP4_ENABLE -DVX_CFG_TCU_MXFP4_ENABLE -DVX_CFG_TCU_NVFP4_ENABLE -DVX_CFG_TCU_INT8_ENABLE -DVX_CFG_TCU_INT4_ENABLE`
  - fp16/bf16 come from the default-on `VX_CFG_TCU_FP16_ENABLE`. mxfp8/mxbf8 come from `MX_ENABLE` + `FP8_ENABLE`.

## The 6 runs

| Run | NUM_THREADS | Formats | FEDP2K | `N` | `SF` | Extra defines (in addition to the common set) |
|---|---|---|---|---|---|---|
| 1 | 8  | FP 8-bit group | no  | 2 | 1 | `-DVX_CFG_NUM_THREADS=8 -DVX_CFG_TCU_FP16_DISABLE -DVX_CFG_TCU_FP8_ENABLE -DVX_CFG_TCU_MX_ENABLE` |
| 2 | 8  | All            | no  | 2 | 1 | `-DVX_CFG_NUM_THREADS=8 -DVX_CFG_TCU_TF32_ENABLE -DVX_CFG_TCU_FP8_ENABLE -DVX_CFG_TCU_MX_ENABLE -DVX_CFG_TCU_FP4_ENABLE -DVX_CFG_TCU_MXFP4_ENABLE -DVX_CFG_TCU_NVFP4_ENABLE -DVX_CFG_TCU_INT8_ENABLE -DVX_CFG_TCU_INT4_ENABLE` |
| 3 | 32 | FP 8-bit group | no  | 4 | 1 | `-DVX_CFG_NUM_THREADS=32 -DVX_CFG_TCU_FP16_DISABLE -DVX_CFG_TCU_FP8_ENABLE -DVX_CFG_TCU_MX_ENABLE` |
| 4 | 32 | FP 8-bit group | yes | 8 | 1 | `-DVX_CFG_NUM_THREADS=32 -DVX_CFG_TCU_FEDP2K -DVX_CFG_TCU_FP16_DISABLE -DVX_CFG_TCU_FP8_ENABLE -DVX_CFG_TCU_MX_ENABLE` |
| 5 | 32 | All            | no  | 4 | 2 | `-DVX_CFG_NUM_THREADS=32 -DVX_CFG_TCU_TF32_ENABLE -DVX_CFG_TCU_FP8_ENABLE -DVX_CFG_TCU_MX_ENABLE -DVX_CFG_TCU_FP4_ENABLE -DVX_CFG_TCU_MXFP4_ENABLE -DVX_CFG_TCU_NVFP4_ENABLE -DVX_CFG_TCU_INT8_ENABLE -DVX_CFG_TCU_INT4_ENABLE` |
| 6 | 32 | All            | yes | 8 | 4 | `-DVX_CFG_NUM_THREADS=32 -DVX_CFG_TCU_FEDP2K -DVX_CFG_TCU_TF32_ENABLE -DVX_CFG_TCU_FP8_ENABLE -DVX_CFG_TCU_MX_ENABLE -DVX_CFG_TCU_FP4_ENABLE -DVX_CFG_TCU_MXFP4_ENABLE -DVX_CFG_TCU_NVFP4_ENABLE -DVX_CFG_TCU_INT8_ENABLE -DVX_CFG_TCU_INT4_ENABLE` |

There is no NUM_THREADS=8 + FEDP2K run because it is the same N=4 FEDP as run 3 or run 5.

Where the `N` and `SF` values come from:
- **`N`:** `NUM_THREADS=8` gives `TCU_TC_K=2`; `NUM_THREADS=32` gives `TCU_TC_K=4`; FEDP2K doubles it.
- **`SF`:** this is the most scale slots any enabled MX format needs (the package's `TCU_MX_MAX_SF`, i.e. what `VX_tcu_core` passes).
  - FP 8-bit group: MXFP8/MXBF8 use 32-element blocks and an FEDP holds at most N×4 = 32 fp8 elements, so SF = 1 at every N.
  - All formats: NVFP4 uses 16-element blocks over N×8 fp4 elements, giving 1, 2 and 4 slots at N = 2, 4, 8.

All 6 configs elaborate cleanly with Verilator lint at these exact settings.

## Source files

Generated headers come from the TOML config. Run `mkdir build && cd build && ../configure --xlen=32` once in the repo, then use:

```
build/hw/VX_config.vh
build/hw/VX_types.vh
```

The `-D` defines above take precedence over these headers' defaults (every macro there is `ifndef`-guarded), so one generated copy serves all 6 runs.

Include directories: `build/hw`, `hw/rtl`, `hw/rtl/libs`, `hw/rtl/interfaces`, `hw/rtl/tcu`, `hw/rtl/tcu/tfr`.
The headers pulled in are `VX_define.vh`, `VX_platform.vh`, `VX_scope.vh`, `VX_config.vh` and `VX_types.vh`.

RTL files, packages first in this order. This list was extracted from elaborating run 6, the superset; the other runs use a subset of these modules.

```
hw/rtl/VX_gpu_pkg.sv
hw/rtl/tcu/VX_tcu_pkg.sv
hw/rtl/libs/VX_csa_32.sv
hw/rtl/libs/VX_csa_42.sv
hw/rtl/libs/VX_csa_tree.sv
hw/rtl/libs/VX_find_first.sv
hw/rtl/libs/VX_ks_adder.sv
hw/rtl/libs/VX_lzc.sv
hw/rtl/libs/VX_pipe_register.sv
hw/rtl/libs/VX_popcount.sv
hw/rtl/libs/VX_wallace_mul.sv
hw/rtl/tcu/tfr/VX_tcu_tfr_acc.sv
hw/rtl/tcu/tfr/VX_tcu_tfr_align.sv
hw/rtl/tcu/tfr/VX_tcu_tfr_classifier.sv
hw/rtl/tcu/tfr/VX_tcu_tfr_exc_reduce.sv
hw/rtl/tcu/tfr/VX_tcu_tfr_lane_mask.sv
hw/rtl/tcu/tfr/VX_tcu_tfr_max_exp.sv
hw/rtl/tcu/tfr/VX_tcu_tfr_mul_f16.sv
hw/rtl/tcu/tfr/VX_tcu_tfr_mul_f4.sv
hw/rtl/tcu/tfr/VX_tcu_tfr_mul_f8.sv
hw/rtl/tcu/tfr/VX_tcu_tfr_mul_i4.sv
hw/rtl/tcu/tfr/VX_tcu_tfr_mul_i8.sv
hw/rtl/tcu/tfr/VX_tcu_tfr_mul_join.sv
hw/rtl/tcu/tfr/VX_tcu_tfr_norm_round.sv
hw/rtl/tcu/tfr/VX_tcu_tfr_pipe_register.sv
hw/rtl/tcu/tfr/VX_tcu_tfr_shared_mul.sv
hw/rtl/tcu/tfr/VX_tcu_tfr_wmul.sv
hw/rtl/tcu/tfr/VX_tcu_fedp_tfr.sv
```

## Notes for the synthesis setup

- **Clock and reset:** a single clock `clk`, with synchronous `reset` and `enable`. The FEDP is a 4-stage pipeline: multiply, align, accumulate, normalize/round.
- **Lane mask:** `vld_mask` is the dual-side sparse lane mask. With `LANE_MASK=0` (the default) the FEDP does not gate on it. Drive it all-ones, or leave it as an input; tying it to all-ones lets synthesis remove the mask logic.

# Ref from prev run

## Rough estimate from previous run (Ten-Four data)

Prior Ten-Four ASIC results (7 nm, 0.7 V), mapped onto runs 1, 2, 3 and 5. These are estimates to sanity-check the new
Synopsys numbers against, not results from these exact configs.

| Run | NUM_THREADS | Formats | FEDP2K | `N` | Source | Freq. (GHz) | Area (mm²) | Throughput (GFLOPS) |
|---|---|---|---|---|---|---|---|---|
| 1 | 8  | FP 8-bit group | no  | 2 | Ten-Four (Ours) †      | 1.66        | 1.06e-3  | 26.6 |
| 2 | 8  | All            | no  | 2 | Ten-Four (All fmts) †  | 1.58        | 2.40e-3  | 25.2 |
| 3 | 32 | FP 8-bit group | no  | 4 | Ten-Four (Ours) ‡      | 1.55        | 1.93e-3  | 49.6 |
| 5 | 32 | All            | no  | 4 | Ten-Four (All fmts) ‡  | 1.58        | 4.65e-3  | 50.4 |

How the estimates map to the runs:
- **Matching by throughput:** in the Ten-Four data, GFLOPS ÷ GHz = 16 FLOP/cycle for the † rows and 32 FLOP/cycle for the ‡ rows. At 8-bit inputs that is an N=2 FEDP (8 MACs/cycle) and an N=4 FEDP (16 MACs/cycle). So † matches runs 1 and 2 (NUM_THREADS=8), and ‡ matches runs 3 and 5 (NUM_THREADS=32).
- **"Ours" vs "All fmts":** "Ten-Four (Ours)" is the reduced-format FEDP and maps to the FP 8-bit group runs. "Ten-Four (All fmts)" maps to the all-formats runs.
- **Runs 4 and 6 (FEDP2K, N=8) are new configurations** and have no previous result.
