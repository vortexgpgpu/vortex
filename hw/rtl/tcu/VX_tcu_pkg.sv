// Copyright © 2019-2023
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

`ifndef VX_TCU_PKG_VH
`define VX_TCU_PKG_VH

`include "VX_define.vh"

`IGNORE_UNUSED_BEGIN

package VX_tcu_pkg;

    import VX_gpu_pkg::*;

    // Supported floating-point types
    // WARNING: Changing this list requires updating format utility functions below
    localparam TCU_FP32_ID  = 0;
    localparam TCU_FP16_ID  = 1;
    localparam TCU_BF16_ID  = 2;
    localparam TCU_FP8_ID   = 3;
    localparam TCU_BF8_ID   = 4;
    localparam TCU_TF32_ID  = 5;
    localparam TCU_MXFP8_ID = 6;
    localparam TCU_NVFP4_ID = 7;
    // Supported integer-point types
    localparam TCU_I32_ID   = 8;
    localparam TCU_I8_ID    = 9;
    localparam TCU_U8_ID    = 10;
    localparam TCU_I4_ID    = 11;
    localparam TCU_U4_ID    = 12;
    localparam TCU_MXI8_ID  = 13;
    localparam TCU_FMT_WIDTH= 4;

    // Set configuration parameters
    localparam TCU_NT = `NUM_THREADS;

    localparam TCU_WG_NRA = 4;  // A registers per warp (fixed)
    localparam TCU_WG_NR = 32;  // max NRC (C/D registers, variable via cd_nregs)

    localparam TCU_NR = 8;
    localparam TCU_DK = 0;
    localparam TCU_DP = 0;

    // Tile dimensions
    localparam TCU_TILE_CAP = TCU_NT * TCU_NR;
    localparam TCU_LG_TILE_CAP = $clog2(TCU_TILE_CAP);
    localparam TCU_TILE_EN = TCU_LG_TILE_CAP / 2;
    localparam TCU_TILE_EM = TCU_LG_TILE_CAP - TCU_TILE_EN;

    localparam TCU_TILE_M = 1 << TCU_TILE_EM;
    localparam TCU_TILE_N = 1 << TCU_TILE_EN;
    localparam TCU_TILE_K = (TCU_DK != 0) ? TCU_DK : (TCU_DP != 0) ? TCU_DP : (TCU_TILE_CAP / ((TCU_TILE_M > TCU_TILE_N) ? TCU_TILE_M : TCU_TILE_N));

    // Block dimensions
    localparam TCU_BLOCK_CAP = TCU_NT;
    localparam TCU_LG_BLOCK_CAP = $clog2(TCU_BLOCK_CAP);
    localparam TCU_BLOCK_EN = TCU_LG_BLOCK_CAP / 2;
    localparam TCU_BLOCK_EM = TCU_LG_BLOCK_CAP - TCU_BLOCK_EN;

    localparam TCU_TC_M = 1 << TCU_BLOCK_EM;
    localparam TCU_TC_N = 1 << TCU_BLOCK_EN;
    localparam TCU_TC_K = (TCU_DP != 0) ? TCU_DP : (TCU_BLOCK_CAP / ((TCU_TC_M > TCU_TC_N) ? TCU_TC_M : TCU_TC_N));

    // Step counts
    localparam TCU_M_STEPS = TCU_TILE_M / TCU_TC_M;
    localparam TCU_N_STEPS = TCU_TILE_N / TCU_TC_N;
    localparam TCU_K_STEPS = TCU_TILE_K / TCU_TC_K;

    // A micro-tiling
    localparam TCU_A_BLOCK_SIZE = TCU_TC_M * TCU_TC_K;
    localparam TCU_A_SUB_BLOCKS = TCU_BLOCK_CAP / TCU_A_BLOCK_SIZE;

    // B micro-tiling (dense)
    localparam TCU_B_BLOCK_SIZE = TCU_TC_K * TCU_TC_N;
    localparam TCU_B_SUB_BLOCKS = TCU_BLOCK_CAP / TCU_B_BLOCK_SIZE;

    // WGMMA per-warp tile dimensions (NRA=4 fixed, NRC=NR variable).
    // Derived from block geometry: xtileM = 2*tcM, xtileK = 2*tcK.
    // m_steps = k_steps = 2 always.
    localparam TCU_WG_TILE_M = 2 * TCU_TC_M;
    localparam TCU_WG_TILE_K = 2 * TCU_TC_K;
    localparam TCU_WG_TILE_N = (TCU_WG_NR * TCU_NT) / TCU_WG_TILE_M;

    // WG step counts: block geometry (TC_M/TC_N/TC_K) unchanged, tile is larger
    localparam TCU_WG_M_STEPS = TCU_WG_TILE_M / TCU_TC_M;
    localparam TCU_WG_N_STEPS = TCU_WG_TILE_N / TCU_TC_N;
    localparam TCU_WG_K_STEPS = TCU_WG_TILE_K / TCU_TC_K;

    localparam TCU_WG_UOPS = TCU_WG_M_STEPS * TCU_WG_N_STEPS * TCU_WG_K_STEPS;

    // WG A/B micro-tiling (block geometry is shared with non-WG)
    localparam TCU_WG_A_BLOCK_SIZE = TCU_TC_M * TCU_TC_K;
    localparam TCU_WG_A_SUB_BLOCKS = TCU_BLOCK_CAP / TCU_WG_A_BLOCK_SIZE;

    localparam TCU_WG_B_BLOCK_SIZE = TCU_TC_K * TCU_TC_N;
    localparam TCU_WG_B_SUB_BLOCKS = TCU_BLOCK_CAP / TCU_WG_B_BLOCK_SIZE;

    // Symmetric sparse flag (NT=4, NT=16: block_em == block_en)
    localparam SYM_SPARSE = (TCU_BLOCK_EM == TCU_BLOCK_EN);

    // B micro-tiling (sparse 2:4)
    // NT=8/32: standard interleaved layout (tcK × tcN × 2 = NT lanes per block)
    // NT=16 (SYM_SPARSE): WMMA_SP uses column-pair layout (2 cols × tcK × 2 = NT lanes);
    //   WGMMA_SP needs the full tcK × tcN × 2 candidate lanes (may exceed TCU_BLOCK_CAP).
    localparam TCU_B_BLOCK_SIZE_SP    = SYM_SPARSE ? TCU_BLOCK_CAP : (TCU_TC_K * TCU_TC_N) * 2;
    localparam TCU_B_SUB_BLOCKS_SP    = TCU_BLOCK_CAP / TCU_B_BLOCK_SIZE_SP;
    // WGMMA_SP always needs the full candidate lane set, regardless of SYM_SPARSE.
    localparam TCU_WG_B_BLOCK_SIZE_SP = TCU_TC_K * TCU_TC_N * 2;
    // Width of the tbuf_rs2_data port: wider only when SPARSE is enabled (WGMMA_SP path).
    // Without SPARSE, only TCU_BLOCK_CAP lanes are ever consumed, so keep the port narrow.
`ifdef TCU_SPARSE_ENABLE
    localparam TCU_WG_RS2_WIDTH = TCU_WG_B_BLOCK_SIZE_SP;
`else
    localparam TCU_WG_RS2_WIDTH = TCU_BLOCK_CAP;
`endif

    localparam TCU_MIN_FMT_WIDTH = 4; //int4
    localparam TCU_MAX_ELT_RATIO = 32 / TCU_MIN_FMT_WIDTH;

    // Max metadata widths (sized for widest type: 4-bit elements, I_RATIO=8)
    localparam TCU_MAX_META_ROW_WIDTH   = TCU_TC_K * 2 * TCU_MAX_ELT_RATIO;
    localparam TCU_MAX_META_BLOCK_WIDTH = TCU_NT   * 2 * TCU_MAX_ELT_RATIO;

    // Meta-store micro-op expansion parameters
    localparam TCU_META_PER_WARP_DEPTH = TCU_M_STEPS * (TCU_K_STEPS / 2);
    localparam TCU_META_COLS_PER_LOAD  = (TCU_BLOCK_CAP >= TCU_META_PER_WARP_DEPTH)
        ? (TCU_BLOCK_CAP / TCU_META_PER_WARP_DEPTH) : 1;

    // Partial-bank write parameters (NT < PER_WARP_DEPTH)
    localparam TCU_BANKS_PER_STORE = (TCU_NT < TCU_META_PER_WARP_DEPTH)
        ? TCU_NT : TCU_META_PER_WARP_DEPTH;
    localparam TCU_STORES_PER_COL = (TCU_META_PER_WARP_DEPTH + TCU_NT - 1) / TCU_NT;

    // Register counts
    localparam TCU_NRA = (TCU_TILE_M * TCU_TILE_K) / TCU_NT;
    localparam TCU_NRB = (TCU_TILE_N * TCU_TILE_K) / TCU_NT;
    localparam TCU_NRC = (TCU_TILE_M * TCU_TILE_N) / TCU_NT;

    // Register base addresses
    localparam TCU_RC    = 0;
    localparam TCU_WG_RC = TCU_RC;  // WGMMA C accumulator starts at same base
    localparam TCU_WG_RA = 24;     // WGMMA A register base (fixed f24..f27)
    localparam TCU_RA = 10;
    localparam TCU_RB = (TCU_NRB == 4) ? 28 : 24;

    localparam TCU_UOPS = TCU_M_STEPS * TCU_N_STEPS * TCU_K_STEPS;

    localparam TCU_MAX_INPUTS = TCU_TC_K * TCU_MAX_ELT_RATIO;

    `ifdef TCU_TF32_ENABLE
        localparam TCU_EXP_BITS = 10;
    `elsif TCU_BF16_ENABLE
        localparam TCU_EXP_BITS = 10;
    `else
        localparam TCU_EXP_BITS = 9;
    `endif

    typedef struct packed {
        logic is_zero;
        logic is_sub;
        logic is_inf;
        logic is_nan;
    } fedp_class_t;

    typedef struct packed {
        logic is_inf;
        logic is_nan;
        logic sign;
    } fedp_excep_t;

    function automatic int exp_bits(input int fmt);
        case (fmt)
            TCU_FP32_ID: return 8;
            TCU_FP16_ID: return 5;
            TCU_BF16_ID: return 8;
            TCU_FP8_ID:  return 4;
            TCU_BF8_ID:  return 5;
            TCU_TF32_ID: return 8;
            default:     return 0;
        endcase
    endfunction

    function automatic int sig_bits(input int fmt);
        case (fmt)
            TCU_FP32_ID: return 23;
            TCU_FP16_ID: return 10;
            TCU_BF16_ID: return 7;
            TCU_FP8_ID:  return 3;
            TCU_BF8_ID:  return 2;
            TCU_TF32_ID: return 10;
            default:     return 0;
        endcase
    endfunction

    function automatic int sign_pos(input int fmt);
        case (fmt)
            TCU_FP32_ID: return 31;
            TCU_FP16_ID: return 15;
            TCU_BF16_ID: return 15;
            TCU_FP8_ID:  return 7;
            TCU_BF8_ID:  return 7;
            TCU_TF32_ID: return 18;
            default:     return 0;
        endcase
    endfunction

    function automatic int unsigned tcu_fmt_width(input logic [3:0] fmt);
        case (fmt)
            TCU_FP16_ID, TCU_BF16_ID:
                return 16;
            TCU_NVFP4_ID, TCU_I4_ID, TCU_U4_ID:
                return 4;
            TCU_FP8_ID,
            TCU_BF8_ID,
            TCU_I8_ID,
            TCU_U8_ID,
            TCU_MXFP8_ID,
            TCU_MXI8_ID:
                return 8;
            TCU_FP32_ID,
            TCU_I32_ID,
            TCU_TF32_ID:
                return 32;
            default:
                return 0;
        endcase
    endfunction

    function automatic logic tcu_fmt_is_int(input logic [TCU_FMT_WIDTH-1:0] fmt);
        return fmt[TCU_FMT_WIDTH-1];
    endfunction

    function automatic logic tcu_fmt_is_signed_int(input logic [TCU_FMT_WIDTH-2:0] int_fmt);
        return int_fmt[0];
    endfunction

    function automatic logic tcu_fmt_is_bfloat(input logic [TCU_FMT_WIDTH-2:0] float_fmt);
        return !float_fmt[0];
    endfunction

    // meta_num_cols / meta_total_store_uops are called combinationally from
    // VX_tcu_uops.sv on the ibuffer → uop_data critical path. Writing them
    // as case statements returning compile-time constants keeps Vivado from
    // synthesising the (TCU_BLOCK_CAP + hw - 1) / hw expression as a
    // variable-divisor ceil-divide (which expanded to 9 CARRY8 stages and
    // blew the 300 MHz budget). Each arm is a pure constant, so the whole
    // lookup collapses to a single 4→5-bit LUT.
    function automatic logic [4:0] meta_num_cols(input logic [3:0] fmt);
        case (fmt)
            TCU_FP16_ID, TCU_BF16_ID:
                return 5'((TCU_BLOCK_CAP + 7)  / 8);   // hw = 16/2 = 8
            TCU_FP8_ID, TCU_BF8_ID, TCU_I8_ID, TCU_U8_ID,
            TCU_MXFP8_ID, TCU_MXI8_ID:
                return 5'((TCU_BLOCK_CAP + 3)  / 4);   // hw = 8/2  = 4
            TCU_NVFP4_ID, TCU_I4_ID, TCU_U4_ID:
                return 5'((TCU_BLOCK_CAP + 1)  / 2);   // hw = 4/2  = 2
            TCU_FP32_ID, TCU_I32_ID, TCU_TF32_ID:
                return 5'((TCU_BLOCK_CAP + 15) / 16);  // hw = 32/2 = 16
            default:
                return 5'd0;
        endcase
    endfunction

    function automatic logic [4:0] meta_total_store_uops(input logic [3:0] fmt);
        case (fmt)
            TCU_FP16_ID, TCU_BF16_ID:
                return 5'(((TCU_BLOCK_CAP + 7)  / 8)  * TCU_STORES_PER_COL);
            TCU_FP8_ID, TCU_BF8_ID, TCU_I8_ID, TCU_U8_ID,
            TCU_MXFP8_ID, TCU_MXI8_ID:
                return 5'(((TCU_BLOCK_CAP + 3)  / 4)  * TCU_STORES_PER_COL);
            TCU_NVFP4_ID, TCU_I4_ID, TCU_U4_ID:
                return 5'(((TCU_BLOCK_CAP + 1)  / 2)  * TCU_STORES_PER_COL);
            TCU_FP32_ID, TCU_I32_ID, TCU_TF32_ID:
                return 5'(((TCU_BLOCK_CAP + 15) / 16) * TCU_STORES_PER_COL);
            default:
                return 5'd0;
        endcase
    endfunction

    // Tracing info
`ifdef SIMULATION
    task trace_fmt(input int level, input [3:0] fmt);
        case (fmt)
            TCU_FP32_ID:  `TRACE(level, ("fp32"))
            TCU_FP16_ID:  `TRACE(level, ("fp16"))
            TCU_BF16_ID:  `TRACE(level, ("bf16"))
            TCU_FP8_ID:   `TRACE(level, ("fp8"))
            TCU_BF8_ID:   `TRACE(level, ("bf8"))
            TCU_TF32_ID:  `TRACE(level, ("tf32"))
            TCU_MXFP8_ID: `TRACE(level, ("mxfp8"))
            TCU_NVFP4_ID: `TRACE(level, ("nvfp4"))
            TCU_I32_ID:   `TRACE(level, ("i32"))
            TCU_I8_ID:    `TRACE(level, ("i8"))
            TCU_U8_ID:    `TRACE(level, ("u8"))
            TCU_I4_ID:    `TRACE(level, ("i4"))
            TCU_U4_ID:    `TRACE(level, ("u4"))
            TCU_MXI8_ID:  `TRACE(level, ("mxi8"))
            default:      `TRACE(level, ("?"))
        endcase
    endtask

    task trace_ex_op(input int level,
                     input [INST_OP_BITS-1:0] op_type,
                     input op_args_t op_args
    );
        case (INST_TCU_BITS'(op_type))
            INST_TCU_WMMA: begin
            `ifdef TCU_SPARSE_ENABLE
                `TRACE(level, (op_args.tcu.is_sparse ? "WMMA.SP." : "WMMA."));
            `else
                `TRACE(level, ("WMMA."));
            `endif
                trace_fmt(level, op_args.tcu.fmt_s);
                `TRACE(level, ("."));
                trace_fmt(level, op_args.tcu.fmt_d);
                `TRACE(level, (".%0d.%0d", op_args.tcu.step_m, op_args.tcu.step_n));
            end
        `ifdef TCU_WGMMA_ENABLE
            INST_TCU_WGMMA: begin
            `ifdef TCU_SPARSE_ENABLE
                `TRACE(level, (op_args.tcu.is_sparse ? "WGMMA.SP." : "WGMMA."));
            `else
                `TRACE(level, ("WGMMA."));
            `endif
                trace_fmt(level, op_args.tcu.fmt_s);
                `TRACE(level, ("."));
                trace_fmt(level, op_args.tcu.fmt_d);
                `TRACE(level, (".%0d.%sS.%0d.%0d",
                    (op_args.tcu.cd_nregs == 2'd0) ? 8 : (op_args.tcu.cd_nregs == 2'd1) ? 16 : 32,
                    op_args.tcu.a_from_smem ? "S" : "R",
                    op_args.tcu.step_m, op_args.tcu.step_n));
            end
        `endif
        `ifdef TCU_SPARSE_ENABLE
            INST_TCU_META_STORE: begin
                `TRACE(level, ("META_STORE."));
                trace_fmt(level, op_args.tcu.fmt_s);
            end
        `endif
            default: `TRACE(level, ("?"))
        endcase
    endtask
`endif

    `DECL_EXECUTE_T (tcu, `NUM_TCU_LANES);

endpackage

`IGNORE_UNUSED_END

`endif // VX_TCU_PKG_VH
