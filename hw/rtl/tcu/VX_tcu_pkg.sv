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

package VX_tcu_pkg;

    import VX_gpu_pkg::*;

    // Set configuration parameters
    localparam TCU_NT = `NUM_THREADS;
    localparam TCU_NR = 8;
    localparam TCU_DP = 0;

    // Supported data types
    localparam TCU_FP32_ID = 0;
    localparam TCU_FP16_ID = 1;
    localparam TCU_BF16_ID = 2;
    localparam TCU_I32_ID  = 8;
    localparam TCU_I8_ID   = 9;
    localparam TCU_U8_ID   = 10;
    localparam TCU_I4_ID   = 11;
    localparam TCU_U4_ID   = 12;

    // Tile dimensions
    localparam TCU_TILE_CAP = TCU_NT * TCU_NR;
    localparam TCU_LG_TILE_CAP = $clog2(TCU_TILE_CAP);
    localparam TCU_TILE_EN = TCU_LG_TILE_CAP / 2;
    localparam TCU_TILE_EM = TCU_LG_TILE_CAP - TCU_TILE_EN;

    localparam TCU_TILE_M = 1 << TCU_TILE_EM;
    localparam TCU_TILE_N = 1 << TCU_TILE_EN;
    localparam TCU_TILE_K = TCU_TILE_CAP / ((TCU_TILE_M > TCU_TILE_N) ? TCU_TILE_M : TCU_TILE_N);

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

    // B micro-tiling
    localparam TCU_B_BLOCK_SIZE = TCU_TC_K * TCU_TC_N;
    localparam TCU_B_SUB_BLOCKS = TCU_BLOCK_CAP / TCU_B_BLOCK_SIZE;

    // Register counts
    //localparam TCU_NRA = (TCU_TILE_M * TCU_TILE_K) / TCU_NT;
    localparam TCU_NRB = (TCU_TILE_N * TCU_TILE_K) / TCU_NT;
    //localparam TCU_NRC = (TCU_TILE_M * TCU_TILE_N) / TCU_NT;

    // Register base addresses
    localparam TCU_RA = 0;
    localparam TCU_RB = (TCU_NRB == 4) ? 28 : 10;
    localparam TCU_RC = (TCU_NRB == 4) ? 10 : 24;

    localparam TCU_UOPS = TCU_M_STEPS * TCU_N_STEPS * TCU_K_STEPS;

    // Tracing info
`ifdef SIMULATION
    task trace_fmt(input int level, input [3:0] fmt);
        case (fmt)
            TCU_FP32_ID: `TRACE(level, ("fp32"))
            TCU_FP16_ID: `TRACE(level, ("fp16"))
            TCU_BF16_ID: `TRACE(level, ("bf16"))
            TCU_I32_ID:  `TRACE(level, ("i32"))
            TCU_I8_ID:   `TRACE(level, ("i8"))
            TCU_U8_ID:   `TRACE(level, ("u8"))
            TCU_I4_ID:   `TRACE(level, ("i4"))
            TCU_U4_ID:   `TRACE(level, ("u4"))
            default:     `TRACE(level, ("?"))
        endcase
    endtask

    task trace_ex_op(input int level,
                     input [INST_OP_BITS-1:0] op_type,
                     input op_args_t op_args
    );
        case (INST_TCU_BITS'(op_type))
            INST_TCU_WMMA: begin
                `TRACE(level, ("WMMA."));
                trace_fmt(level, op_args.tcu.fmt_s);
                `TRACE(level, ("."));
                trace_fmt(level, op_args.tcu.fmt_d);
                `TRACE(level, (".%0d.%0d", op_args.tcu.step_m, op_args.tcu.step_n));
            end
            default: `TRACE(level, ("?"))
        endcase
    endtask
`endif

    `DECL_EXECUTE_T (tcu_exe_t, `NUM_TCU_LANES);
    `DECL_RESULT_T (tcu_res_t, `NUM_TCU_LANES);

endpackage

`endif // VX_TCU_PKG_VH
