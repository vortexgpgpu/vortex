// Copyright 2019-2023
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

`include "VX_define.vh"

`ifdef TCU_SPARSE_ENABLE

/* verilator lint_off UNUSEDSIGNAL */

module VX_tcu_sel import VX_gpu_pkg::*, VX_tcu_pkg::*; #(
    parameter `STRING INSTANCE_ID = "",
    parameter ROW_IDX = 0
) (
    input  wire [3:0]                              fmt_s,
    input  wire [TCU_TC_K-1:0][31:0]               b_col_1,
    input  wire [TCU_TC_K-1:0][31:0]               b_col_2,
    input  wire [TCU_MAX_META_BLOCK_WIDTH-1:0]      vld_meta_block,
    output logic [TCU_TC_K-1:0][31:0]               b_col
);
    `UNUSED_SPARAM (INSTANCE_ID);

    // Per-I_RATIO metadata row widths
    localparam MRW_R2 = TCU_TC_K * 2 * 2;   // I_RATIO=2 (fp16/bf16)
    localparam MRW_R4 = TCU_TC_K * 2 * 4;   // I_RATIO=4 (int8/fp8)
    localparam MRW_R8 = TCU_TC_K * 2 * 8;   // I_RATIO=8 (int4)

    // Extract metadata row slices for each I_RATIO variant
    wire [MRW_R2-1:0] meta_row_r2 = vld_meta_block[MRW_R2 * ROW_IDX +: MRW_R2];
    wire [MRW_R4-1:0] meta_row_r4 = vld_meta_block[MRW_R4 * ROW_IDX +: MRW_R4];
    wire [MRW_R8-1:0] meta_row_r8 = vld_meta_block[MRW_R8 * ROW_IDX +: MRW_R8];

    // Three parallel gather outputs
    wire [TCU_TC_K-1:0][31:0] b_col_r2, b_col_r4, b_col_r8;

    for (genvar k = 0; k < TCU_TC_K; ++k) begin : g_bmux

        // ---- I_RATIO=2 path (fp16/bf16: ELT_W=16) ----
        begin : g_r2
            localparam I_R = 2;
            localparam EW  = 16;
            wire [I_R-1:0] mask_lo = meta_row_r2[I_R * k              +: I_R];
            wire [I_R-1:0] mask_hi = meta_row_r2[I_R * (TCU_TC_K + k) +: I_R];
            wire [3:0] grp_mask = {mask_hi, mask_lo};

            wire [EW-1:0] elem0 = b_col_1[k][0  +: EW];
            wire [EW-1:0] elem1 = b_col_1[k][EW +: EW];
            wire [EW-1:0] elem2 = b_col_2[k][0  +: EW];
            wire [EW-1:0] elem3 = b_col_2[k][EW +: EW];

            wire [EW-1:0] sel_0 = grp_mask[0] ? elem0 :
                                   grp_mask[1] ? elem1 :
                                   grp_mask[2] ? elem2 : elem3;

            wire [EW-1:0] sel_1 = grp_mask[3] ? elem3 :
                                   grp_mask[2] ? elem2 :
                                   grp_mask[1] ? elem1 : elem0;

            assign b_col_r2[k] = {sel_1, sel_0};
        end

        // ---- I_RATIO=4 path (int8/fp8: ELT_W=8) ----
        begin : g_r4
            localparam I_R = 4;
            localparam EW  = 8;
            wire [I_R-1:0] grp_mask_lo = meta_row_r4[I_R * k              +: I_R];
            wire [I_R-1:0] grp_mask_hi = meta_row_r4[I_R * (TCU_TC_K + k) +: I_R];

            wire [EW-1:0] lo_0 = grp_mask_lo[0] ? b_col_1[k][0*EW +: EW] :
                                  grp_mask_lo[1] ? b_col_1[k][1*EW +: EW] :
                                                   b_col_1[k][2*EW +: EW];
            wire [EW-1:0] lo_1 = grp_mask_lo[3] ? b_col_1[k][3*EW +: EW] :
                                  grp_mask_lo[2] ? b_col_1[k][2*EW +: EW] :
                                                   b_col_1[k][1*EW +: EW];

            wire [EW-1:0] hi_0 = grp_mask_hi[0] ? b_col_2[k][0*EW +: EW] :
                                  grp_mask_hi[1] ? b_col_2[k][1*EW +: EW] :
                                                   b_col_2[k][2*EW +: EW];
            wire [EW-1:0] hi_1 = grp_mask_hi[3] ? b_col_2[k][3*EW +: EW] :
                                  grp_mask_hi[2] ? b_col_2[k][2*EW +: EW] :
                                                   b_col_2[k][1*EW +: EW];

            assign b_col_r4[k] = {hi_1, hi_0, lo_1, lo_0};
        end

        // ---- I_RATIO=8 path (int4: ELT_W=4) ----
        begin : g_r8
            localparam I_R = 8;
            localparam EW  = 4;
            wire [I_R-1:0] grp_mask_lo = meta_row_r8[I_R * k              +: I_R];
            wire [I_R-1:0] grp_mask_hi = meta_row_r8[I_R * (TCU_TC_K + k) +: I_R];
            wire [3:0] sg0_mask = grp_mask_lo[3:0];
            wire [3:0] sg1_mask = grp_mask_lo[7:4];
            wire [3:0] sg2_mask = grp_mask_hi[3:0];
            wire [3:0] sg3_mask = grp_mask_hi[7:4];

            wire [EW-1:0] sg0_0 = sg0_mask[0] ? b_col_1[k][0*EW +: EW] :
                                    sg0_mask[1] ? b_col_1[k][1*EW +: EW] :
                                                  b_col_1[k][2*EW +: EW];
            wire [EW-1:0] sg0_1 = sg0_mask[3] ? b_col_1[k][3*EW +: EW] :
                                    sg0_mask[2] ? b_col_1[k][2*EW +: EW] :
                                                  b_col_1[k][1*EW +: EW];
            wire [EW-1:0] sg1_0 = sg1_mask[0] ? b_col_1[k][4*EW +: EW] :
                                    sg1_mask[1] ? b_col_1[k][5*EW +: EW] :
                                                  b_col_1[k][6*EW +: EW];
            wire [EW-1:0] sg1_1 = sg1_mask[3] ? b_col_1[k][7*EW +: EW] :
                                    sg1_mask[2] ? b_col_1[k][6*EW +: EW] :
                                                  b_col_1[k][5*EW +: EW];
            wire [EW-1:0] sg2_0 = sg2_mask[0] ? b_col_2[k][0*EW +: EW] :
                                    sg2_mask[1] ? b_col_2[k][1*EW +: EW] :
                                                  b_col_2[k][2*EW +: EW];
            wire [EW-1:0] sg2_1 = sg2_mask[3] ? b_col_2[k][3*EW +: EW] :
                                    sg2_mask[2] ? b_col_2[k][2*EW +: EW] :
                                                  b_col_2[k][1*EW +: EW];
            wire [EW-1:0] sg3_0 = sg3_mask[0] ? b_col_2[k][4*EW +: EW] :
                                    sg3_mask[1] ? b_col_2[k][5*EW +: EW] :
                                                  b_col_2[k][6*EW +: EW];
            wire [EW-1:0] sg3_1 = sg3_mask[3] ? b_col_2[k][7*EW +: EW] :
                                    sg3_mask[2] ? b_col_2[k][6*EW +: EW] :
                                                  b_col_2[k][5*EW +: EW];

            assign b_col_r8[k] = {sg3_1, sg3_0, sg2_1, sg2_0, sg1_1, sg1_0, sg0_1, sg0_0};
        end

        // ---- Output mux: select path based on fmt_s ----
        always_comb begin
            case (fmt_s)
                TCU_FP16_ID, TCU_BF16_ID:
                    b_col[k] = b_col_r2[k];
                TCU_NVFP4_ID, TCU_I4_ID, TCU_U4_ID:
                    b_col[k] = b_col_r8[k];
                default:
                    b_col[k] = b_col_r4[k];
            endcase
        end

    end

endmodule

/* verilator lint_on UNUSEDSIGNAL */

`endif // TCU_SPARSE_ENABLE
