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

/* verilator lint_off UNUSEDSIGNAL */

module VX_tcu_sel import VX_gpu_pkg::*, VX_tcu_pkg::*; #(
    parameter `STRING INSTANCE_ID = "",
    parameter META_ROW_WIDTH = 16,
    parameter I_RATIO        = 4,
    parameter ELT_W          = 8
) (
    input  wire [TCU_TC_K-1:0][31:0] b_col_1,
    input  wire [TCU_TC_K-1:0][31:0] b_col_2,
    input  wire [META_ROW_WIDTH-1:0] vld_meta_row,
    output wire [TCU_TC_K-1:0][31:0] b_col
);
    `UNUSED_SPARAM (INSTANCE_ID);

    for (genvar k = 0; k < TCU_TC_K; ++k) begin : g_bmux

        if (I_RATIO == 4) begin : g_ratio4
            // int8: b_col_1 and b_col_2 are separate 4-element groups
            // Select 2 valid from each group -> 4 output elements (4x8=32 bits)
            wire [I_RATIO-1:0] grp_mask_lo = vld_meta_row[I_RATIO * k              +: I_RATIO];
            wire [I_RATIO-1:0] grp_mask_hi = vld_meta_row[I_RATIO * (TCU_TC_K + k) +: I_RATIO];

            wire [ELT_W-1:0] lo_0 = grp_mask_lo[0] ? b_col_1[k][0*ELT_W +: ELT_W] :
                                     grp_mask_lo[1] ? b_col_1[k][1*ELT_W +: ELT_W] :
                                                      b_col_1[k][2*ELT_W +: ELT_W];
            wire [ELT_W-1:0] lo_1 = grp_mask_lo[3] ? b_col_1[k][3*ELT_W +: ELT_W] :
                                     grp_mask_lo[2] ? b_col_1[k][2*ELT_W +: ELT_W] :
                                                      b_col_1[k][1*ELT_W +: ELT_W];

            wire [ELT_W-1:0] hi_0 = grp_mask_hi[0] ? b_col_2[k][0*ELT_W +: ELT_W] :
                                     grp_mask_hi[1] ? b_col_2[k][1*ELT_W +: ELT_W] :
                                                      b_col_2[k][2*ELT_W +: ELT_W];
            wire [ELT_W-1:0] hi_1 = grp_mask_hi[3] ? b_col_2[k][3*ELT_W +: ELT_W] :
                                     grp_mask_hi[2] ? b_col_2[k][2*ELT_W +: ELT_W] :
                                                      b_col_2[k][1*ELT_W +: ELT_W];

            assign b_col[k] = {hi_1, hi_0, lo_1, lo_0};

        end else if (I_RATIO == 2) begin : g_ratio2
            // fp16: b_col_1 and b_col_2 together form ONE 4-element group
            // Select 2 valid from the combined group -> 2 output elements (2x16=32 bits)
            wire [I_RATIO-1:0] mask_lo = vld_meta_row[I_RATIO * k              +: I_RATIO];
            wire [I_RATIO-1:0] mask_hi = vld_meta_row[I_RATIO * (TCU_TC_K + k) +: I_RATIO];
            wire [3:0] grp_mask = {mask_hi, mask_lo};

            // Pool of 4 fp16 elements across 2 registers
            wire [ELT_W-1:0] elem0 = b_col_1[k][0      +: ELT_W];
            wire [ELT_W-1:0] elem1 = b_col_1[k][ELT_W  +: ELT_W];
            wire [ELT_W-1:0] elem2 = b_col_2[k][0      +: ELT_W];
            wire [ELT_W-1:0] elem3 = b_col_2[k][ELT_W  +: ELT_W];

            // First valid (scan from LSB)
            wire [ELT_W-1:0] sel_0 = grp_mask[0] ? elem0 :
                                      grp_mask[1] ? elem1 :
                                      grp_mask[2] ? elem2 : elem3;

            // Last valid (scan from MSB)
            wire [ELT_W-1:0] sel_1 = grp_mask[3] ? elem3 :
                                      grp_mask[2] ? elem2 :
                                      grp_mask[1] ? elem1 : elem0;

            assign b_col[k] = {sel_1, sel_0};
        end

    end

endmodule

/* verilator lint_on UNUSEDSIGNAL */
