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

    // Sparse B-column mux: compress valid elements using 2:4 metadata
    // Per K position: 2 groups of I_RATIO elements -> I_RATIO valid elements
    for (genvar k = 0; k < TCU_TC_K; ++k) begin : g_bmux
        wire [I_RATIO-1:0] grp_mask_lo = vld_meta_row[I_RATIO * k              +: I_RATIO];
        wire [I_RATIO-1:0] grp_mask_hi = vld_meta_row[I_RATIO * (TCU_TC_K + k) +: I_RATIO];

        // Group lo: first 2 valid elements from b_col_1[k]
        wire [ELT_W-1:0] lo_0 = grp_mask_lo[0] ? b_col_1[k][0*ELT_W +: ELT_W] :
                                 grp_mask_lo[1] ? b_col_1[k][1*ELT_W +: ELT_W] :
                                                  b_col_1[k][2*ELT_W +: ELT_W];
        wire [ELT_W-1:0] lo_1 = grp_mask_lo[3] ? b_col_1[k][3*ELT_W +: ELT_W] :
                                 grp_mask_lo[2] ? b_col_1[k][2*ELT_W +: ELT_W] :
                                                  b_col_1[k][1*ELT_W +: ELT_W];

        // Group hi: first 2 valid elements from b_col_2[k]
        wire [ELT_W-1:0] hi_0 = grp_mask_hi[0] ? b_col_2[k][0*ELT_W +: ELT_W] :
                                 grp_mask_hi[1] ? b_col_2[k][1*ELT_W +: ELT_W] :
                                                  b_col_2[k][2*ELT_W +: ELT_W];
        wire [ELT_W-1:0] hi_1 = grp_mask_hi[3] ? b_col_2[k][3*ELT_W +: ELT_W] :
                                 grp_mask_hi[2] ? b_col_2[k][2*ELT_W +: ELT_W] :
                                                  b_col_2[k][1*ELT_W +: ELT_W];

        // Pack 4 valid elements into b_col[k]
        assign b_col[k] = {hi_1, hi_0, lo_1, lo_0};
    end

endmodule
