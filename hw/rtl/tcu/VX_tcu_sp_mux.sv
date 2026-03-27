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

//
// 2:4 structured sparsity gather mux.
//
// For a given (i, j) FEDP cell at row ROW_IDX, this module selects
// the sparse B-column elements based on the validity metadata mask.
//
// Three parallel gather datapaths (I_RATIO = 2, 4, 8) are always
// computed; the output mux selects based on fmt_s at runtime.
//
// The vld_mask input is a unified metadata slice: it comes from
// VX_tcu_meta (for WMMA_SP) or from VX_tcu_tile_buf (for WGMMA_SP),
// pre-muxed by VX_tcu_core. This module is format-agnostic w.r.t.
// the metadata source.
//

module VX_tcu_sp_mux import VX_gpu_pkg::*, VX_tcu_pkg::*; #(
    parameter `STRING INSTANCE_ID = "",
    parameter ROW_IDX = 0
) (
    input  wire [3:0]                      fmt_s,
    input  wire [TCU_TC_K-1:0][31:0]       b_col_in1,
    input  wire [TCU_TC_K-1:0][31:0]       b_col_in2,
    input  wire [TCU_MAX_META_BLOCK_WIDTH-1:0] vld_mask,
    output logic [TCU_TC_K-1:0][31:0]      b_col_out
);
    `UNUSED_SPARAM (INSTANCE_ID);

    // Per-I_RATIO metadata row widths
    localparam MRW_R2 = TCU_TC_K * 2 * 2;
    localparam MRW_R4 = TCU_TC_K * 2 * 4;
    localparam MRW_R8 = TCU_TC_K * 2 * 8;
    localparam USED_LO = MRW_R8 * ROW_IDX;
    localparam USED_HI = MRW_R8 * (ROW_IDX + 1);
    if (USED_LO > 0) begin : g_vld_mask_lo_unused
        `UNUSED_VAR (vld_mask[USED_LO-1:0])
    end
    if (USED_HI < TCU_MAX_META_BLOCK_WIDTH) begin : g_vld_mask_hi_unused
        `UNUSED_VAR (vld_mask[TCU_MAX_META_BLOCK_WIDTH-1:USED_HI])
    end

    // Extract metadata row slices for each I_RATIO variant
    wire [MRW_R2-1:0] meta_row_r2 = vld_mask[MRW_R2 * ROW_IDX +: MRW_R2];
    wire [MRW_R4-1:0] meta_row_r4 = vld_mask[MRW_R4 * ROW_IDX +: MRW_R4];
    wire [MRW_R8-1:0] meta_row_r8 = vld_mask[MRW_R8 * ROW_IDX +: MRW_R8];

    // Three parallel gather outputs
    wire [TCU_TC_K-1:0][31:0] b_col_r2, b_col_r4, b_col_r8;

    for (genvar k = 0; k < TCU_TC_K; ++k) begin : g_bmux

        // ---- I_RATIO=2 path (ELT_W=16) ----
        begin : g_r2
            localparam I_R = 2;
            localparam EW  = 16;

            wire [EW-1:0] b1_elems[I_R];
            wire [EW-1:0] b2_elems[I_R];
            for (genvar i = 0; i < I_R; ++i) begin : g_elems_r2
                assign b1_elems[i] = b_col_in1[k][i*EW +: EW];
                assign b2_elems[i] = b_col_in2[k][i*EW +: EW];
            end

            wire [I_R-1:0] mask_lo = meta_row_r2[I_R * k +: I_R];
            wire [I_R-1:0] mask_hi = meta_row_r2[I_R * (TCU_TC_K + k) +: I_R];
            wire [2*I_R-1:0] grp_mask = {mask_hi, mask_lo};

            wire [EW-1:0] sel_0 = grp_mask[0] ? b1_elems[0] :
                                  grp_mask[1] ? b1_elems[1] :
                                  grp_mask[2] ? b2_elems[0] : b2_elems[1];

            wire [EW-1:0] sel_1 = grp_mask[3] ? b2_elems[1] :
                                  grp_mask[2] ? b2_elems[0] :
                                  grp_mask[1] ? b1_elems[1] : b1_elems[0];

            assign b_col_r2[k] = {sel_1, sel_0};
        end

        // ---- I_RATIO=4 path (ELT_W=8) ----
        begin : g_r4
            localparam I_R = 4;
            localparam EW  = 8;

            wire [EW-1:0] b1_elems[I_R];
            wire [EW-1:0] b2_elems[I_R];
            for (genvar i = 0; i < I_R; ++i) begin : g_elems_r4
                assign b1_elems[i] = b_col_in1[k][i*EW +: EW];
                assign b2_elems[i] = b_col_in2[k][i*EW +: EW];
            end

            wire [I_R-1:0] grp_mask_lo = meta_row_r4[I_R * k +: I_R];
            wire [I_R-1:0] grp_mask_hi = meta_row_r4[I_R * (TCU_TC_K + k) +: I_R];

            wire [EW-1:0] lo_0 = grp_mask_lo[0] ? b1_elems[0] :
                                 grp_mask_lo[1] ? b1_elems[1] :
                                                  b1_elems[2];
            wire [EW-1:0] lo_1 = grp_mask_lo[3] ? b1_elems[3] :
                                 grp_mask_lo[2] ? b1_elems[2] :
                                                  b1_elems[1];

            wire [EW-1:0] hi_0 = grp_mask_hi[0] ? b2_elems[0] :
                                 grp_mask_hi[1] ? b2_elems[1] :
                                                  b2_elems[2];
            wire [EW-1:0] hi_1 = grp_mask_hi[3] ? b2_elems[3] :
                                 grp_mask_hi[2] ? b2_elems[2] :
                                                  b2_elems[1];

            assign b_col_r4[k] = {hi_1, hi_0, lo_1, lo_0};
        end

        // ---- I_RATIO=8 path (ELT_W=4) ----
        begin : g_r8
            localparam I_R = 8;
            localparam EW  = 4;

            wire [EW-1:0] b1_elems[I_R];
            wire [EW-1:0] b2_elems[I_R];
            for (genvar i = 0; i < I_R; ++i) begin : g_elems_r8
                assign b1_elems[i] = b_col_in1[k][i*EW +: EW];
                assign b2_elems[i] = b_col_in2[k][i*EW +: EW];
            end

            wire [I_R-1:0] grp_mask_lo = meta_row_r8[I_R * k +: I_R];
            wire [I_R-1:0] grp_mask_hi = meta_row_r8[I_R * (TCU_TC_K + k) +: I_R];

            wire [3:0] sg0_mask = grp_mask_lo[3:0];
            wire [3:0] sg1_mask = grp_mask_lo[7:4];
            wire [3:0] sg2_mask = grp_mask_hi[3:0];
            wire [3:0] sg3_mask = grp_mask_hi[7:4];

            wire [EW-1:0] sg0_0 = sg0_mask[0] ? b1_elems[0] : sg0_mask[1] ? b1_elems[1] : b1_elems[2];
            wire [EW-1:0] sg0_1 = sg0_mask[3] ? b1_elems[3] : sg0_mask[2] ? b1_elems[2] : b1_elems[1];
            wire [EW-1:0] sg1_0 = sg1_mask[0] ? b1_elems[4] : sg1_mask[1] ? b1_elems[5] : b1_elems[6];
            wire [EW-1:0] sg1_1 = sg1_mask[3] ? b1_elems[7] : sg1_mask[2] ? b1_elems[6] : b1_elems[5];

            wire [EW-1:0] sg2_0 = sg2_mask[0] ? b2_elems[0] : sg2_mask[1] ? b2_elems[1] : b2_elems[2];
            wire [EW-1:0] sg2_1 = sg2_mask[3] ? b2_elems[3] : sg2_mask[2] ? b2_elems[2] : b2_elems[1];
            wire [EW-1:0] sg3_0 = sg3_mask[0] ? b2_elems[4] : sg3_mask[1] ? b2_elems[5] : b2_elems[6];
            wire [EW-1:0] sg3_1 = sg3_mask[3] ? b2_elems[7] : sg3_mask[2] ? b2_elems[6] : b2_elems[5];

            assign b_col_r8[k] = {sg3_1, sg3_0, sg2_1, sg2_0, sg1_1, sg1_0, sg0_1, sg0_0};
        end

    end

    // Output mux: select path based on format
    always_comb begin
        automatic int unsigned elt_width = tcu_fmt_width(fmt_s);
        case (elt_width)
            32:      b_col_out = b_col_in1;
            16:      b_col_out = b_col_r2;
            8:       b_col_out = b_col_r4;
            4:       b_col_out = b_col_r8;
            default: b_col_out = 'x;
        endcase
    end

endmodule

`endif // TCU_SPARSE_ENABLE
