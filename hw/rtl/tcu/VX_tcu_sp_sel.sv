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

`ifdef VX_CFG_TCU_SPARSE_ENABLE

// 2:4 structured sparsity: candidate selects of one FEDP row.
// Across the element widths (32/16/8/4 bits), output nibble p of B word k is
// one of six fixed nibbles of the two candidate words b_col_in1[k] and
// b_col_in2[k] (VX_tcu_sp_mux's candidate tables). This module decodes, once
// per row, which candidate each output nibble takes, one-hot; every FEDP of
// the row (all columns) shares the result, so the per-FEDP mux is an AND-OR.
// A cell that computes ROW_SELS rows in turn (VX_CFG_TCU_PE_RATIO) is at
// row ROW_IDX + row_sel * ROW_STRIDE.

module VX_tcu_sp_sel import VX_gpu_pkg::*, VX_tcu_pkg::*; #(
    parameter ROW_IDX = 0,
    parameter ROW_STRIDE = 1,
    parameter ROW_SELS = 1
) (
    input  wire [`LOG2UP(ROW_SELS)-1:0]        row_sel,
    input  wire [4:0]                          fmt_s,
    input  wire [TCU_MAX_META_BLOCK_WIDTH-1:0] vld_mask,
    output logic [TCU_TC_K-1:0][7:0][5:0]      sel
);
    // Per-I_RATIO metadata row widths
    localparam MRW_R1 = TCU_TC_K * 2 * 1;
    localparam MRW_R2 = TCU_TC_K * 2 * 2;
    localparam MRW_R4 = TCU_TC_K * 2 * 4;
    localparam MRW_R8 = TCU_TC_K * 2 * 8;

    wire [MRW_R1-1:0] meta_row_r1;
    wire [MRW_R2-1:0] meta_row_r2;
    wire [MRW_R4-1:0] meta_row_r4;
    wire [MRW_R8-1:0] meta_row_r8;

    if (ROW_SELS == 1) begin : g_row_fixed
        localparam USED_LO = MRW_R8 * ROW_IDX;
        localparam USED_HI = MRW_R8 * (ROW_IDX + 1);
        if (USED_LO > 0) begin : g_vld_mask_lo_unused
            `UNUSED_VAR (vld_mask[USED_LO-1:0])
        end
        if (USED_HI < TCU_MAX_META_BLOCK_WIDTH) begin : g_vld_mask_hi_unused
            `UNUSED_VAR (vld_mask[TCU_MAX_META_BLOCK_WIDTH-1:USED_HI])
        end
        `UNUSED_VAR (row_sel)
        assign meta_row_r1 = vld_mask[MRW_R1 * ROW_IDX +: MRW_R1];
        assign meta_row_r2 = vld_mask[MRW_R2 * ROW_IDX +: MRW_R2];
        assign meta_row_r4 = vld_mask[MRW_R4 * ROW_IDX +: MRW_R4];
        assign meta_row_r8 = vld_mask[MRW_R8 * ROW_IDX +: MRW_R8];
    end else begin : g_row_sel
        wire [31:0] row = ROW_IDX + 32'(row_sel) * ROW_STRIDE;
        `UNUSED_VAR (vld_mask)
        assign meta_row_r1 = vld_mask[MRW_R1 * row +: MRW_R1];
        assign meta_row_r2 = vld_mask[MRW_R2 * row +: MRW_R2];
        assign meta_row_r4 = vld_mask[MRW_R4 * row +: MRW_R4];
        assign meta_row_r8 = vld_mask[MRW_R8 * row +: MRW_R8];
    end
    // For I_RATIO=1 (tf32), only the lo bits [TCU_TC_K-1:0] are used per k-step;
    // the hi bits are the complement and are not needed.
    `UNUSED_VAR (meta_row_r1[MRW_R1-1:TCU_TC_K])

    // Each K word's group masks, per element width.
    wire [TCU_TC_K-1:0]      m1;
    wire [TCU_TC_K-1:0][1:0] m2lo, m2hi;
    wire [TCU_TC_K-1:0][3:0] m4lo, m4hi;
    wire [TCU_TC_K-1:0][7:0] m8lo, m8hi;
    for (genvar k = 0; k < TCU_TC_K; ++k) begin : g_masks
        assign m1[k]   = meta_row_r1[k];
        assign m2lo[k] = meta_row_r2[2 * k +: 2];
        assign m2hi[k] = meta_row_r2[2 * (TCU_TC_K + k) +: 2];
        assign m4lo[k] = meta_row_r4[4 * k +: 4];
        assign m4hi[k] = meta_row_r4[4 * (TCU_TC_K + k) +: 4];
        assign m8lo[k] = meta_row_r8[8 * k +: 8];
        assign m8hi[k] = meta_row_r8[8 * (TCU_TC_K + k) +: 8];
    end

    // Per width, the first kept element of a group takes the lowest set mask
    // bit and the second the highest, with fixed fallbacks for a mask with
    // fewer bits. The width comes from fmt_s, ahead of the metadata: it is
    // decoded apart and AND-ORs the per-width selects, so the metadata path
    // sees no width case.
    wire is_w32 = (tcu_fmt_width(fmt_s) == 32);
    wire is_w16 = (tcu_fmt_width(fmt_s) == 16);
    wire is_w8  = (tcu_fmt_width(fmt_s) == 8);
    wire is_w4  = (tcu_fmt_width(fmt_s) == 4);
    logic [TCU_TC_K-1:0][7:0][5:0] s32, s16, s8, s4;
    always_comb begin
        for (int k = 0; k < TCU_TC_K; ++k) begin
            s32[k][0] = m1[k] ? 6'b000001 : 6'b000010;
            s32[k][1] = m1[k] ? 6'b000001 : 6'b000010;
            s32[k][2] = m1[k] ? 6'b000001 : 6'b000010;
            s32[k][3] = m1[k] ? 6'b000001 : 6'b000010;
            s32[k][4] = m1[k] ? 6'b000001 : 6'b000010;
            s32[k][5] = m1[k] ? 6'b000001 : 6'b000010;
            s32[k][6] = m1[k] ? 6'b000001 : 6'b000010;
            s32[k][7] = m1[k] ? 6'b000001 : 6'b000010;
            s16[k][0] = m2lo[k][0] ? 6'b000001 : m2lo[k][1] ? 6'b000100 : m2hi[k][0] ? 6'b000010 : 6'b001000;
            s16[k][1] = m2lo[k][0] ? 6'b000001 : m2lo[k][1] ? 6'b000100 : m2hi[k][0] ? 6'b000010 : 6'b001000;
            s16[k][2] = m2lo[k][0] ? 6'b000001 : m2lo[k][1] ? 6'b000100 : m2hi[k][0] ? 6'b000010 : 6'b001000;
            s16[k][3] = m2lo[k][0] ? 6'b000001 : m2lo[k][1] ? 6'b000100 : m2hi[k][0] ? 6'b000010 : 6'b001000;
            s16[k][4] = m2hi[k][1] ? 6'b000010 : m2hi[k][0] ? 6'b000100 : m2lo[k][1] ? 6'b000001 : 6'b001000;
            s16[k][5] = m2hi[k][1] ? 6'b000010 : m2hi[k][0] ? 6'b000100 : m2lo[k][1] ? 6'b000001 : 6'b001000;
            s16[k][6] = m2hi[k][1] ? 6'b000010 : m2hi[k][0] ? 6'b000100 : m2lo[k][1] ? 6'b000001 : 6'b001000;
            s16[k][7] = m2hi[k][1] ? 6'b000010 : m2hi[k][0] ? 6'b000100 : m2lo[k][1] ? 6'b000001 : 6'b001000;
            s8[k][0] = m4lo[k][0] ? 6'b000001 : m4lo[k][1] ? 6'b010000 : 6'b000100;
            s8[k][1] = m4lo[k][0] ? 6'b000001 : m4lo[k][1] ? 6'b010000 : 6'b000100;
            s8[k][2] = m4lo[k][3] ? 6'b000100 : m4lo[k][2] ? 6'b010000 : 6'b000001;
            s8[k][3] = m4lo[k][3] ? 6'b000100 : m4lo[k][2] ? 6'b010000 : 6'b000001;
            s8[k][4] = m4hi[k][0] ? 6'b000100 : m4hi[k][1] ? 6'b010000 : 6'b000010;
            s8[k][5] = m4hi[k][0] ? 6'b000100 : m4hi[k][1] ? 6'b010000 : 6'b000010;
            s8[k][6] = m4hi[k][3] ? 6'b000010 : m4hi[k][2] ? 6'b010000 : 6'b000100;
            s8[k][7] = m4hi[k][3] ? 6'b000010 : m4hi[k][2] ? 6'b010000 : 6'b000100;
            s4[k][0] = m8lo[k][0] ? 6'b000001 : m8lo[k][1] ? 6'b100000 : 6'b010000;
            s4[k][1] = m8lo[k][3] ? 6'b010000 : m8lo[k][2] ? 6'b100000 : 6'b000001;
            s4[k][2] = m8lo[k][4] ? 6'b010000 : m8lo[k][5] ? 6'b100000 : 6'b000100;
            s4[k][3] = m8lo[k][7] ? 6'b000100 : m8lo[k][6] ? 6'b100000 : 6'b010000;
            s4[k][4] = m8hi[k][0] ? 6'b000100 : m8hi[k][1] ? 6'b100000 : 6'b010000;
            s4[k][5] = m8hi[k][3] ? 6'b010000 : m8hi[k][2] ? 6'b100000 : 6'b000100;
            s4[k][6] = m8hi[k][4] ? 6'b010000 : m8hi[k][5] ? 6'b100000 : 6'b000010;
            s4[k][7] = m8hi[k][7] ? 6'b000010 : m8hi[k][6] ? 6'b100000 : 6'b010000;
        end
    end
    always_comb begin
        for (int k = 0; k < TCU_TC_K; ++k) begin
            for (int p = 0; p < 8; ++p) begin
                sel[k][p] = ({6{is_w32}} & s32[k][p]) | ({6{is_w16}} & s16[k][p])
                          | ({6{is_w8}} & s8[k][p]) | ({6{is_w4}} & s4[k][p])
                          | {5'b0, ~(is_w32 | is_w16 | is_w8 | is_w4)};
            end
        end
    end

endmodule

`endif // VX_CFG_TCU_SPARSE_ENABLE
