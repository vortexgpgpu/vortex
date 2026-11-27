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

`include "VX_define.vh"

// One processing element of the tensor core's FEDP array: it takes the A row,
// B column and C element its array position selects (and, for a 2:4 sparse
// micro-op, gathers the B column from its two candidate words), registers them
// and computes their dot product. The operand selection stays with the array,
// where each element's fixed position reduces it to a few select bits.
module VX_tcu_pe import VX_gpu_pkg::*, VX_tcu_pkg::*; #(
    parameter `STRING INSTANCE_ID = "",
    // dot-product width in 32-bit words
    parameter FEDP_K       = TCU_WG_FEDP_K,
    parameter FEDP_LATENCY = `VX_CFG_TCU_LATENCY,
    // MX scale factors per operand
    parameter FEDP_SF      = 1,
    parameter ROW_W        = `LOG2UP(TCU_TC_M),
    parameter COL_W        = `LOG2UP(TCU_TC_N)
) (
    input wire                          clk,
    input wire                          reset,

    input wire [4:0]                    fmt_s,
    input wire [4:0]                    fmt_d,

    input wire [FEDP_K-1:0][31:0]       a_row,
    input wire [FEDP_K-1:0][31:0]       b_col_dense,
`ifdef VX_CFG_TCU_SPARSE_ENABLE
    input wire                          is_sparse,
    // the row's candidate selects (VX_tcu_sp_sel) and the column's candidate words
    input wire [TCU_TC_K-1:0][7:0][5:0] sp_sel,
    input wire [TCU_TC_K-1:0][31:0]     b_col_1,
    input wire [TCU_TC_K-1:0][31:0]     b_col_2,
`endif
    input wire [31:0]                   c_val,
`ifdef VX_CFG_TCU_MX_ENABLE
    input wire [FEDP_SF-1:0][7:0]       sf_a,
    input wire [FEDP_SF-1:0][7:0]       sf_b,
`endif

    // read by the debug traces only
    input wire                          array_issue,
    input tcu_header_t                  header,
    input wire [3:0]                    step_m,
    input wire [3:0]                    step_n,
    input wire [ROW_W-1:0]              row_i,
    input wire [COL_W-1:0]              col_j,

    output wire [31:0]                  d_val
);
    `UNUSED_SPARAM (INSTANCE_ID)
    `UNUSED_PARAM (FEDP_SF)
    `UNUSED_VAR ({array_issue, header, step_m, step_n, row_i, col_j})

    wire [FEDP_K-1:0][31:0] b_col;

`ifdef VX_CFG_TCU_SPARSE_ENABLE
    wire [TCU_TC_K-1:0][31:0] b_col_sparse;

    VX_tcu_sp_mux #(
        .INSTANCE_ID (INSTANCE_ID)
    ) tcu_sp_mux (
        .sel       (sp_sel),
        .b_col_in1 (b_col_1),
        .b_col_in2 (b_col_2),
        .b_col_out (b_col_sparse)
    );
    for (genvar k_idx = 0; k_idx < FEDP_K; ++k_idx) begin : g_sparse_b_select
        if (k_idx < TCU_TC_K) begin : g_lo
            assign b_col[k_idx] = is_sparse ? b_col_sparse[k_idx] : b_col_dense[k_idx];
        end else begin : g_hi
            assign b_col[k_idx] = is_sparse ? 32'b0 : b_col_dense[k_idx];
        end
    end

`ifdef VX_TCU_LD_TRACE
    // GATHER trace: GATHER,wid,step_m,step_n,i,k,bword0,bword1,lo,hi,gathered
    // One line per (i, j, k_idx); emitted only for sparse ops.
    always @(posedge clk) begin
        if (array_issue && is_sparse) begin
            for (int kk = 0; kk < TCU_TC_K; ++kk) begin
                $write("GATHER,%0d,%0d,%0d,%0d,%0d,0x%08h,0x%08h,?,?,0x%08h\n",
                    header.wid, step_m, step_n,
                    row_i, int'(col_j)*TCU_TC_K + kk,
                    b_col_1[kk], b_col_2[kk], b_col_sparse[kk]);
            end
        end
    end
`endif
`else
    assign b_col = b_col_dense;
`endif

    // Dual-side sparse lane mask
`ifdef VX_CFG_TCU_TYPE_TFR
    wire [TCU_MAX_INPUTS-1:0] vld_mask_r;
`ifdef VX_CFG_TCU_DSM_ENABLE
    wire [TCU_MAX_INPUTS-1:0] vld_mask;
    VX_tcu_dsm #(
        .N (FEDP_K)
    ) dual_sparse_mask (
        .clk      (clk),
        .reset    (reset),
        .enable   (1'b1),
        .fmt_s    (fmt_s),
        .a_row    (a_row),
        .b_col    (b_col),
        .vld_mask (vld_mask)
    );
    VX_pipe_register #(
        .DATAW (TCU_MAX_INPUTS)
    ) pipe_vld_mask (
        .clk      (clk),
        .reset    (reset),
        .enable   (1'b1),
        .data_in  (vld_mask),
        .data_out (vld_mask_r)
    );
`else
    assign vld_mask_r = '1;
`endif
`endif

    wire [4:0] fmt_s_r, fmt_d_r;
    wire [FEDP_K-1:0][31:0] a_row_r, b_col_r;
`ifdef VX_CFG_TCU_MX_ENABLE
    wire [FEDP_SF-1:0][7:0] sf_a_r, sf_b_r;
`endif
    wire [31:0] c_val_r;

`ifdef VX_CFG_TCU_MX_ENABLE
    VX_pipe_register #(
        .DATAW (32 + 5 + 5 + FEDP_K * 32 + FEDP_K * 32 + 2 * FEDP_SF * 8)
    ) pipe_fedp (
        .clk      (clk),
        .reset    (reset),
        .enable   (1'b1),
        .data_in  ({c_val,   sf_b,   sf_a,   fmt_s,   fmt_d,   b_col,   a_row}),
        .data_out ({c_val_r, sf_b_r, sf_a_r, fmt_s_r, fmt_d_r, b_col_r, a_row_r})
    );
`else
    VX_pipe_register #(
        .DATAW (32 + 5 + 5 + FEDP_K * 32 + FEDP_K * 32)
    ) pipe_fedp (
        .clk      (clk),
        .reset    (reset),
        .enable   (1'b1),
        .data_in  ({c_val,   fmt_s,   fmt_d,   b_col,   a_row}),
        .data_out ({c_val_r, fmt_s_r, fmt_d_r, b_col_r, a_row_r})
    );
`endif

`ifdef VX_CFG_TCU_TYPE_DPI
    VX_tcu_fedp_dpi #(
        .INSTANCE_ID (INSTANCE_ID),
        .LATENCY (FEDP_LATENCY),
        .N (FEDP_K),
        .SF (FEDP_SF)
    ) fedp (
        .clk   (clk),
        .reset (reset),
        .enable(1'b1),
        .fmt_s (fmt_s_r),
        .fmt_d (fmt_d_r),
        .a_row(a_row_r),
        .b_col(b_col_r),
    `ifdef VX_CFG_TCU_MX_ENABLE
        .sf_a  (sf_a_r),
        .sf_b  (sf_b_r),
    `endif
        .c_val (c_val_r),
        .d_val (d_val)
    );
`elsif VX_CFG_TCU_TYPE_BHF
    VX_tcu_fedp_bhf #(
        .INSTANCE_ID (INSTANCE_ID),
        .LATENCY (FEDP_LATENCY),
        .N (FEDP_K)
    ) fedp (
        .clk   (clk),
        .reset (reset),
        .enable(1'b1),
        .fmt_s (fmt_s_r),
        .fmt_d (fmt_d_r),
        .a_row(a_row_r),
        .b_col(b_col_r),
        .c_val (c_val_r),
        .d_val (d_val)
    );
`elsif VX_CFG_TCU_TYPE_FPNEW
    VX_tcu_fedp_fpnew #(
        .INSTANCE_ID (INSTANCE_ID),
        .LATENCY (FEDP_LATENCY),
        .N (FEDP_K)
    ) fedp (
        .clk   (clk),
        .reset (reset),
        .enable(1'b1),
        .fmt_s (fmt_s_r),
        .fmt_d (fmt_d_r),
        .a_row (a_row_r),
        .b_col (b_col_r),
        .c_val (c_val_r),
        .d_val (d_val)
    );
`elsif VX_CFG_TCU_TYPE_TFR
    VX_tcu_fedp_tfr #(
        .INSTANCE_ID (INSTANCE_ID),
        .LATENCY (FEDP_LATENCY),
        .N (FEDP_K),
        .SF (FEDP_SF),
        .USE_DSP (`VX_CFG_TCU_USE_DSP)
    ) fedp (
        .clk   (clk),
        .reset (reset),
        .enable(1'b1),
        .vld_mask(vld_mask_r),
        .fmt_s (fmt_s_r),
        .fmt_d (fmt_d_r),
        .a_row (a_row_r),
        .b_col (b_col_r),
    `ifdef VX_CFG_TCU_MX_ENABLE
        .sf_a  (sf_a_r),
        .sf_b  (sf_b_r),
    `endif
        .c_val (c_val_r),
        .d_val (d_val)
    );
`elsif VX_CFG_TCU_TYPE_DSP
    VX_tcu_fedp_dsp #(
        .INSTANCE_ID (INSTANCE_ID),
        .LATENCY (FEDP_LATENCY),
        .N (FEDP_K)
    ) fedp (
        .clk   (clk),
        .reset (reset),
        .enable(1'b1),
        .fmt_s (fmt_s_r),
        .fmt_d (fmt_d_r),
        .a_row(a_row_r),
        .b_col(b_col_r),
        .c_val (c_val_r),
        .d_val (d_val)
    );
`else
    VX_tcu_fedp_no_implementation_selected fedp ();
`endif

`ifdef DBG_TRACE_TCU
    always @(posedge clk) begin
        if (array_issue) begin
            `TRACE(3, ("%t: %s FEDP-enq: wid=%0d, cta_id=%0d, i=%0d, j=%0d, m=%0d, n=%0d, a_row=", $time, INSTANCE_ID, header.wid, header.cta_id, row_i, col_j, step_m, step_n))
            `TRACE_ARRAY1D(2, "0x%0h", a_row, FEDP_K)
            `TRACE(3, (", b_col="));
            `TRACE_ARRAY1D(2, "0x%0h", b_col, FEDP_K)
            `TRACE(3, (", c_val=0x%0h (#%0d)\n", c_val, header.uuid));
        end
    end
`endif

endmodule
