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

module VX_tcu_fedp_tet import VX_tcu_pkg::*; #(
    parameter `STRING INSTANCE_ID = "",
    parameter LANE_MASK = 0,
    parameter LATENCY = 0,
    parameter N = TCU_TC_K,
    parameter SF = 1,
    parameter W = 25
) (
    input  wire clk,
    input  wire reset,
    input  wire enable,
    input  wire [TCU_MAX_INPUTS-1:0] vld_mask,
    input  wire [4:0] fmt_s,
    input  wire [4:0] fmt_d,
    input  wire [N-1:0][31:0] a_row,
    input  wire [N-1:0][31:0] b_col,
`ifdef VX_CFG_TCU_MX_ENABLE
    input  wire [SF-1:0][7:0] sf_a,
    input  wire [SF-1:0][7:0] sf_b,
`endif
    input  wire [31:0]        c_val,
    output wire [31:0]        d_val
);
    `UNUSED_SPARAM (INSTANCE_ID)
    `UNUSED_VAR (fmt_d)

`ifndef VX_CFG_TCU_MX_ENABLE
    `UNUSED_PARAM (SF)
`endif

    localparam TCK     = 2 * N;
    localparam EXP_W   = TCU_EXP_BITS;
    localparam EXC_W   = $bits(fedp_excep_t);
    localparam C_HI_W  = 7;
    localparam HR = $clog2(TCK+1);

    localparam ALN_SIG_W = W + 2;
    localparam ACC_SIG_W = W + 1 + HR;

    localparam MUL0_LATENCY = 1;
    localparam MUL1_LATENCY = 1;
    localparam ALN0_LATENCY = 1;
    localparam ALN1_LATENCY = 1;
    localparam ACC0_LATENCY = 1;
    localparam ACC1_LATENCY = 1;
    localparam NRM0_LATENCY = 1;
    localparam NRM1_LATENCY = 1;
    localparam MUL_LATENCY  = MUL0_LATENCY + MUL1_LATENCY;
    localparam ALN_LATENCY  = ALN0_LATENCY + ALN1_LATENCY;
    localparam ACC_LATENCY  = ACC0_LATENCY + ACC1_LATENCY;
    localparam NRM_LATENCY  = NRM0_LATENCY + NRM1_LATENCY;
    localparam TOTAL_LATENCY = MUL_LATENCY + ALN_LATENCY + ACC_LATENCY + NRM_LATENCY;

    `STATIC_ASSERT (LATENCY == 0 || LATENCY == TOTAL_LATENCY,
        ("invalid latency! expected=%0d, actual=%0d", TOTAL_LATENCY, LATENCY))

    localparam S0_IDX = 0;
    localparam S1_IDX = S0_IDX + MUL0_LATENCY;
    localparam S2_IDX = S1_IDX + MUL1_LATENCY;
    localparam S3_IDX = S2_IDX + ALN0_LATENCY;
    localparam S4_IDX = S3_IDX + ALN1_LATENCY;
    localparam S5_IDX = S4_IDX + ACC0_LATENCY;
    localparam S6_IDX = S5_IDX + ACC1_LATENCY;
    localparam S7_IDX = S6_IDX + NRM0_LATENCY;
    localparam S8_IDX = S7_IDX + NRM1_LATENCY;
    `UNUSED_PARAM(S8_IDX)

    reg [TOTAL_LATENCY-1:0] vld_pipe_r;
    reg [TOTAL_LATENCY-1:0][31:0] req_pipe_r;
    reg [31:0] req_id;

    wire vld_any = (|vld_mask) && (LANE_MASK != 0);

    always_ff @(posedge clk) begin
        if (reset) begin
            vld_pipe_r <= '0;
            req_pipe_r <= '0;
            req_id     <= 0;
        end else if (enable) begin
            vld_pipe_r <= {vld_pipe_r[TOTAL_LATENCY-2:0], vld_any};
            req_pipe_r <= {req_pipe_r[TOTAL_LATENCY-2:0], req_id};
            req_id     <= req_id + 32'(vld_any);
        end
    end

    wire [TOTAL_LATENCY:0] vld_pipe = {vld_pipe_r, (~reset && enable && vld_any)};
    wire [TOTAL_LATENCY:0][31:0] req_pipe = {req_pipe_r, req_id};

    // ======================================================================
    // Stage 0/1: Multiply & Diff Matrix
    // ======================================================================

    wire [TCK:0][EXP_W-1:0]   exponents;
    wire [TCK:0]              exp_sel;
    wire [TCK-1:0][TCK-1:0][EXP_W:0] exp_diff_mat;
    wire [TCK:0][W-1:0]       raw_sigs;
    fedp_excep_t              exceptions;
    wire [TCK-1:0]            lane_mask;

    wire is_int = tcu_fmt_is_int(fmt_s);

    wire [7:0] cval_top = c_val[31:24];
    wire [6:0] cval_hi = cval_top[7:1] + 7'(cval_top[0]);
    wire [C_HI_W-1:0] mul_cval_hi;
    wire               mul_is_int;

    VX_tcu_tet_shared_mul #(
        .INSTANCE_ID (INSTANCE_ID),
        .N           (N),
        .W           (W),
        .WA          (ACC_SIG_W),
        .EXP_W       (EXP_W),
        .TCK         (TCK),
        .SF          (SF)
    ) shared_mul (
        .clk          (clk),
        .reset        (reset),
        .enable       (enable),
        .valid_in     (vld_pipe[S0_IDX]),
        .req_id       (req_pipe[S0_IDX]),
        .vld_mask     (vld_mask | TCU_MAX_INPUTS'(LANE_MASK == 0)),
        .fmt_s        (fmt_s),
        .a_row        (a_row),
        .b_col        (b_col),
        .c_val        (c_val),
    `ifdef VX_CFG_TCU_MX_ENABLE
        .sf_a         (sf_a),
        .sf_b         (sf_b),
    `endif
        .exponents    (exponents),
        .exp_sel      (exp_sel),
        .exp_diff_mat (exp_diff_mat),
        .raw_sigs     (raw_sigs),
        .exceptions   (exceptions),
        .lane_mask    (lane_mask)
    );

    VX_tcu_tet_register #(
        .DATAW (C_HI_W + 1),
        .DEPTH (MUL0_LATENCY)
    ) pipe_mul_ctrl (
        .clk      (clk),
        .reset    (reset),
        .enable   (enable),
        .data_in  ({cval_hi,      is_int}),
        .data_out ({mul_cval_hi,  mul_is_int})
    );

    wire [TCK:0][EXP_W-1:0]   s2_exponents;
    wire [TCK:0]              s2_exp_sel;
    wire [TCK-1:0][TCK-1:0][EXP_W:0] s2_exp_diff_mat;
    fedp_excep_t              s2_exceptions;
    wire [TCK-1:0]            s2_lane_mask;
    wire [TCK:0][W-1:0]       s2_raw_sig;
    wire                      s2_is_int;
    wire [C_HI_W-1:0]         s2_cval_hi;

    wire [TCK-1:0][W-1:0] pipe_mul_lane_din, pipe_mul_lane_dout;
    `MAP_AOS_SOA(i, TCK, pipe_mul_lane_din[i], raw_sigs[i])
    `MAP_AOS_SOA(i, TCK, s2_raw_sig[i], pipe_mul_lane_dout[i])

    VX_tcu_tet_pipe_register #(
        .NUM_LANES  (TCK),
        .SHARED_DATAW(((TCK+1)*EXP_W) + (TCK+1) + (TCK*TCK*(EXP_W+1)) + EXC_W + TCK + W + C_HI_W + 1),
        .LANE_DATAW (W),
        .DEPTH      (MUL1_LATENCY),
        .LANE_MASK  (LANE_MASK)
    ) pipe_mul (
        .clk             (clk),
        .reset           (reset),
        .enable          (enable),
        .lane_mask       (lane_mask),
        .shared_data_in  ({exponents,    exp_sel,    exp_diff_mat,    exceptions,    lane_mask,    raw_sigs[TCK],   mul_cval_hi, mul_is_int}),
        .shared_data_out ({s2_exponents, s2_exp_sel, s2_exp_diff_mat, s2_exceptions, s2_lane_mask, s2_raw_sig[TCK], s2_cval_hi, s2_is_int}),
        .lane_data_in    (pipe_mul_lane_din),
        .lane_data_out   (pipe_mul_lane_dout)
    );

    // ======================================================================
    // Stage 2/3: Max Exp & Alignment
    // ======================================================================

    wire [TCK:0][ALN_SIG_W-1:0] s3_aln_sigs;
    wire [TCK:0]                s3_aln_sticky;
    wire [EXP_W-1:0]            s3_max_exp;
    fedp_excep_t                s3_exceptions;
    wire [TCK-1:0]              s3_lane_mask;
    wire                        s3_is_int;
    wire [C_HI_W-1:0]           s3_cval_hi;

    VX_tcu_tet_align #(
        .INSTANCE_ID (INSTANCE_ID),
        .N           (TCK+1),
        .WI          (W),
        .WO          (ALN_SIG_W)
    ) align (
        .clk         (clk),
        .reset       (reset),
        .enable      (enable),
        .valid_in    (vld_pipe[S2_IDX]),
        .req_id      (req_pipe[S2_IDX]),
        .exponents   (s2_exponents),
        .sel_exp     (s2_exp_sel),
        .diff_mat    (s2_exp_diff_mat),
        .sigs_in     (s2_raw_sig),
        .is_int      (s2_is_int),
        .max_exp     (s3_max_exp),
        .sigs_out    (s3_aln_sigs),
        .sticky_bits (s3_aln_sticky)
    );

    VX_tcu_tet_register #(
        .DATAW (EXC_W + TCK + C_HI_W + 1),
        .DEPTH (ALN0_LATENCY)
    ) pipe_aln_ctrl (
        .clk      (clk),
        .reset    (reset),
        .enable   (enable),
        .data_in  ({s2_exceptions, s2_lane_mask, s2_cval_hi, s2_is_int}),
        .data_out ({s3_exceptions, s3_lane_mask, s3_cval_hi, s3_is_int})
    );

    wire [EXP_W-1:0]            s4_max_exp;
    fedp_excep_t                s4_exceptions;
    wire [TCK-1:0]              s4_lane_mask;
    wire [TCK:0][ALN_SIG_W-1:0] s4_aln_sigs;
    wire [TCK:0]                s4_aln_sticky;
    wire                        s4_is_int;
    wire [C_HI_W-1:0]           s4_cval_hi;

    wire [TCK-1:0][(ALN_SIG_W + 1)-1:0] pipe_aln_lane_din, pipe_aln_lane_dout;
    `MAP_AOS_SOA(i, TCK, pipe_aln_lane_din[i], {s3_aln_sigs[i], s3_aln_sticky[i]})
    `MAP_AOS_SOA(i, TCK, {s4_aln_sigs[i], s4_aln_sticky[i]}, pipe_aln_lane_dout[i])

    VX_tcu_tet_pipe_register #(
        .NUM_LANES  (TCK),
        .SHARED_DATAW(EXP_W + EXC_W + TCK + ALN_SIG_W + 1 + C_HI_W + 1),
        .LANE_DATAW (ALN_SIG_W + 1),
        .DEPTH      (ALN1_LATENCY),
        .LANE_MASK  (LANE_MASK)
    ) pipe_aln (
        .clk             (clk),
        .reset           (reset),
        .enable          (enable),
        .lane_mask       (s3_lane_mask),
        .shared_data_in  ({s3_max_exp, s3_exceptions, s3_lane_mask, s3_aln_sigs[TCK], s3_aln_sticky[TCK], s3_cval_hi, s3_is_int}),
        .shared_data_out ({s4_max_exp, s4_exceptions, s4_lane_mask, s4_aln_sigs[TCK], s4_aln_sticky[TCK], s4_cval_hi, s4_is_int}),
        .lane_data_in    (pipe_aln_lane_din),
        .lane_data_out   (pipe_aln_lane_dout)
    );

    // ======================================================================
    // Stage 4/5: Accumulation
    // ======================================================================

    wire [ACC_SIG_W-1:0] s5_acc_sum;
    wire                 s5_acc_sticky;
    wire [EXP_W-1:0]     s5_max_exp;
    fedp_excep_t         s5_exceptions;
    wire                 s5_is_int;
    wire [C_HI_W-1:0]    s5_cval_hi;

    VX_tcu_tet_acc #(
        .INSTANCE_ID (INSTANCE_ID),
        .N           (TCK+1),
        .WI          (ALN_SIG_W),
        .WO          (ACC_SIG_W)
    ) acc (
        .clk        (clk),
        .reset      (reset),
        .enable     (enable),
        .valid_in   (vld_pipe[S4_IDX]),
        .req_id     (req_pipe[S4_IDX]),
        .lane_mask  (s4_lane_mask),
        .is_int     (s4_is_int),
        .sigs_in    (s4_aln_sigs),
        .sticky_in  (s4_aln_sticky),
        .sig_out    (s5_acc_sum),
        .sticky_out (s5_acc_sticky)
    );

    VX_tcu_tet_register #(
        .DATAW (EXP_W + EXC_W + C_HI_W + 1),
        .DEPTH (ACC0_LATENCY)
    ) pipe_acc_ctrl (
        .clk      (clk),
        .reset    (reset),
        .enable   (enable),
        .data_in  ({s4_max_exp, s4_exceptions, s4_cval_hi, s4_is_int}),
        .data_out ({s5_max_exp, s5_exceptions, s5_cval_hi, s5_is_int})
    );

    wire [EXP_W-1:0]      s6_max_exp;
    wire [ACC_SIG_W-1:0]  s6_acc_sum;
    fedp_excep_t          s6_exceptions;
    wire                  s6_acc_sticky;
    wire                  s6_is_int;
    wire [C_HI_W-1:0]     s6_cval_hi;

    VX_tcu_tet_register #(
        .DATAW (EXP_W + ACC_SIG_W + EXC_W + 1 + C_HI_W + 1),
        .DEPTH (ACC1_LATENCY)
    ) pipe_acc (
        .clk      (clk),
        .reset    (reset),
        .enable   (enable),
        .data_in  ({s5_max_exp, s5_acc_sum, s5_exceptions, s5_acc_sticky, s5_cval_hi, s5_is_int}),
        .data_out ({s6_max_exp, s6_acc_sum, s6_exceptions, s6_acc_sticky, s6_cval_hi, s6_is_int})
    );

    // ======================================================================
    // Stage 6/7: Normalization & rounding
    // ======================================================================

    wire [31:0] final_result;

    VX_tcu_tet_norm_round #(
        .INSTANCE_ID (INSTANCE_ID),
        .EXP_W       (EXP_W),
        .C_HI_W      (C_HI_W),
        .WA          (ACC_SIG_W)
    ) norm_round (
        .clk        (clk),
        .reset      (reset),
        .enable     (enable),
        .valid_in   (vld_pipe[S6_IDX]),
        .req_id     (req_pipe[S6_IDX]),
        .max_exp    (s6_max_exp),
        .acc_sig    (s6_acc_sum),
        .sticky_in  (s6_acc_sticky),
        .exceptions (s6_exceptions),
        .cval_hi    (s6_cval_hi),
        .is_int     (s6_is_int),
        .result     (final_result)
    );

    VX_tcu_tet_register #(
        .DATAW (32),
        .DEPTH (NRM1_LATENCY)
    ) pipe_norm_round (
        .clk      (clk),
        .reset    (reset),
        .enable   (enable),
        .data_in  (final_result),
        .data_out (d_val)
    );

`ifdef DBG_TRACE_TCU
    always_ff @(posedge clk) begin
        if (vld_pipe[S0_IDX]) begin
            `TRACE(4, ("%t: %s FEDP-S0(%0d): fmt_s=%0d, a_row=", $time, INSTANCE_ID, req_pipe[S0_IDX], fmt_s));
            `TRACE_ARRAY1D(4, "0x%0h", a_row, N)
            `TRACE(4, (", b_col="))
            `TRACE_ARRAY1D(4, "0x%0h", b_col, N)
            `TRACE(4, (", c_val=0x%0h, vld_mask=%b\n", c_val, vld_mask))
        end
    end

    always_ff @(posedge clk) begin
        if (vld_pipe[S2_IDX]) begin
            `TRACE(4, ("%t: %s FEDP-S2(%0d): is_int=%b, cval_hi=0x%0h, exponents=", $time, INSTANCE_ID, req_pipe[S2_IDX], s2_is_int, s2_cval_hi));
            `TRACE_ARRAY1D(4, "0x%0h", s2_exponents, (TCK+1))
            `TRACE(4, (", raw_sig="))
            `TRACE_ARRAY1D(4, "0x%0h", s2_raw_sig, (TCK+1))
            `TRACE(4, (", exceptions=%0b, lane_mask=%b\n", s2_exceptions, s2_lane_mask))
        end
    end

    always_ff @(posedge clk) begin
        if (vld_pipe[S4_IDX]) begin
            `TRACE(4, ("%t: %s FEDP-S4(%0d): is_int=%b, cval_hi=0x%0h, max_exp=0x%0h, aln_sig=",
                $time, INSTANCE_ID, req_pipe[S4_IDX], s4_is_int, s4_cval_hi, s4_max_exp));
            `TRACE_ARRAY1D(4, "0x%0h", s4_aln_sigs, (TCK+1))
            `TRACE(4, (", sticky_bits="))
            `TRACE_ARRAY1D(4, "0b%b", s4_aln_sticky, (TCK+1))
            `TRACE(4, (", exceptions=%0b\n", s4_exceptions))
        end
    end

    always_ff @(posedge clk) begin
        if (vld_pipe[S6_IDX]) begin
            `TRACE(4, ("%t: %s FEDP-S6(%0d): is_int=%b, cval_hi=0x%0h, acc_sig=0x%0h, max_exp=0x%0h, sticky=%b, exceptions=%0b\n",
                $time, INSTANCE_ID, req_pipe[S6_IDX], s6_is_int, s6_cval_hi, s6_acc_sum, s6_max_exp, s6_acc_sticky, s6_exceptions));
        end
    end

    always_ff @(posedge clk) begin
        if (vld_pipe[S8_IDX]) begin
            `TRACE(4, ("%t: %s FEDP-S8(%0d): result=0x%0h\n", $time, INSTANCE_ID, req_pipe[S8_IDX], d_val));
        end
    end
`endif

endmodule
