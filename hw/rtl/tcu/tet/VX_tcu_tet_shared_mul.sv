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

module VX_tcu_tet_shared_mul import VX_tcu_pkg::*;  #(
    parameter `STRING INSTANCE_ID = "",
    parameter N = 2,
    parameter W = 25,
    parameter WA = 28,
    parameter EXP_W = 10,
    parameter TCK = 2 * N,
    parameter SF = 1
) (
    input wire              clk,
    input wire              reset,
    input wire              enable,
    input wire              valid_in,
    input wire [31:0]       req_id,

    input wire [TCU_MAX_INPUTS-1:0] vld_mask,

    input wire [4:0]        fmt_s,

    input wire [N-1:0][31:0] a_row,
    input wire [N-1:0][31:0] b_col,
    input wire [31:0]        c_val,
`ifdef VX_CFG_TCU_MX_ENABLE
    input wire [SF-1:0][7:0] sf_a,
    input wire [SF-1:0][7:0] sf_b,
`endif
    output wire [TCK:0][EXP_W-1:0] exponents,
    output wire [TCK:0]            exp_sel,
    output wire [TCK-1:0][TCK-1:0][EXP_W:0] exp_diff_mat,

    output wire [TCK:0][W-1:0] raw_sigs,
    output wire fedp_excep_t   exceptions,
    output wire [TCK-1:0]      lane_mask
);
    `UNUSED_SPARAM (INSTANCE_ID)

`ifndef VX_CFG_TCU_MX_ENABLE
    `UNUSED_PARAM (SF)
`endif

`ifdef VX_CFG_TCU_FP16_ENABLE
`define TET_MUL_F16_ENABLE
`elsif VX_CFG_TCU_TF32_ENABLE
`define TET_MUL_F16_ENABLE
`endif

`ifdef TET_MUL_F16_ENABLE
    wire [TCK-1:0][24:0]      mul_f16_sig;
    wire [TCK-1:0][EXP_W-1:0] mul_f16_exp;
    fedp_excep_t [TCK-1:0]    mul_f16_exc;

    VX_tcu_tet_mul_f16 #(
        .INSTANCE_ID (INSTANCE_ID),
        .N           (N),
        .TCK         (TCK),
        .W           (W),
        .WA          (WA),
        .EXP_W       (EXP_W)
    ) mul_f16 (
        .clk        (clk),
        .reset      (reset),
        .enable     (enable),
        .valid_in   (valid_in),
        .req_id     (req_id),
        .vld_mask   (vld_mask),
        .fmt_f      (fmt_s[3:0]),
        .a_row      (a_row),
        .b_col      (b_col),
        .result_sig (mul_f16_sig),
        .result_exp (mul_f16_exp),
        .exceptions (mul_f16_exc)
    );

    wire [TCK-1:0][24:0]      s1_mul_f16_sig = mul_f16_sig;
    wire [TCK-1:0][EXP_W-1:0] s1_mul_f16_exp = mul_f16_exp;
    fedp_excep_t [TCK-1:0]    s1_mul_f16_exc;
    assign s1_mul_f16_exc = mul_f16_exc;
`endif

`ifdef VX_CFG_TCU_FP8_ENABLE
    wire [TCK-1:0][24:0]      mul_f8_sig;
    wire [TCK-1:0][EXP_W-1:0] mul_f8_exp;
    fedp_excep_t [TCK-1:0]    mul_f8_exc;

    wire [SF-1:0][TCK-1:0][24:0]      mul_f8_sig_s;
    wire [SF-1:0][TCK-1:0][EXP_W-1:0] mul_f8_exp_s;
    fedp_excep_t [SF-1:0][TCK-1:0]    mul_f8_exc_s;

    for (genvar s = 0; s < SF; ++s) begin : g_mul_f8_sf
        VX_tcu_tet_mul_f8 #(
            .INSTANCE_ID (INSTANCE_ID),
            .N           (N),
            .TCK         (TCK),
            .W           (W),
            .WA          (WA),
            .EXP_W       (EXP_W)
        ) mul_f8 (
            .reset      (reset),
            .enable     (enable),
            .clk        (clk),
            .valid_in   (valid_in),
            .req_id     (req_id),
            .vld_mask   (vld_mask),
            .fmt_f      (fmt_s[3:0]),
            .a_row      (a_row),
            .b_col      (b_col),
        `ifdef VX_CFG_TCU_MX_ENABLE
            .sf_a       (sf_a[s]),
            .sf_b       (sf_b[s]),
        `endif
            .result_sig (mul_f8_sig_s[s]),
            .result_exp (mul_f8_exp_s[s]),
            .exceptions (mul_f8_exc_s[s])
        );
    end

    for (genvar i = 0; i < TCK; ++i) begin : g_mul_f8_lane
        localparam SF_SLOT = (i * SF) / TCK;
        assign mul_f8_sig[i] = mul_f8_sig_s[SF_SLOT][i];
        assign mul_f8_exp[i] = mul_f8_exp_s[SF_SLOT][i];
        assign mul_f8_exc[i] = mul_f8_exc_s[SF_SLOT][i];
    end

    wire [TCK-1:0][24:0]      s1_mul_f8_sig      = mul_f8_sig;
    wire [TCK-1:0][EXP_W-1:0] s1_mul_f8_exp      = mul_f8_exp;
    fedp_excep_t [TCK-1:0]    s1_mul_f8_exc;
    assign s1_mul_f8_exc = mul_f8_exc;
`endif

`ifdef VX_CFG_TCU_MX_ENABLE
`ifdef VX_CFG_TCU_FP4_ENABLE
    wire [TCK-1:0][24:0]      mul_f4_sig;
    wire [TCK-1:0][EXP_W-1:0] mul_f4_exp;
    fedp_excep_t [TCK-1:0]    mul_f4_exc;

    wire [SF-1:0][TCK-1:0][24:0]      mul_f4_sig_s;
    wire [SF-1:0][TCK-1:0][EXP_W-1:0] mul_f4_exp_s;
    fedp_excep_t [SF-1:0][TCK-1:0]    mul_f4_exc_s;

    for (genvar s = 0; s < SF; ++s) begin : g_mul_f4_sf
        VX_tcu_tet_mul_f4 #(
            .INSTANCE_ID (INSTANCE_ID),
            .N           (N),
            .TCK         (TCK),
            .W           (W),
            .WA          (WA),
            .EXP_W       (EXP_W)
        ) mul_f4 (
            .clk        (clk),
            .valid_in   (valid_in),
            .req_id     (req_id),
            .vld_mask   (vld_mask),
            .fmt_f      (fmt_s),
            .a_row      (a_row),
            .b_col      (b_col),
            .sf_a       (sf_a[s]),
            .sf_b       (sf_b[s]),
            .result_sig (mul_f4_sig_s[s]),
            .result_exp (mul_f4_exp_s[s]),
            .exceptions (mul_f4_exc_s[s])
        );
    end

    for (genvar i = 0; i < TCK; ++i) begin : g_mul_f4_lane
        localparam SF_SLOT = (i * SF) / TCK;
        assign mul_f4_sig[i] = mul_f4_sig_s[SF_SLOT][i];
        assign mul_f4_exp[i] = mul_f4_exp_s[SF_SLOT][i];
        assign mul_f4_exc[i] = mul_f4_exc_s[SF_SLOT][i];
    end

    wire [TCK-1:0][24:0]      s1_mul_f4_sig;
    wire [TCK-1:0][EXP_W-1:0] s1_mul_f4_exp;
    fedp_excep_t [TCK-1:0]    s1_mul_f4_exc;

    VX_tcu_tet_register #(
        .DATAW ((TCK * 25) + (TCK * EXP_W) + (TCK * $bits(fedp_excep_t))),
        .DEPTH (1)
    ) pipe_mul_f4 (
        .clk      (clk),
        .reset    (reset),
        .enable   (enable),
        .data_in  ({mul_f4_sig,    mul_f4_exp,    mul_f4_exc}),
        .data_out ({s1_mul_f4_sig, s1_mul_f4_exp, s1_mul_f4_exc})
    );
`endif
`endif

`ifdef VX_CFG_TCU_INT8_ENABLE
    wire [TCK-1:0][24:0] mul_int8_sig;
    wire [SF-1:0][TCK-1:0][24:0] mul_int8_sig_s;
    for (genvar s = 0; s < SF; ++s) begin : g_mul_i8_sf
        VX_tcu_tet_mul_i8 #(
            .INSTANCE_ID (INSTANCE_ID),
            .N           (N),
            .TCK         (TCK)
        ) mul_int8 (
            .reset      (reset),
            .enable     (enable),
            .clk        (clk),
            .valid_in   (valid_in),
            .req_id     (req_id),
            .vld_mask   (vld_mask),
            .fmt_i      (fmt_s[3:0]),
            .a_row      (a_row),
            .b_col      (b_col),
        `ifdef VX_CFG_TCU_MX_ENABLE
            .sf_a       (sf_a[s]),
            .sf_b       (sf_b[s]),
        `endif
            .result     (mul_int8_sig_s[s])
        );
    end

    for (genvar i = 0; i < TCK; ++i) begin : g_mul_i8_lane
        localparam SF_SLOT = (i * SF) / TCK;
        assign mul_int8_sig[i] = mul_int8_sig_s[SF_SLOT][i];
    end

    wire [TCK-1:0][24:0] s1_mul_int8_sig = mul_int8_sig;
`endif

`ifdef VX_CFG_TCU_INT4_ENABLE
    wire [TCK-1:0][24:0] mul_int4_sig;
    VX_tcu_tet_mul_i4 #(
        .INSTANCE_ID (INSTANCE_ID),
        .N           (N),
        .TCK         (TCK)
    ) mul_int4 (
        .clk      (clk),
        .valid_in (valid_in),
        .req_id   (req_id),
        .vld_mask (vld_mask),
        .fmt_i    (fmt_s[3:0]),
        .a_row    (a_row),
        .b_col    (b_col),
        .result   (mul_int4_sig)
    );

    wire [TCK-1:0][24:0] s1_mul_int4_sig;
    VX_tcu_tet_register #(
        .DATAW (TCK * 25),
        .DEPTH (1)
    ) pipe_mul_i4 (
        .clk      (clk),
        .reset    (reset),
        .enable   (enable),
        .data_in  (mul_int4_sig),
        .data_out (s1_mul_int4_sig)
    );
`endif

    wire                      s1_valid;
    wire [31:0]               s1_req_id;
    wire [TCU_MAX_INPUTS-1:0] s1_vld_mask;
    wire [4:0]                s1_fmt_s;
    wire [31:0]               s1_c_val;

    VX_tcu_tet_register #(
        .DATAW (1 + 32 + TCU_MAX_INPUTS + 5 + 32),
        .DEPTH (1)
    ) pipe_mul_ctrl (
        .clk      (clk),
        .reset    (reset),
        .enable   (enable),
        .data_in  ({valid_in,  req_id,    vld_mask,    fmt_s,    c_val}),
        .data_out ({s1_valid,  s1_req_id, s1_vld_mask, s1_fmt_s, s1_c_val})
    );

    fedp_excep_t [TCK:0] join_exceptions;

    VX_tcu_tet_mul_join #(
        .INSTANCE_ID (INSTANCE_ID),
        .N           (N),
        .TCK         (TCK),
        .W           (W),
        .WA          (WA),
        .EXP_W       (EXP_W)
    ) join_stage (
        .clk      (clk),
        .valid_in (s1_valid),
        .req_id   (s1_req_id),
        .fmt_s    (s1_fmt_s),
        .c_val    (s1_c_val),

    `ifdef TET_MUL_F16_ENABLE
        .sig_f16  (s1_mul_f16_sig),
        .exp_f16  (s1_mul_f16_exp),
        .exc_f16  (s1_mul_f16_exc),
    `endif

    `ifdef VX_CFG_TCU_FP8_ENABLE
        .sig_f8   (s1_mul_f8_sig),
        .exp_f8   (s1_mul_f8_exp),
        .exc_f8   (s1_mul_f8_exc),
    `endif

    `ifdef VX_CFG_TCU_MX_ENABLE
    `ifdef VX_CFG_TCU_FP4_ENABLE
        .sig_f4   (s1_mul_f4_sig),
        .exp_f4   (s1_mul_f4_exp),
        .exc_f4   (s1_mul_f4_exc),
    `endif
    `endif

    `ifdef VX_CFG_TCU_INT8_ENABLE
        .sig_int8 (s1_mul_int8_sig),
    `endif
    `ifdef VX_CFG_TCU_INT4_ENABLE
        .sig_int4 (s1_mul_int4_sig),
    `endif

        .sig_out  (raw_sigs),
        .exp_out  (exponents),
        .exc_out  (join_exceptions)
    );

    VX_tcu_tet_exc_reduce #(
        .TCK (TCK)
    ) exc_reduce (
        .exc_in  (join_exceptions),
        .exc_out (exceptions)
    );

    VX_tcu_tet_max_exp #(
        .N     (TCK+1),
        .WIDTH (EXP_W)
    ) find_diff_mat (
        .exponents (exponents),
        .sel_exp   (exp_sel),
        .diff_mat  (exp_diff_mat)
    );

    VX_tcu_tet_lane_mask #(
        .N   (N),
        .TCK (TCK)
    ) lane_mask_inst (
        .vld_mask  (s1_vld_mask),
        .fmt_s     (s1_fmt_s),
        .lane_mask (lane_mask)
    );

endmodule

`ifdef TET_MUL_F16_ENABLE
`undef TET_MUL_F16_ENABLE
`endif
