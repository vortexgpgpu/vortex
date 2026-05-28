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

module VX_tcu_tfr_mul_join import VX_tcu_pkg::*; #(
    parameter `STRING INSTANCE_ID = "",
    parameter N     = 2,
    parameter TCK   = 2 * N,
    parameter W     = 25,
    parameter WA    = 28,
    parameter EXP_W = 10
) (
    input wire                      clk,
    input wire                      valid_in,
    input wire [31:0]               req_id,

    input wire [4:0]                fmt_s,
    input wire [31:0]               c_val,

`ifdef TCU_FP16_ENABLE
`define TFR_JOIN_F16_ENABLE
`elsif TCU_TF32_ENABLE
`define TFR_JOIN_F16_ENABLE
`endif

`ifdef TFR_JOIN_F16_ENABLE
    input wire [TCK-1:0][24:0]      sig_f16,
    input wire [TCK-1:0][EXP_W-1:0] exp_f16,
    input fedp_excep_t [TCK-1:0]    exc_f16,
`endif

`ifdef TCU_FP8_ENABLE
    input wire [TCK-1:0][24:0]      sig_f8,
    input wire [TCK-1:0][EXP_W-1:0] exp_f8,
    input fedp_excep_t [TCK-1:0]    exc_f8,
`endif

`ifdef TCU_MX_ENABLE
`ifdef TCU_FP4_ENABLE
    input wire [TCK-1:0][24:0]      sig_f4,
    input wire [TCK-1:0][EXP_W-1:0] exp_f4,
    input fedp_excep_t [TCK-1:0]    exc_f4,
`endif
`endif

`ifdef TCU_INT8_ENABLE
    input wire [TCK-1:0][24:0]      sig_int8,
`endif
`ifdef TCU_INT4_ENABLE
    input wire [TCK-1:0][24:0]      sig_int4,
`endif

    output logic [TCK:0][24:0]      sig_out,
    output logic [TCK:0][EXP_W-1:0] exp_out,
    output fedp_excep_t [TCK:0]     exc_out
);
    `UNUSED_SPARAM (INSTANCE_ID)
    `UNUSED_VAR ({clk, req_id, valid_in})

    logic [TCK-1:0][24:0]      sig_sel;
    logic [TCK-1:0][EXP_W-1:0] exp_sel;
    fedp_excep_t [TCK-1:0]     exc_sel;

    // Path selection
    always_comb begin
        case (fmt_s)
        `ifdef TFR_JOIN_F16_ENABLE
        `ifdef TCU_FP16_ENABLE
            TCU_FP16_ID,
            TCU_BF16_ID: begin
                sig_sel = sig_f16;
                exp_sel = exp_f16;
                exc_sel = exc_f16;
            end
        `endif
        `ifdef TCU_TF32_ENABLE
            TCU_TF32_ID: begin
                sig_sel = sig_f16;
                exp_sel = exp_f16;
                exc_sel = exc_f16;
            end
        `endif
        `endif

        `ifdef TCU_FP8_ENABLE
            TCU_FP8_ID, TCU_BF8_ID
        `ifdef TCU_MX_ENABLE
            , TCU_MXFP8_ID, TCU_MXBF8_ID
        `endif
            : begin
                sig_sel = sig_f8;
                exp_sel = exp_f8;
                exc_sel = exc_f8;
            end
        `endif

        `ifdef TCU_MX_ENABLE
        `ifdef TCU_FP4_ENABLE
        `ifdef TCU_MXFP4_ENABLE
            TCU_MXFP4_ID: begin
                sig_sel = sig_f4;
                exp_sel = exp_f4;
                exc_sel = exc_f4;
            end
        `endif
        `ifdef TCU_NVFP4_ENABLE
            TCU_NVFP4_ID: begin
                sig_sel = sig_f4;
                exp_sel = exp_f4;
                exc_sel = exc_f4;
            end
        `endif
        `endif
        `endif

        `ifdef TCU_INT8_ENABLE
            TCU_I8_ID, TCU_U8_ID
        `ifdef TCU_MX_ENABLE
            , TCU_MXI8_ID
        `endif
            : begin
                sig_sel = sig_int8;
                exp_sel = '0;
                exc_sel = '0;
            end
        `endif
        `ifdef TCU_INT4_ENABLE
            TCU_I4_ID, TCU_U4_ID: begin
                sig_sel = sig_int4;
                exp_sel = '0;
                exc_sel = '0;
            end
        `endif
            default: begin
                sig_sel = '0;
                exp_sel = '0;
                exc_sel = '0;
            end
        endcase
    end

    wire c_is_int = tcu_fmt_is_int(fmt_s);

    // C-term handling
    fedp_class_t cls_c;
    VX_tcu_tfr_classifier #(
        .EXP_W (8),
        .MAN_W (23)
    ) class_c (
        .exp (c_val[30:23]),
        .man (c_val[22:0]),
        .cls (cls_c)
    );

    wire c_sign = c_val[31];

    wire [24:0] c_sig_final = c_is_int ? c_val[24:0] : (cls_c.is_zero ? 25'd0 : {c_val[31], (cls_c.is_sub ? 1'b0 : 1'b1), c_val[22:0]});

    wire [7:0] c_exp_raw = (cls_c.is_sub || cls_c.is_zero) ? 8'd1 : c_val[30:23];
    wire [EXP_W-1:0] c_exp_adj = EXP_W'(c_exp_raw) - EXP_W'(W-1) + EXP_W'(WA-1) + 128;
    wire [EXP_W-1:0] c_exp_final = cls_c.is_zero ? '0 : c_exp_adj;

    // Output aggregation
    assign sig_out = {c_sig_final, sig_sel};
    assign exp_out = {c_exp_final, exp_sel};

    for (genvar i = 0; i < TCK; ++i) begin : g_exc
        assign exc_out[i] = exc_sel[i];
    end

    assign exc_out[TCK].is_nan = cls_c.is_nan;
    assign exc_out[TCK].is_inf = cls_c.is_inf;
    assign exc_out[TCK].sign   = c_sign;

endmodule

`ifdef TFR_JOIN_F16_ENABLE
`undef TFR_JOIN_F16_ENABLE
`endif
