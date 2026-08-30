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

// Format-shared multiply front end, split across the multiply-stage seam:
// classify/extract, exponent join and the raw multiplies run pre-seam; the
// per-format significand reduction, format join and max-exponent select run
// post-seam on registered products (PROD_REG stages, DSP48 PREG on FPGA).
// All outputs are post-seam aligned.
module VX_tcu_tfr_shared_mul import VX_tcu_pkg::*;  #(
    parameter `STRING INSTANCE_ID = "",
    parameter N = 2,            // Number of 32-bit input registers
    parameter W = 25,           // Product width
    parameter WA = 28,          // Accumulator width
    parameter EXP_W = 10,       // Max exponent width
    parameter TCK = 2 * N,      // Max physical lanes
    parameter SF = 1,           // Scale factor slots
    parameter USE_DSP = 0,      // map mantissa multipliers onto DSP48 slices
    parameter PROD_REG = 0      // product/flag register stages (multiply-stage seam)
) (
    input wire              clk,
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

    output wire [TCK:0][24:0] raw_sigs,
    output fedp_excep_t       exceptions,
    output wire [TCK-1:0]     lane_mask
);
    `UNUSED_SPARAM (INSTANCE_ID)
`ifndef VX_CFG_TCU_MX_ENABLE
    `UNUSED_PARAM (SF)
`endif
    `UNUSED_VAR ({clk, req_id, valid_in})

`ifdef VX_CFG_TCU_FP16_ENABLE
`define TFR_MUL_F16_ENABLE
`elsif VX_CFG_TCU_TF32_ENABLE
`define TFR_MUL_F16_ENABLE
`endif


`ifdef TFR_MUL_F16_ENABLE
    // F16 / BF16 / TF32
    wire [TCK-1:0][24:0]      mul_f16_sig;
    wire [TCK-1:0][EXP_W-1:0] mul_f16_exp;
    fedp_excep_t [TCK-1:0]    mul_f16_exc;

    VX_tcu_tfr_mul_f16 #(
        .N(N),
        .TCK(TCK),
        .W(W),
        .WA(WA),
        .EXP_W(EXP_W),
        .USE_DSP(USE_DSP),
        .PROD_REG(PROD_REG)
    ) mul_f16 (
        .clk        (clk),
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
`endif

`ifdef VX_CFG_TCU_FP8_ENABLE
    // FP8 / BF8 / MXFP8 / MXBF8
    wire [TCK-1:0][24:0]      mul_f8_sig;
    wire [TCK-1:0][EXP_W-1:0] mul_f8_exp;
    fedp_excep_t [TCK-1:0]    mul_f8_exc;
    wire [TCK-1:0]            mul_f8_zero;

    wire [SF-1:0][TCK-1:0][24:0]      mul_f8_sig_s;
    wire [SF-1:0][TCK-1:0][EXP_W-1:0] mul_f8_exp_s;
    fedp_excep_t [SF-1:0][TCK-1:0]    mul_f8_exc_s;
    wire [SF-1:0][TCK-1:0]            mul_f8_zero_s;

    for (genvar s = 0; s < SF; ++s) begin : g_mul_f8_sf
        VX_tcu_tfr_mul_f8 #(
            .N(N),
            .TCK(TCK),
            .W(W),
            .WA(WA),
            .EXP_W(EXP_W),
            .USE_DSP(USE_DSP),
            .PROD_REG(PROD_REG)
        ) mul_f8 (
            .clk        (clk),
            .enable     (enable),
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
            .exceptions (mul_f8_exc_s[s]),
            .sig_zero   (mul_f8_zero_s[s])
        );
    end

    for (genvar i = 0; i < TCK; ++i) begin : g_mul_f8_lane
        localparam SF_SLOT = (i * SF) / TCK;
        assign mul_f8_sig[i]  = mul_f8_sig_s[SF_SLOT][i];
        assign mul_f8_exp[i]  = mul_f8_exp_s[SF_SLOT][i];
        assign mul_f8_exc[i]  = mul_f8_exc_s[SF_SLOT][i];
        assign mul_f8_zero[i] = mul_f8_zero_s[SF_SLOT][i];
    end
`endif

`ifdef VX_CFG_TCU_MX_ENABLE
`ifdef VX_CFG_TCU_FP4_ENABLE
    // MXFP4 / NVFP4
    wire [TCK-1:0][24:0]      mul_f4_sig;
    wire [TCK-1:0][EXP_W-1:0] mul_f4_exp;
    fedp_excep_t [TCK-1:0]    mul_f4_exc;
    wire [TCK-1:0]            mul_f4_zero;

    wire [SF-1:0][TCK-1:0][24:0]      mul_f4_sig_s;
    wire [SF-1:0][TCK-1:0][EXP_W-1:0] mul_f4_exp_s;
    fedp_excep_t [SF-1:0][TCK-1:0]    mul_f4_exc_s;
    wire [SF-1:0][TCK-1:0]            mul_f4_zero_s;

    for (genvar s = 0; s < SF; ++s) begin : g_mul_f4_sf
        VX_tcu_tfr_mul_f4 #(
            .N(N),
            .TCK(TCK),
            .W(W),
            .WA(WA),
            .EXP_W(EXP_W),
            .USE_DSP(USE_DSP),
            .PROD_REG(PROD_REG)
        ) mul_f4 (
            .clk        (clk),
            .enable     (enable),
            .valid_in   (valid_in),
            .req_id     (req_id),
            .vld_mask   (vld_mask),
            .fmt_f      (fmt_s[3:0]),
            .a_row      (a_row),
            .b_col      (b_col),
            .sf_a       (sf_a[s]),
            .sf_b       (sf_b[s]),
            .result_sig (mul_f4_sig_s[s]),
            .result_exp (mul_f4_exp_s[s]),
            .exceptions (mul_f4_exc_s[s]),
            .sig_zero   (mul_f4_zero_s[s])
        );
    end

    for (genvar i = 0; i < TCK; ++i) begin : g_mul_f4_lane
        localparam SF_SLOT = (i * SF) / TCK;
        assign mul_f4_sig[i]  = mul_f4_sig_s[SF_SLOT][i];
        assign mul_f4_exp[i]  = mul_f4_exp_s[SF_SLOT][i];
        assign mul_f4_exc[i]  = mul_f4_exc_s[SF_SLOT][i];
        assign mul_f4_zero[i] = mul_f4_zero_s[SF_SLOT][i];
    end
`endif
`endif

`ifdef VX_CFG_TCU_INT8_ENABLE
    // I8 / U8
    wire [TCK-1:0][24:0] mul_int8_sig;
    VX_tcu_tfr_mul_i8 #(
        .N(N),
        .TCK(TCK),
        .USE_DSP(USE_DSP),
        .PROD_REG(PROD_REG)
    ) mul_int8 (
        .clk        (clk),
        .enable     (enable),
        .valid_in   (valid_in),
        .req_id     (req_id),
        .vld_mask   (vld_mask),
        .fmt_i      (fmt_s[3:0]),
        .a_row      (a_row),
        .b_col      (b_col),
        .result     (mul_int8_sig)
    );
`endif

`ifdef VX_CFG_TCU_INT4_ENABLE
    // I4 / U4
    wire [TCK-1:0][24:0] mul_int4_sig;
    VX_tcu_tfr_mul_i4 #(
        .N(N),
        .TCK(TCK),
        .USE_DSP(USE_DSP),
        .PROD_REG(PROD_REG)
    ) mul_int4 (
        .clk        (clk),
        .enable     (enable),
        .valid_in   (valid_in),
        .req_id     (req_id),
        .vld_mask   (vld_mask),
        .fmt_i      (fmt_s[3:0]),
        .a_row      (a_row),
        .b_col      (b_col),
        .result     (mul_int4_sig)
    );
`endif

    // Pre-seam: exponent/exception join and C-term decompose.
    wire [TCK:0][EXP_W-1:0] join_exp;
    fedp_excep_t [TCK:0]    join_exc;
    wire [24:0]             c_sig;

    VX_tcu_tfr_mul_join #(
        .N(N),
        .TCK(TCK),
        .W(W),
        .WA(WA),
        .EXP_W(EXP_W)
    ) join_stage (
        .clk        (clk),
        .valid_in   (valid_in),
        .req_id     (req_id),

        .fmt_s      (fmt_s),

        .c_val      (c_val),

    `ifdef TFR_MUL_F16_ENABLE
        .exp_f16    (mul_f16_exp),
        .exc_f16    (mul_f16_exc),
    `endif

    `ifdef VX_CFG_TCU_FP8_ENABLE
        .exp_f8     (mul_f8_exp),
        .exc_f8     (mul_f8_exc),
    `endif

    `ifdef VX_CFG_TCU_MX_ENABLE
    `ifdef VX_CFG_TCU_FP4_ENABLE
        .exp_f4     (mul_f4_exp),
        .exc_f4     (mul_f4_exc),
    `endif
    `endif

        .exp_out    (join_exp),
        .exc_out    (join_exc),
        .c_sig      (c_sig)
    );

    fedp_excep_t exceptions_w;
    VX_tcu_tfr_exc_reduce #(
        .TCK (TCK)
    ) exc_reduce (
        .exc_in  (join_exc),
        .exc_out (exceptions_w)
    );

    // Lane mask
    wire [TCK-1:0] lane_mask_w;
    VX_tcu_tfr_lane_mask #(
        .N   (N),
        .TCK (TCK)
    ) lane_mask_inst (
        .vld_mask (vld_mask),
        .fmt_s    (fmt_s),
        .lane_mask(lane_mask_w)
    );

    // Multiply-stage seam: carry the joined exponents and scalar controls to
    // post-seam timing alongside the registered products.
    localparam EXC_W = $bits(fedp_excep_t);

    wire [TCK:0][EXP_W-1:0] join_exp_r;
    wire [24:0]             c_sig_r;
    fedp_excep_t            exceptions_r;
    wire [TCK-1:0]          lane_mask_r;
    wire [4:0]              fmt_s_r;

    VX_pipe_register #(
        .DATAW ((TCK+1)*EXP_W + 25 + EXC_W + TCK + 5),
        .DEPTH (PROD_REG)
    ) pipe_seam (
        .clk      (clk),
        .reset    (1'b0),
        .enable   (enable),
        .data_in  ({join_exp,   c_sig,   exceptions_w, lane_mask_w, fmt_s}),
        .data_out ({join_exp_r, c_sig_r, exceptions_r, lane_mask_r, fmt_s_r})
    );

    // Post-seam: join the per-format significands and cancellation flags.
    logic [TCK-1:0][24:0] sig_sel;
    logic [TCK-1:0]       sig_zero;

    always_comb begin
        case (fmt_s_r)
        `ifdef TFR_MUL_F16_ENABLE
        `ifdef VX_CFG_TCU_FP16_ENABLE
            TCU_FP16_ID,
            TCU_BF16_ID: begin
                sig_sel  = mul_f16_sig;
                sig_zero = '0;
            end
        `endif
        `ifdef VX_CFG_TCU_TF32_ENABLE
            TCU_TF32_ID: begin
                sig_sel  = mul_f16_sig;
                sig_zero = '0;
            end
        `endif
        `endif

        `ifdef VX_CFG_TCU_FP8_ENABLE
            TCU_FP8_ID, TCU_BF8_ID
        `ifdef VX_CFG_TCU_MX_ENABLE
            , TCU_MXFP8_ID, TCU_MXBF8_ID
        `endif
            : begin
                sig_sel  = mul_f8_sig;
                sig_zero = mul_f8_zero;
            end
        `endif

        `ifdef VX_CFG_TCU_MX_ENABLE
        `ifdef VX_CFG_TCU_FP4_ENABLE
        `ifdef VX_CFG_TCU_MXFP4_ENABLE
            TCU_MXFP4_ID: begin
                sig_sel  = mul_f4_sig;
                sig_zero = mul_f4_zero;
            end
        `endif
        `ifdef VX_CFG_TCU_NVFP4_ENABLE
            TCU_NVFP4_ID: begin
                sig_sel  = mul_f4_sig;
                sig_zero = mul_f4_zero;
            end
        `endif
        `endif
        `endif

        `ifdef VX_CFG_TCU_INT8_ENABLE
            TCU_I8_ID, TCU_U8_ID: begin
                sig_sel  = mul_int8_sig;
                sig_zero = '0;
            end
        `endif
        `ifdef VX_CFG_TCU_INT4_ENABLE
            TCU_I4_ID, TCU_U4_ID: begin
                sig_sel  = mul_int4_sig;
                sig_zero = '0;
            end
        `endif
            default: begin
                sig_sel  = '0;
                sig_zero = '0;
            end
        endcase
    end

    assign raw_sigs = {c_sig_r, sig_sel};

    // Exact-cancellation lanes drop out of the max-exponent search.
    for (genvar i = 0; i < TCK; ++i) begin : g_exp_kill
        assign exponents[i] = sig_zero[i] ? '0 : join_exp_r[i];
    end
    assign exponents[TCK] = join_exp_r[TCK];

    // Maximum exponent select
    VX_tcu_tfr_max_exp #(
        .N     (TCK+1),
        .WIDTH (EXP_W)
    ) find_max_exp (
        .exponents (exponents),
        .sel_exp   (exp_sel)
    );

    assign exceptions = exceptions_r;
    assign lane_mask  = lane_mask_r;

endmodule

`ifdef TFR_MUL_F16_ENABLE
`undef TFR_MUL_F16_ENABLE
`endif
