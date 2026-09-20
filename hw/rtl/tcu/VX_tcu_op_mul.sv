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

`ifdef TCU_OP

// TCU_OP multiply bank: one A element times N B elements.
//
// Each product lands in a different accumulator element, so this is an outer
// product and the TFR FEDP multiply banks cannot be instantiated directly --
// their lanes pair a_row[i] with b_col[i] for a dot product, and the fp8 bank
// even folds two sub-products into one lane. What is reused is everything that
// carries the arithmetic: VX_tcu_tfr_wmul for the significand multiply (with
// its DSP48 mapping), and TFR's product encoding, so VX_tcu_tfr_norm_round
// works downstream unchanged.
//
// Products leave here EXACT and unrounded, as {sign, exponent, magnitude} in
// the domain VX_tcu_pkg documents (see TCU_OP_EXP_BIAS): significands are
// padded to TCU_OP_SIG_W so one 11x11 multiplier and one exponent formula
// serve every supported format. Rounding happens once per output element, at
// flush, in VX_tcu_op_accu.
module VX_tcu_op_mul import VX_tcu_pkg::*; #(
    parameter N        = 16,  // products (B elements) per A element
    parameter USE_DSP  = 0,   // map the 11x11 significand multiply onto DSP48
    parameter PROD_REG = 1    // product register stages (DSP48 PREG on FPGA)
) (
    input  wire clk,
    input  wire enable,

    input  wire [TCU_FMT_WIDTH-1:0]       fmt_s,
    input  wire [1:0]                     lg_i_ratio, // elements per word, log2
    input  wire [`VX_CFG_XLEN-1:0]        a_elem,     // A element, right-aligned
    input  wire [N-1:0][`VX_CFG_XLEN-1:0] b_row,      // B words, packed elements
    input  wire [N-1:0]                   valid_in_bitmap,

    output wire [N-1:0][TCU_OP_EXP_W-1:0] prod_exp,
    output wire [N-1:0][TCU_OP_MAG_W-1:0] prod_mag,
    output wire [N-1:0]                   prod_sign,
    output fedp_excep_t [N-1:0]           prod_exc
);
    localparam SIG_W  = TCU_OP_SIG_W;       // 11
    localparam PROD_W = 2 * SIG_W;          // 22
    localparam MAN_W  = SIG_W - 1;          // 10, padded mantissa field

    `STATIC_ASSERT (`VX_CFG_XLEN == 32, ("tcu_op_mul: 32-bit operand words assumed"))
    `STATIC_ASSERT (TCU_OP_MAG_W == PROD_W + 2, ("tcu_op_mul: magnitude must be the product left-aligned by 2"))

    // Field extraction. The exponent's zero/all-ones tests run on the format's
    // OWN width before zero-extension, so they stay correct for every format
    // (VX_tcu_tfr_classifier assumes one width per instance, which is why the
    // TFR banks instantiate several; here the format case already separates
    // them). Subnormals are exact: exponent 0 becomes 1 with no hidden bit.
    typedef struct packed {
        logic             sign;
        logic [7:0]       exp;     // zero-extended raw field
        logic [MAN_W-1:0] man;     // left-aligned to MAN_W bits
        logic             e_zero;
        logic             e_ones;
    } unpack_t;

    function automatic unpack_t tcu_op_unpack (
        input logic [TCU_FMT_WIDTH-1:0] fmt,
        input logic [31:0]              w
    );
        unpack_t u;
        // Narrow formats read only the low field of the word.
        `UNUSED_VAR (w)
        u = '0;
        case (fmt)
            TCU_FP16_ID: begin
                u.sign   = w[15];
                u.exp    = 8'(w[14:10]);
                u.man    = w[9:0];
                u.e_zero = ~|w[14:10];
                u.e_ones =  &w[14:10];
            end
            TCU_BF16_ID: begin
                u.sign   = w[15];
                u.exp    = w[14:7];
                u.man    = {w[6:0], 3'b0};
                u.e_zero = ~|w[14:7];
                u.e_ones =  &w[14:7];
            end
            TCU_FP8_ID: begin
                u.sign   = w[7];
                u.exp    = 8'(w[6:3]);
                u.man    = {w[2:0], 7'b0};
                u.e_zero = ~|w[6:3];
                u.e_ones =  &w[6:3];
            end
            TCU_BF8_ID: begin
                u.sign   = w[7];
                u.exp    = 8'(w[6:2]);
                u.man    = {w[1:0], 8'b0};
                u.e_zero = ~|w[6:2];
                u.e_ones =  &w[6:2];
            end
            default: begin
                // Unsupported format: refused by the op core's assertion.
            end
        endcase
        return u;
    endfunction

    // Per-format exponent constant, resolved once for the whole bank.
    wire [TCU_OP_EXP_W-1:0] exp_k = TCU_OP_EXP_W'(tcu_op_exp_k(fmt_s));

    // ---- A side: unpacked once and shared by every lane ------------------
    unpack_t a_u;
    assign a_u = tcu_op_unpack(fmt_s, a_elem);

    wire             a_is_zero = a_u.e_zero & ~|a_u.man;
    wire             a_is_nan  = a_u.e_ones &  |a_u.man;
    wire             a_is_inf  = a_u.e_ones & ~|a_u.man;
    wire [7:0]       a_exp     = a_u.e_zero ? 8'd1 : a_u.exp;
    wire [SIG_W-1:0] a_sig     = {~a_u.e_zero, a_u.man};

    for (genvar j = 0; j < N; ++j) begin : g_lane
        // The B element's position follows the format's element width; the
        // words are already in place, so this is a word select plus a
        // constant-offset field select -- no bit-granular shifting.
        wire [31:0] b_word = (lg_i_ratio == 2'd0) ? b_row[j] :
                             (lg_i_ratio == 2'd1) ? {16'b0, b_row[j >> 1][((j & 1) << 4) +: 16]} :
                                                    {24'b0, b_row[j >> 2][((j & 3) << 3) +: 8]};

        unpack_t b_u;
        assign b_u = tcu_op_unpack(fmt_s, b_word);

        wire             b_is_zero = b_u.e_zero & ~|b_u.man;
        wire             b_is_nan  = b_u.e_ones &  |b_u.man;
        wire             b_is_inf  = b_u.e_ones & ~|b_u.man;
        wire [7:0]       b_exp     = b_u.e_zero ? 8'd1 : b_u.exp;
        wire [SIG_W-1:0] b_sig     = {~b_u.e_zero, b_u.man};

        wire lane_en = valid_in_bitmap[j];

        wire inf_times_zero = (a_is_inf & b_is_zero) | (a_is_zero & b_is_inf);
        wire any_nan        = a_is_nan | b_is_nan;
        wire is_nan_w  = lane_en & (any_nan | inf_times_zero);
        wire is_inf_w  = lane_en & (a_is_inf | b_is_inf) & ~inf_times_zero & ~any_nan;
        wire is_zero_w = ~lane_en | a_is_zero | b_is_zero | any_nan | a_is_inf | b_is_inf;
        wire sign_w    = a_u.sign ^ b_u.sign;

        // A zero, masked or special term carries exponent 0 and magnitude 0, so
        // it adds nothing; inf/nan travel as flags instead of as a value.
        wire [TCU_OP_EXP_W-1:0] exp_w = is_zero_w ? '0 :
                                        (TCU_OP_EXP_W'(a_exp) + TCU_OP_EXP_W'(b_exp) + exp_k);

        // Significand multiply: exact, unrounded, reused from TFR.
        wire [PROD_W-1:0] sig_prod;
        VX_tcu_tfr_wmul #(
            .N       (SIG_W),
            .USE_DSP (USE_DSP),
            .OUT_REG (PROD_REG)
        ) sig_mul (
            .clk    (clk),
            .enable (enable),
            .a      (a_sig),
            .b      (b_sig),
            .p      (sig_prod)
        );

        // Carry the exponent, sign and exception flags across the product
        // register so they stay aligned with the significand.
        wire [TCU_OP_EXP_W-1:0] exp_r;
        wire sign_r, nan_r, inf_r, zero_r;
        VX_pipe_register #(
            .DATAW (TCU_OP_EXP_W + 4),
            .DEPTH (PROD_REG)
        ) pipe_seam (
            .clk      (clk),
            .reset    (1'b0),
            .enable   (enable),
            .data_in  ({exp_w, sign_w, is_nan_w, is_inf_w, is_zero_w}),
            .data_out ({exp_r, sign_r, nan_r,    inf_r,    zero_r})
        );

        assign prod_exp[j]        = exp_r;
        assign prod_mag[j]        = zero_r ? '0 : {sig_prod, 2'b0};
        assign prod_sign[j]       = sign_r;
        assign prod_exc[j].is_nan = nan_r;
        assign prod_exc[j].is_inf = inf_r;
        assign prod_exc[j].sign   = sign_r;
    end

endmodule

`endif // TCU_OP
