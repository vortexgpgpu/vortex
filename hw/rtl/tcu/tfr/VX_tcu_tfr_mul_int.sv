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

// Integer dot-product lanes: I8/U8 (two products per lane) and I4/U4 (four).
// With USE_DSP, each byte of a lane owns one DSP48 that serves whichever
// integer format is active: one signed 9x9 int8 product, or the two int4
// products of the byte's nibbles packed as unsigned magnitudes, so int8 and
// int4 together take two DSPs per lane.
module VX_tcu_tfr_mul_int import VX_tcu_pkg::*; #(
    parameter `STRING INSTANCE_ID = "",
    parameter N   = 2,
    parameter TCK = 2 * N,
    parameter USE_DSP = 0,  // map the integer multiplies onto DSP48 slices
    parameter PROD_REG = 0  // product/flag register stages (multiply-stage seam)
) (
    input wire                      clk,
    input wire                      enable,
    input wire                      valid_in,
    input wire [31:0]               req_id,

    input wire [TCU_MAX_INPUTS-1:0] vld_mask,
    input wire [3:0]                fmt_i,

    input wire [N-1:0][31:0]        a_row,
    input wire [N-1:0][31:0]        b_col,

    output logic [TCK-1:0][24:0]    result
);
    `UNUSED_SPARAM (INSTANCE_ID)
    `UNUSED_VAR ({clk, req_id, valid_in})

`ifdef VX_CFG_TCU_INT8_ENABLE
    localparam I8 = 1;
`else
    localparam I8 = 0;
`endif
`ifdef VX_CFG_TCU_INT4_ENABLE
    localparam I4 = 1;
`else
    localparam I4 = 0;
`endif

    // Nibble pair packing stride: each 4x4 magnitude product (<= 8 bits) and
    // the discarded cross-term field (< 2^9) stay clear of each other.
    localparam K = 9;
    // Signed 9x9 int8 product width; also the DSP operand width.
    localparam OPW = 17;

    wire is_signed_int = fmt_i[3] || tcu_fmt_is_signed_int(fmt_i);
    // I4/U4 among the integer formats; any other format's result is dropped.
    wire is_int4 = (I4 != 0) && ((I8 == 0) || fmt_i[2] || (fmt_i[1] && fmt_i[0]));

    // Result-select format, delayed to post-seam timing.
    wire [3:0] fmt_i_r;
    VX_pipe_register #(
        .DATAW (4),
        .DEPTH (PROD_REG)
    ) pipe_fmt (
        .clk      (clk),
        .reset    (1'b0),
        .enable   (enable),
        .data_in  (fmt_i),
        .data_out (fmt_i_r)
    );

    for (genvar i = 0; i < TCK; ++i) begin : g_lane

        wire [3:0] pvalid;
        for (genvar j = 0; j < 4; ++j) begin : g_vld
            assign pvalid[j] = vld_mask[i * 4 + j];
        end

        // Invalid lanes are masked post-seam so the DSP input cone stays flat.
        wire [3:0] pvalid_r;
        VX_pipe_register #(
            .DATAW (4),
            .DEPTH (PROD_REG)
        ) pipe_valid (
            .clk      (clk),
            .reset    (1'b0),
            .enable   (enable),
            .data_in  (pvalid),
            .data_out (pvalid_r)
        );

        // Seam-registered products: int8 per byte; int4 per nibble as masked
        // tree operands plus their +1 completion count.
        wire [1:0][16:0] prod_i8;
        wire [3:0][9:0]  y_prod_i4;
        wire [9:0]       y_corr_i4;

        if (USE_DSP != 0) begin : g_dsp
            // int4 a*b == sign(a^b) * (|a|*|b|): the DSP multiplies magnitudes
            // and the sign is re-applied in fabric.
            wire [3:0]      psign;
            wire [3:0][7:0] magp;
            for (genvar p = 0; p < 2; ++p) begin : g_byte
                localparam OFF = (i % 2) * 16 + p * 8;
                wire [7:0] raw_a = a_row[i/2][OFF +: 8];
                wire [7:0] raw_b = b_col[i/2][OFF +: 8];

                wire [1:0][3:0] mag_a, mag_b;
                for (genvar h = 0; h < 2; ++h) begin : g_mag
                    wire sgn_a = is_signed_int & raw_a[h*4+3];
                    wire sgn_b = is_signed_int & raw_b[h*4+3];
                    assign mag_a[h] = sgn_a ? (4'(~raw_a[h*4 +: 4]) + 4'b1) : raw_a[h*4 +: 4];
                    assign mag_b[h] = sgn_b ? (4'(~raw_b[h*4 +: 4]) + 4'b1) : raw_b[h*4 +: 4];
                    assign psign[p*2+h] = sgn_a ^ sgn_b;
                end

                // int8: the byte sign-extended to the product width, so the low
                // bits of the unsigned product are the signed product; int4:
                // {|hi|, 0, |lo|} at stride K.
                wire sext_a = is_signed_int & raw_a[7];
                wire sext_b = is_signed_int & raw_b[7];
                wire [OPW-1:0] op_a = is_int4 ? OPW'({mag_a[1], {(K-4){1'b0}}, mag_a[0]})
                                              : {{(OPW-8){sext_a}}, raw_a};
                wire [OPW-1:0] op_b = is_int4 ? OPW'({mag_b[1], {(K-4){1'b0}}, mag_b[0]})
                                              : {{(OPW-8){sext_b}}, raw_b};

                // The full product is registered so it lands in the DSP48 PREG.
                (* use_dsp = "yes" *) wire [2*OPW-1:0] prod = op_a * op_b;
                wire [2*OPW-1:0] prod_r;
                VX_pipe_register #(
                    .DATAW (2*OPW),
                    .DEPTH (PROD_REG)
                ) prod_reg (
                    .clk      (clk),
                    .reset    (1'b0),
                    .enable   (enable),
                    .data_in  (prod),
                    .data_out (prod_r)
                );
                assign prod_i8[p]     = prod_r[16:0];
                assign magp[p*2+0]    = prod_r[7:0];
                assign magp[p*2+1]    = prod_r[2*K +: 8];
                `UNUSED_VAR (prod_r)
            end

            wire [3:0] psign_r;
            VX_pipe_register #(
                .DATAW (4),
                .DEPTH (PROD_REG)
            ) pipe_sign (
                .clk      (clk),
                .reset    (1'b0),
                .enable   (enable),
                .data_in  (psign),
                .data_out (psign_r)
            );

            // -m = ~m + 1: the negated magnitudes enter the tree as ones'
            // complements and their +1s as one count operand, so no negate
            // sits between the DSP and the tree.
            wire [3:0] pneg = psign_r & pvalid_r;
            for (genvar j = 0; j < 4; ++j) begin : g_sign
                wire [9:0] pm = {2'b0, magp[j]} & {10{pvalid_r[j]}};
                assign y_prod_i4[j] = pneg[j] ? ~pm : pm;
            end
            wire [2:0] pneg_cnt = 3'(pneg[0]) + 3'(pneg[1]) + 3'(pneg[2]) + 3'(pneg[3]);
            assign y_corr_i4 = 10'(pneg_cnt);
        end else begin : g_lut
            `UNUSED_VAR (is_int4)
            for (genvar p = 0; p < 2; ++p) begin : g_i8
                wire [7:0] raw_a = a_row[i/2][(i%2)*16 + p*8 +: 8];
                wire [7:0] raw_b = b_col[i/2][(i%2)*16 + p*8 +: 8];
                wire signed [8:0] s_a = is_signed_int ? $signed({raw_a[7], raw_a}) : $signed({1'b0, raw_a});
                wire signed [8:0] s_b = is_signed_int ? $signed({raw_b[7], raw_b}) : $signed({1'b0, raw_b});
                wire signed [16:0] prod_full = s_a * s_b;
                VX_pipe_register #(
                    .DATAW (17),
                    .DEPTH (PROD_REG)
                ) pipe_prod (
                    .clk      (clk),
                    .reset    (1'b0),
                    .enable   (enable),
                    .data_in  (prod_full),
                    .data_out (prod_i8[p])
                );
            end
            for (genvar j = 0; j < 4; ++j) begin : g_i4
                wire [3:0] raw_a = a_row[i/2][(i%2)*16 + j*4 +: 4];
                wire [3:0] raw_b = b_col[i/2][(i%2)*16 + j*4 +: 4];
                wire signed [4:0] s_a = is_signed_int ? $signed({raw_a[3], raw_a}) : $signed({1'b0, raw_a});
                wire signed [4:0] s_b = is_signed_int ? $signed({raw_b[3], raw_b}) : $signed({1'b0, raw_b});
                wire signed [9:0] prod_full = s_a * s_b;
                wire [9:0] prod_r;
                VX_pipe_register #(
                    .DATAW (10),
                    .DEPTH (PROD_REG)
                ) pipe_prod (
                    .clk      (clk),
                    .reset    (1'b0),
                    .enable   (enable),
                    .data_in  (prod_full),
                    .data_out (prod_r)
                );
                assign y_prod_i4[j] = prod_r & {10{pvalid_r[j]}};
            end
            assign y_corr_i4 = '0;
        end

        wire [9:0] i4_sum, i4_carry;
        VX_csa_tree #(
            .N (5),
            .W (10),
            .S (10)
        ) i4_csa (
            .operands ({y_corr_i4, y_prod_i4}),
            .sum      (i4_sum),
            .carry    (i4_carry)
        );

        wire [9:0] i4_res;
        VX_ks_adder #(
            .N (10),
            .BYPASS (`FORCE_BUILTIN_ADDER(10))
        ) i4_ksa (
            .cin   (1'b0),
            .dataa (i4_sum),
            .datab (i4_carry),
            .sum   (i4_res),
            `UNUSED_PIN(cout)
        );

        wire [16:0] i8_res;
        VX_ks_adder #(
            .N (17),
            .BYPASS (`FORCE_BUILTIN_ADDER(17))
        ) i8_ksa (
            .cin   (1'b0),
            .dataa (prod_i8[0] & {17{pvalid_r[0]}}),
            .datab (prod_i8[1] & {17{pvalid_r[2]}}),
            .sum   (i8_res),
            `UNUSED_PIN(cout)
        );

        always_comb begin
            case ({1'b1, fmt_i_r})
            `ifdef VX_CFG_TCU_INT8_ENABLE
                TCU_I8_ID: result[i] = 25'($signed(i8_res));
                TCU_U8_ID: result[i] = {8'b0, i8_res};
            `endif
            `ifdef VX_CFG_TCU_INT4_ENABLE
                TCU_I4_ID: result[i] = 25'($signed(i4_res));
                TCU_U4_ID: result[i] = {15'b0, i4_res};
            `endif
                default:   result[i] = '0;
            endcase
        end
    `ifndef VX_CFG_TCU_INT8_ENABLE
        `UNUSED_VAR (i8_res)
    `endif
    `ifndef VX_CFG_TCU_INT4_ENABLE
        `UNUSED_VAR (i4_res)
    `endif
    end

endmodule
