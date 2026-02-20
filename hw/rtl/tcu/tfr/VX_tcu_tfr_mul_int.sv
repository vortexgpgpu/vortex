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

module VX_tcu_tfr_mul_int import VX_tcu_pkg::*; #(
    parameter `STRING INSTANCE_ID = "",
    parameter N   = 2,
    parameter TCK = 2 * N
) (
    input wire                      clk,
    input wire                      valid_in,
    input wire [31:0]               req_id,

    input wire [TCU_MAX_INPUTS-1:0] vld_mask,
    input wire [2:0]                fmt_i,

    input wire [N-1:0][31:0]        a_row,
    input wire [N-1:0][31:0]        b_col,
    input wire [7:0]                sf_a,
    input wire [7:0]                sf_b,

    output logic [TCK-1:0][24:0]    result
);
    `UNUSED_SPARAM (INSTANCE_ID)
    `UNUSED_VAR ({clk, req_id, valid_in})

    wire is_signed_int = tcu_fmt_is_signed_int(fmt_i);

    // --- MXINT8 Scale Factor (shared across all lanes) --------------------
    wire signed [7:0] sf_wo_bias_a = sf_a - 8'd133;
    wire signed [7:0] sf_wo_bias_b = sf_b - 8'd133;
    wire signed [7:0] sf_wo_bias   = sf_wo_bias_a + sf_wo_bias_b;
    wire        [7:0] abs_sf       = -sf_wo_bias;

    // ----------------------------------------------------------------------
    // 2. Multiplication & Accumulation
    // ----------------------------------------------------------------------
    for (genvar i = 0; i < TCK; ++i) begin : g_lane

        // --- I8/U8 Processing ---------------------------------------------
        wire signed [16:0] y_prod_i8 [2];
        for (genvar j = 0; j < 2; ++j) begin : g_i8
            wire lane_valid = vld_mask[i * 4 + j * 2];
            wire [7:0] raw_a = a_row[i/2][(i%2)*16 + j*8 +: 8];
            wire [7:0] raw_b = b_col[i/2][(i%2)*16 + j*8 +: 8];
            wire signed [8:0] s_a = is_signed_int ? $signed({raw_a[7], raw_a}) : $signed({1'b0, raw_a});
            wire signed [8:0] s_b = is_signed_int ? $signed({raw_b[7], raw_b}) : $signed({1'b0, raw_b});
            wire signed [16:0] prod_full = s_a * s_b;
            assign y_prod_i8[j] = prod_full & {17{lane_valid}};
        end

        wire [16:0] y_i8_add_res;
        VX_ks_adder #(
            .N(17),
            .BYPASS (`FORCE_BUILTIN_ADDER(17))
        ) i8_ksa (
            .cin   (1'b0),
            .dataa (y_prod_i8[0]),
            .datab (y_prod_i8[1]),
            .sum   (y_i8_add_res),
            `UNUSED_PIN(cout)
        );

        // --- MXINT8 Per-Product Scaling -----------------------------------
        // Apply scale shift per-product with truncation toward zero (matching C++ (int32_t)(float) cast)
        wire signed [24:0] y_mxi8_scaled [2];
        for (genvar j = 0; j < 2; ++j) begin : g_mxi8
            wire signed [24:0] prod_ext    = 25'($signed(y_prod_i8[j]));
            wire        [24:0] trunc_bias  = prod_ext[24] ? ((25'd1 << abs_sf) - 25'd1) : 25'd0;
            wire signed [24:0] prod_biased = prod_ext + $signed(trunc_bias);
            assign y_mxi8_scaled[j] = sf_wo_bias[7] ? (prod_biased >>> abs_sf) : (prod_ext <<< sf_wo_bias);
        end
        
        wire [24:0] y_mxi8_add_res;
        VX_ks_adder #(
            .N(25),
            .BYPASS (`FORCE_BUILTIN_ADDER(25))
        ) mxi8_ksa (
            .cin   (1'b0),
            .dataa (y_mxi8_scaled[0]),
            .datab (y_mxi8_scaled[1]),
            .sum   (y_mxi8_add_res),
            `UNUSED_PIN(cout)
        );

        // --- I4/U4 Processing ---------------------------------------------
        wire signed [3:0][9:0] y_prod_i4;
        for (genvar j = 0; j < 4; ++j) begin : g_i4
            wire lane_valid = vld_mask[i * 4 + j];
            wire [3:0] raw_a = a_row[i/2][(i%2)*16 + j*4 +: 4];
            wire [3:0] raw_b = b_col[i/2][(i%2)*16 + j*4 +: 4];
            wire signed [4:0] s_a = is_signed_int ? $signed({raw_a[3], raw_a}) : $signed({1'b0, raw_a});
            wire signed [4:0] s_b = is_signed_int ? $signed({raw_b[3], raw_b}) : $signed({1'b0, raw_b});
            wire signed [9:0] prod_full = s_a * s_b;
            assign y_prod_i4[j] = prod_full & {10{lane_valid}};
        end

        wire [9:0] y_i4_sum, y_i4_carry;
        VX_csa_tree #(
            .N (4),
            .W (10),
            .S (10)
        ) i4_csa (
            .operands (y_prod_i4),
            .sum      (y_i4_sum),
            .carry    (y_i4_carry)
        );

        wire [9:0] y_i4_add_res;
        VX_ks_adder #(
            .N (10),
            .BYPASS (`FORCE_BUILTIN_ADDER(10))
        ) i4_ksa (
            .cin   (1'b0),
            .dataa (y_i4_sum),
            .datab (y_i4_carry),
            .sum   (y_i4_add_res),
            `UNUSED_PIN(cout)
        );

        // --- Output Muxing ------------------------------------------------
        always_comb begin
            case ({1'b1, fmt_i})
                TCU_I8_ID: result[i] = 25'($signed(y_i8_add_res));
                TCU_U8_ID: result[i] = {8'b0, y_i8_add_res};
                TCU_I4_ID: result[i] = 25'($signed(y_i4_add_res));
                TCU_U4_ID: result[i] = {15'b0, y_i4_add_res};
                TCU_MXI8_ID: result[i] = y_mxi8_add_res;
                default:   result[i] = 'x;
            endcase
        end
    end

endmodule
