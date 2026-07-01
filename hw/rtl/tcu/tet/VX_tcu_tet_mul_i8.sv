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

module VX_tcu_tet_mul_i8 import VX_tcu_pkg::*; #(
    parameter `STRING INSTANCE_ID = "",
    parameter N   = 2,
    parameter TCK = 2 * N
) (
    input wire                      clk,
    input wire                      reset,
    input wire                      enable,
    input wire                      valid_in,
    input wire [31:0]               req_id,

    input wire [TCU_MAX_INPUTS-1:0] vld_mask,
    input wire [3:0]                fmt_i,

    input wire [N-1:0][31:0]        a_row,
    input wire [N-1:0][31:0]        b_col,
`ifdef VX_CFG_TCU_MX_ENABLE
    input wire [7:0]                sf_a,
    input wire [7:0]                sf_b,
`endif

    output logic [TCK-1:0][24:0]    result
);
    `UNUSED_SPARAM (INSTANCE_ID)
    `UNUSED_VAR ({req_id, valid_in})
    for (genvar i = 1; i < TCU_MAX_INPUTS; i += 2) begin : g_unused_vld
        `UNUSED_VAR (vld_mask[i])
    end

    wire is_signed_int = fmt_i[3] || tcu_fmt_is_signed_int(fmt_i);

    // Multiplication and accumulation
    for (genvar i = 0; i < TCK; ++i) begin : g_lane

        wire signed [16:0] y_prod_i8 [2];
        for (genvar j = 0; j < 2; ++j) begin : g_i8
            wire lane_valid = vld_mask[i * 4 + j * 2];
            wire [7:0] raw_a = a_row[i/2][(i%2)*16 + j*8 +: 8];
            wire [7:0] raw_b = b_col[i/2][(i%2)*16 + j*8 +: 8];
            wire signed [8:0] s_a = is_signed_int ? $signed({raw_a[7], raw_a}) : $signed({1'b0, raw_a});
            wire signed [8:0] s_b = is_signed_int ? $signed({raw_b[7], raw_b}) : $signed({1'b0, raw_b});
            (* use_dsp = "yes" *) wire signed [16:0] prod_full = s_a * s_b;
            assign y_prod_i8[j] = prod_full & {17{lane_valid}};
        end

        wire signed [16:0] s1_y_prod_i8 [2];
        wire s1_is_i8;
        wire s1_is_u8;
    `ifdef VX_CFG_TCU_MX_ENABLE
        wire signed [8:0] combined_sf = $signed({1'b0, sf_a}) + $signed({1'b0, sf_b}) - 9'sd266;
        wire s1_is_mxi8;
        wire signed [8:0] s1_combined_sf;
    `endif

        VX_tcu_tet_register #(
        `ifdef VX_CFG_TCU_MX_ENABLE
            .DATAW ((2 * 17) + 9 + 3),
        `else
            .DATAW ((2 * 17) + 2),
        `endif
            .DEPTH (1)
        ) pipe_i8_s0 (
            .clk      (clk),
            .reset    (reset),
            .enable   (enable),
        `ifdef VX_CFG_TCU_MX_ENABLE
            .data_in  ({y_prod_i8[0],    y_prod_i8[1],    combined_sf,    (fmt_i == 4'(TCU_MXI8_ID)), (fmt_i == 4'(TCU_U8_ID)), (fmt_i == 4'(TCU_I8_ID))}),
            .data_out ({s1_y_prod_i8[0], s1_y_prod_i8[1], s1_combined_sf, s1_is_mxi8,                 s1_is_u8,                 s1_is_i8})
        `else
            .data_in  ({y_prod_i8[0],    y_prod_i8[1],    (fmt_i == 4'(TCU_U8_ID)), (fmt_i == 4'(TCU_I8_ID))}),
            .data_out ({s1_y_prod_i8[0], s1_y_prod_i8[1], s1_is_u8,                 s1_is_i8})
        `endif
        );

        wire [16:0] y_i8_add_res;
        VX_ks_adder #(
            .N(17),
            .BYPASS (`FORCE_BUILTIN_ADDER(17))
        ) i8_ksa (
            .cin   (1'b0),
            .dataa (s1_y_prod_i8[0]),
            .datab (s1_y_prod_i8[1]),
            .sum   (y_i8_add_res),
            `UNUSED_PIN(cout)
        );

`ifdef VX_CFG_TCU_MX_ENABLE
        wire is_right_shift = s1_combined_sf[8];
        wire shift_overflow = (s1_combined_sf > 9'sd24) || (s1_combined_sf < -9'sd24);
        wire [4:0] shift_amount = is_right_shift ? (-s1_combined_sf[4:0]) : s1_combined_sf[4:0];

        wire signed [24:0] y_mxi8_scaled [2];
        for (genvar j = 0; j < 2; ++j) begin : g_mxi8
            wire signed [24:0] raw_prod = {{8{s1_y_prod_i8[j][16]}}, s1_y_prod_i8[j]};
            wire [24:0] abs_prod = raw_prod[24] ? -raw_prod : raw_prod;
            wire signed [24:0] right_shifted = raw_prod[24] ? -25'($signed(abs_prod >> shift_amount))
                                                            :  25'($signed(abs_prod >> shift_amount));
            assign y_mxi8_scaled[j] = shift_overflow ? 25'sd0
                                    : is_right_shift  ? right_shifted
                                    :                   (raw_prod <<< shift_amount);
        end

        wire [24:0] y_mxi8_add_res;
        VX_ks_adder #(
            .N      (25),
            .BYPASS (`FORCE_BUILTIN_ADDER(25))
        ) mxi8_ksa (
            .cin   (1'b0),
            .dataa (y_mxi8_scaled[0]),
            .datab (y_mxi8_scaled[1]),
            .sum   (y_mxi8_add_res),
            `UNUSED_PIN(cout)
        );
`endif

        // Output muxing
        always_comb begin
            if (s1_is_i8) begin
                result[i] = 25'($signed(y_i8_add_res));
            end else if (s1_is_u8) begin
                result[i] = {8'b0, y_i8_add_res};
        `ifdef VX_CFG_TCU_MX_ENABLE
            end else if (s1_is_mxi8) begin
                result[i] = y_mxi8_add_res;
        `endif
            end else begin
                result[i] = '0;
            end
        end
    end

endmodule
