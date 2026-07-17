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

module VX_tcu_tfr_mul_i8 import VX_tcu_pkg::*; #(
    parameter `STRING INSTANCE_ID = "",
    parameter N   = 2,
    parameter TCK = 2 * N,
    parameter USE_DSP = 0,  // map the int8 multiplies onto DSP48 slices
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
    for (genvar i = 1; i < TCU_MAX_INPUTS; i += 2) begin : g_unused_vld
        `UNUSED_VAR (vld_mask[i])
    end

    wire is_signed_int = fmt_i[3] || tcu_fmt_is_signed_int(fmt_i);

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

`ifdef VX_CFG_TCU_MX_ENABLE
    // MX shift controls depend only on the scale factors: pre-seam compute,
    // post-seam use.
    wire signed [8:0] combined_sf   = $signed(sf_a + sf_b - 9'd266);
    wire is_right_shift_w = combined_sf[8];
    wire shift_overflow_w = (combined_sf > 9'sd24) || (combined_sf < -9'sd24);
    wire [4:0] shift_amount_w = is_right_shift_w ? (-combined_sf[4:0]) : combined_sf[4:0];

    wire is_right_shift, shift_overflow;
    wire [4:0] shift_amount;
    VX_pipe_register #(
        .DATAW (1 + 1 + 5),
        .DEPTH (PROD_REG)
    ) pipe_sf (
        .clk      (clk),
        .reset    (1'b0),
        .enable   (enable),
        .data_in  ({is_right_shift_w, shift_overflow_w, shift_amount_w}),
        .data_out ({is_right_shift,   shift_overflow,   shift_amount})
    );
`endif

    // Multiplication and accumulation
    for (genvar i = 0; i < TCK; ++i) begin : g_lane

        wire signed [16:0] y_prod_i8 [2];
        for (genvar j = 0; j < 2; ++j) begin : g_i8
            wire lane_valid = vld_mask[i * 4 + j * 2];
            wire [7:0] raw_a = a_row[i/2][(i%2)*16 + j*8 +: 8];
            wire [7:0] raw_b = b_col[i/2][(i%2)*16 + j*8 +: 8];
            wire signed [8:0] s_a = is_signed_int ? $signed({raw_a[7], raw_a}) : $signed({1'b0, raw_a});
            wire signed [8:0] s_b = is_signed_int ? $signed({raw_b[7], raw_b}) : $signed({1'b0, raw_b});
            // 9x9 signed product; USE_DSP maps it to a DSP48 (else LUT fabric).
            // PROD_REG lands the product in the DSP48 PREG; invalid lanes are
            // masked post-seam so the DSP input cone stays flat.
            wire signed [16:0] prod_full;
            if (USE_DSP != 0) begin : g_dsp
                (* use_dsp = "yes" *) wire signed [16:0] dsp_prod = s_a * s_b;
                assign prod_full = dsp_prod;
            end else begin : g_lut
                assign prod_full = s_a * s_b;
            end
            wire [16:0] prod_r;
            VX_pipe_register #(
                .DATAW (17),
                .DEPTH (PROD_REG)
            ) pipe_prod (
                .clk      (clk),
                .reset    (1'b0),
                .enable   (enable),
                .data_in  (prod_full),
                .data_out (prod_r)
            );
            wire lane_valid_r;
            VX_pipe_register #(
                .DATAW (1),
                .DEPTH (PROD_REG)
            ) pipe_valid (
                .clk      (clk),
                .reset    (1'b0),
                .enable   (enable),
                .data_in  (lane_valid),
                .data_out (lane_valid_r)
            );
            assign y_prod_i8[j] = prod_r & {17{lane_valid_r}};
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

`ifdef VX_CFG_TCU_MX_ENABLE
        wire signed [24:0] y_mxi8_scaled [2];
        for (genvar j = 0; j < 2; ++j) begin : g_mxi8
            wire signed [24:0] raw_prod = {{8{y_prod_i8[j][16]}}, y_prod_i8[j]};
            assign y_mxi8_scaled[j] = shift_overflow ? 25'sd0
                                    : is_right_shift  ? (raw_prod >>> shift_amount)
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
            case ({1'b1, fmt_i_r})
                TCU_I8_ID: result[i] = 25'($signed(y_i8_add_res));
                TCU_U8_ID: result[i] = {8'b0, y_i8_add_res};
                default:   result[i] = '0;
            endcase
        end
    end

endmodule
