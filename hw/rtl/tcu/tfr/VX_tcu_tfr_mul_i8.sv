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
    parameter TCK = 2 * N
) (
    input wire                      clk,
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
    `UNUSED_VAR ({clk, req_id, valid_in})
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

    `ifdef VX_CFG_TCU_MX_ENABLE
        // MXINT8 scaling
        wire signed [8:0] combined_sf = $signed(sf_a + sf_b - 9'd266);
        wire signed [24:0] y_mxi8_scaled [2];
        for (genvar j = 0; j < 2; ++j) begin : g_mxi8
            wire signed [24:0] raw_prod = {{8{y_prod_i8[j][16]}}, y_prod_i8[j]};
            wire        [8:0]  shift_amt = -combined_sf;
            wire        [24:0] abs_prod = raw_prod[24] ? -raw_prod : raw_prod;
            wire signed [24:0] scaled_prod_r = abs_prod >> shift_amt;
            wire signed [24:0] scaled_prod = raw_prod[24] ? -scaled_prod_r : scaled_prod_r;
            assign y_mxi8_scaled[j] = combined_sf[8] ? scaled_prod : (raw_prod <<< combined_sf);
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
    `endif

        // Output muxing
        always_comb begin
            case ({1'b1, fmt_i})
                TCU_I8_ID: result[i] = 25'($signed(y_i8_add_res));
                TCU_U8_ID: result[i] = {8'b0, y_i8_add_res};
            `ifdef VX_CFG_TCU_MX_ENABLE
                TCU_MXI8_ID: result[i] = y_mxi8_add_res;
            `endif
                default:   result[i] = '0;
            endcase
        end
    end

endmodule
