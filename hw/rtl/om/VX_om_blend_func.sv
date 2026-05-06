//!/bin/bash

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

`include "VX_om_define.vh"

module VX_blend_func #(
    parameter INDEX = 0
) (
    input wire [`VX_OM_BLEND_FUNC_BITS-1:0] func,
    input wire [3:0][7:0]  src,
    input wire [3:0][7:0]  dst,
    input wire [3:0][7:0]  cst,
    output wire [7:0]      result
);

    wire [7:0] one_minus_dst_a = 8'hFF - dst[3];

    reg [7:0] result_r;

    always @(*) begin
        case (func)
            `VX_OM_BLEND_FUNC_ZERO:                 result_r = 8'h0;
            `VX_OM_BLEND_FUNC_ONE:                  result_r = 8'hFF;
            `VX_OM_BLEND_FUNC_SRC_RGB:              result_r = src[INDEX];
            `VX_OM_BLEND_FUNC_ONE_MINUS_SRC_RGB:    result_r = 8'hFF - src[INDEX];
            `VX_OM_BLEND_FUNC_SRC_A:                result_r = src[3];
            `VX_OM_BLEND_FUNC_ONE_MINUS_SRC_A:      result_r = 8'hFF - src[3];
            `VX_OM_BLEND_FUNC_DST_RGB:              result_r = dst[INDEX];
            `VX_OM_BLEND_FUNC_ONE_MINUS_DST_RGB:    result_r = 8'hFF - dst[INDEX];
            `VX_OM_BLEND_FUNC_DST_A:                result_r = dst[3];
            `VX_OM_BLEND_FUNC_ONE_MINUS_DST_A:      result_r = one_minus_dst_a;
            `VX_OM_BLEND_FUNC_CONST_RGB:            result_r = cst[INDEX];
            `VX_OM_BLEND_FUNC_ONE_MINUS_CONST_RGB:  result_r = 8'hFF - cst[INDEX];
            `VX_OM_BLEND_FUNC_CONST_A:              result_r = cst[3];
            `VX_OM_BLEND_FUNC_ONE_MINUS_CONST_A:    result_r = 8'hFF - cst[3];
            `VX_OM_BLEND_FUNC_ALPHA_SAT: begin
                if (INDEX < 3) begin
                    result_r = (src[3] < one_minus_dst_a) ? src[3] : one_minus_dst_a;
                end else begin
                    result_r = 8'hFF;
                end
            end
            default:                              result_r = 8'hx;
        endcase
    end

    assign result = result_r;

endmodule

module VX_om_blend_func import VX_om_pkg::*; #(
    //--
) (
    input wire [`VX_OM_BLEND_FUNC_BITS-1:0] func_rgb,
    input wire [`VX_OM_BLEND_FUNC_BITS-1:0] func_a,

    input om_color_t src_color,
    input om_color_t dst_color,
    input om_color_t cst_color,

    output om_color_t factor_out
);
    VX_blend_func #(0) blend_func_b (func_rgb, src_color, dst_color, cst_color, factor_out.argb[7:0]);
    VX_blend_func #(1) blend_func_g (func_rgb, src_color, dst_color, cst_color, factor_out.argb[15:8]);
    VX_blend_func #(2) blend_func_r (func_rgb, src_color, dst_color, cst_color, factor_out.argb[23:16]);
    VX_blend_func #(3) blend_func_a (func_a,   src_color, dst_color, cst_color, factor_out.argb[31:24]);

endmodule
