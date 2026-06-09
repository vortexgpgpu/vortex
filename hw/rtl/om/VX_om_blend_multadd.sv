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

module VX_om_blend_multadd import VX_om_pkg::*; #(
    parameter LATENCY = 1
) (
    input wire clk,
    input wire reset,

    input wire enable,

    input wire [`VX_OM_BLEND_MODE_BITS-1:0] mode_rgb,
    input wire [`VX_OM_BLEND_MODE_BITS-1:0] mode_a,

    input om_color_t src_color,
    input om_color_t dst_color,

    input om_color_t src_factor,
    input om_color_t dst_factor,

    output om_color_t color_out
);

    `STATIC_ASSERT((LATENCY == 3), ("invalid parameter"))
    `UNUSED_VAR (reset)

    // multiply-add

    reg [15:0] prod_src_r, prod_src_g, prod_src_b, prod_src_a;
    reg [15:0] prod_dst_r, prod_dst_g, prod_dst_b, prod_dst_a;
    reg [16:0] sum_r, sum_g, sum_b, sum_a;

    always @(posedge clk) begin
        if (enable) begin
            prod_src_r <= src_color.argb[23:16] * src_factor.argb[23:16];
            prod_src_g <= src_color.argb[15:8] * src_factor.argb[15:8];
            prod_src_b <= src_color.argb[7:0] * src_factor.argb[7:0];
            prod_src_a <= src_color.argb[31:24] * src_factor.argb[31:24];

            prod_dst_r <= dst_color.argb[23:16] * dst_factor.argb[23:16];
            prod_dst_g <= dst_color.argb[15:8] * dst_factor.argb[15:8];
            prod_dst_b <= dst_color.argb[7:0] * dst_factor.argb[7:0];
            prod_dst_a <= dst_color.argb[31:24] * dst_factor.argb[31:24];

            case (mode_rgb)
                `VX_OM_BLEND_MODE_ADD: begin
                    sum_r <= prod_src_r + prod_dst_r + 16'h80;
                    sum_g <= prod_src_g + prod_dst_g + 16'h80;
                    sum_b <= prod_src_b + prod_dst_b + 16'h80;
                end
                `VX_OM_BLEND_MODE_SUB: begin
                    sum_r <= prod_src_r - prod_dst_r + 16'h80;
                    sum_g <= prod_src_g - prod_dst_g + 16'h80;
                    sum_b <= prod_src_b - prod_dst_b + 16'h80;
                end
                `VX_OM_BLEND_MODE_REV_SUB: begin
                    sum_r <= prod_dst_r - prod_src_r + 16'h80;
                    sum_g <= prod_dst_g - prod_src_g + 16'h80;
                    sum_b <= prod_dst_b - prod_src_b + 16'h80;
                end
            endcase
            case (mode_a)
                `VX_OM_BLEND_MODE_ADD: begin
                    sum_a <= prod_src_a + prod_dst_a + 16'h80;
                end
                `VX_OM_BLEND_MODE_SUB: begin
                    sum_a <= prod_src_a - prod_dst_a + 16'h80;
                end
                `VX_OM_BLEND_MODE_REV_SUB: begin
                    sum_a <= prod_dst_a - prod_src_a + 16'h80;
                end
            endcase
        end
    end

    // clamp to (0, 255 * 256)

    reg [15:0] clamp_r, clamp_g, clamp_b, clamp_a;

    always @(*) begin
        case (mode_rgb)
            `VX_OM_BLEND_MODE_ADD: begin
                clamp_r = (sum_r > 17'hFF00) ? 16'hFF00 : sum_r[15:0];
                clamp_g = (sum_g > 17'hFF00) ? 16'hFF00 : sum_g[15:0];
                clamp_b = (sum_b > 17'hFF00) ? 16'hFF00 : sum_b[15:0];
            end
            `VX_OM_BLEND_MODE_SUB,
            `VX_OM_BLEND_MODE_REV_SUB: begin
                clamp_r = sum_r[16] ? 16'h0 : sum_r[15:0];
                clamp_g = sum_g[16] ? 16'h0 : sum_g[15:0];
                clamp_b = sum_b[16] ? 16'h0 : sum_b[15:0];
            end
            default: begin
                clamp_r = 'x;
                clamp_g = 'x;
                clamp_b = 'x;
            end
        endcase
        case (mode_a)
            `VX_OM_BLEND_MODE_ADD: begin
                clamp_a = (sum_a > 17'hFF00) ? 16'hFF00 : sum_a[15:0];
            end
            `VX_OM_BLEND_MODE_SUB,
            `VX_OM_BLEND_MODE_REV_SUB: begin
                clamp_a = sum_a[16] ? 16'h0 : sum_a[15:0];
            end
            default: begin
                clamp_a = 'x;
            end
        endcase
    end

    // divide by 255

    om_color_t result;
    assign result.argb[31:24] = 8'((clamp_a + (clamp_a >> 8)) >> 8);
    assign result.argb[23:16] = 8'((clamp_r + (clamp_r >> 8)) >> 8);
    assign result.argb[15:8]  = 8'((clamp_g + (clamp_g >> 8)) >> 8);
    assign result.argb[7:0]   = 8'((clamp_b + (clamp_b >> 8)) >> 8);

    VX_pipe_register #(
        .DATAW (32)
    ) pipe_reg (
        .clk      (clk),
        `UNUSED_PIN (reset),
        .enable   (enable),
        .data_in  (result),
        .data_out (color_out)
    );

endmodule
