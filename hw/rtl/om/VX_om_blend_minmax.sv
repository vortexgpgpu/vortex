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

module VX_om_blend_minmax import VX_om_pkg::*; #(
    parameter LATENCY = 1
) (
    input wire clk,
    input wire reset,

    input wire enable,

    input om_color_t src_color,
    input om_color_t dst_color,

    output om_color_t min_out,
    output om_color_t max_out
);

    `UNUSED_VAR (reset)

    om_color_t tmp_min;
    om_color_t tmp_max;

    always @(*) begin
        if (src_color.argb[23:16] > dst_color.argb[23:16]) begin
            tmp_max.argb[23:16] = src_color.argb[23:16];
            tmp_min.argb[23:16] = dst_color.argb[23:16];
        end else begin
            tmp_max.argb[23:16] = dst_color.argb[23:16];
            tmp_min.argb[23:16] = src_color.argb[23:16];
        end

        if (src_color.argb[15:8] > dst_color.argb[15:8]) begin
            tmp_max.argb[15:8] = src_color.argb[15:8];
            tmp_min.argb[15:8] = dst_color.argb[15:8];
        end else begin
            tmp_max.argb[15:8] = dst_color.argb[15:8];
            tmp_min.argb[15:8] = src_color.argb[15:8];
        end

        if (src_color.argb[7:0] > dst_color.argb[7:0]) begin
            tmp_max.argb[7:0] = src_color.argb[7:0];
            tmp_min.argb[7:0] = dst_color.argb[7:0];
        end else begin
            tmp_max.argb[7:0] = dst_color.argb[7:0];
            tmp_min.argb[7:0] = src_color.argb[7:0];
        end

        if (src_color.argb[31:24] > dst_color.argb[31:24]) begin
            tmp_max.argb[31:24] = src_color.argb[31:24];
            tmp_min.argb[31:24] = dst_color.argb[31:24];
        end else begin
            tmp_max.argb[31:24] = dst_color.argb[31:24];
            tmp_min.argb[31:24] = src_color.argb[31:24];
        end
    end

    VX_shift_register #(
        .DATAW (32 + 32),
        .DEPTH (LATENCY)
    ) shift_reg (
        .clk      (clk),
        `UNUSED_PIN (reset),
        .enable   (enable),
        .data_in  ({tmp_max, tmp_min}),
        .data_out ({max_out, min_out})
    );

endmodule
