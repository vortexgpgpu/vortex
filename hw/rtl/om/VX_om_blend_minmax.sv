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

    input rgba_t src_color,
    input rgba_t dst_color,

    output rgba_t min_out,
    output rgba_t max_out
);

    `UNUSED_VAR (reset)

    rgba_t tmp_min;
    rgba_t tmp_max;

    always @(*) begin   
        if (src_color.r > dst_color.r) begin
            tmp_max.r = src_color.r;
            tmp_min.r = dst_color.r;
        end else begin
            tmp_max.r = dst_color.r;
            tmp_min.r = src_color.r;
        end

        if (src_color.g > dst_color.g) begin
            tmp_max.g = src_color.g;
            tmp_min.g = dst_color.g;
        end else begin
            tmp_max.g = dst_color.g;
            tmp_min.g = src_color.g;
        end

        if (src_color.b > dst_color.b) begin
            tmp_max.b = src_color.b;
            tmp_min.b = dst_color.b;
        end else begin
            tmp_max.b = dst_color.b;
            tmp_min.b = src_color.b;
        end

        if (src_color.a > dst_color.a) begin
            tmp_max.a = src_color.a;
            tmp_min.a = dst_color.a;
        end else begin
            tmp_max.a = dst_color.a;
            tmp_min.a = src_color.a;
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
