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

// VX_rtu_fmac3 — fused sum of up to three signed fp32 products. Each term is a
// pre-formed 48-bit mantissa product with a 9-bit product exponent (ea+eb) and
// a sign; an unused term passes pe=0/prod=0. The terms are aligned to the common
// (max) exponent, summed in extended precision, then normalized and rounded
// ONCE. Finite inputs only; subnormals flushed to zero. result latency = 3.
// Shared by VX_rtu_fdot3 (3 terms) and VX_rtu_fcross3 (2 terms per axis).

`include "VX_define.vh"

module VX_rtu_fmac3 #(
    parameter PW = 48
) (
    input  wire             clk,
    input  wire             reset,
    input  wire             enable,
    input  wire [2:0]       sign,
    input  wire [2:0][8:0]  pe,      // product exponent ea+eb (0 => unused term)
    input  wire [2:0][PW-1:0] prod,  // 48-bit mantissa product
    output wire [31:0]      result
);
    localparam GW  = 32;             // guard bits below the product
    localparam FW  = PW + GW;        // aligned field width
    localparam SW  = FW + 2;         // signed-sum magnitude width
    localparam LZW = `LOG2UP(SW);

    // ── stage 0: align to max product exponent, signed sum ───────────────
    wire [8:0] pe01   = (pe[0] > pe[1]) ? pe[0] : pe[1];
    wire [8:0] max_pe = (pe01 > pe[2]) ? pe01 : pe[2];

    wire signed [SW:0] term [3];
    for (genvar i = 0; i < 3; ++i) begin : g_align
        wire [8:0]    sh    = max_pe - pe[i];
        wire [FW-1:0] field = ({{(FW-PW){1'b0}}, prod[i]} << GW) >> sh;
        wire signed [SW:0] fext = $signed({{(SW+1-FW){1'b0}}, field});
        assign term[i] = sign[i] ? -fext : fext;
    end
    wire signed [SW:0] sum = term[0] + term[1] + term[2];

    wire [9+SW:0] s1_data;
    VX_pipe_register #(.DATAW (9 + SW + 1), .DEPTH (1)) p0 (
        .clk (clk), .reset (reset), .enable (enable),
        .data_in  ({max_pe, sum}),
        .data_out (s1_data)
    );
    wire [8:0]         s1_max_pe = s1_data[9+SW -: 9];
    wire signed [SW:0] s1_sum    = s1_data[SW:0];

    // ── stage 1: sign/abs, normalize, round (RNE) ────────────────────────
    wire          neg  = s1_sum[SW];
    wire [SW-1:0] absS = neg ? (~s1_sum[SW-1:0] + 1'b1) : s1_sum[SW-1:0];
    wire          zero = (absS == '0);

    wire [LZW-1:0] lz;
    wire           lz_valid;
    VX_lzc #(.N (SW)) lzc_i (.data_in (absS), .data_out (lz), .valid_out (lz_valid));

    wire [SW-1:0] norm  = absS << lz;          // leading 1 at bit SW-1
    wire [23:0]   mant  = norm[SW-1 -: 24];
    wire          g_bit = norm[SW-1-24];
    wire          r_bit = norm[SW-1-25];
    wire          stky  = |norm[SW-1-26 : 0];

    wire round_up = g_bit & (r_bit | stky | mant[0]);
    wire [24:0] mant_r = mant + round_up;
    wire        carry  = mant_r[24];
    wire [22:0] frac   = carry ? mant_r[23:1] : mant_r[22:0];

    wire signed [10:0] rexp = $signed({2'b0, s1_max_pe}) - 11'sd124
                            - $signed({{(11-LZW){1'b0}}, lz}) + $signed({10'd0, carry});

    reg [31:0] res;
    always @(*) begin
        if (zero || !lz_valid || rexp <= 11'sd0)
            res = {neg, 31'd0};
        else if (rexp >= 11'sd255)
            res = {neg, 8'hFF, 23'd0};
        else
            res = {neg, rexp[7:0], frac};
    end

    VX_pipe_register #(.DATAW (32), .DEPTH (1)) p1 (
        .clk (clk), .reset (reset), .enable (enable),
        .data_in (res), .data_out (result)
    );

endmodule
