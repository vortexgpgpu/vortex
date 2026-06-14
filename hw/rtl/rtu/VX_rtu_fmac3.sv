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
// ONCE. Finite inputs only; subnormals flushed to zero. Deeply pipelined
// (align | add | lzc | normalize | round) to hold 300 MHz; latency = 5. Shared
// by VX_rtu_fdot3 (3 terms) and VX_rtu_fcross3 (2 terms per axis).

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

    // ── stage 1: align each term to the max product exponent ─────────────
    wire [8:0] pe01   = (pe[0] > pe[1]) ? pe[0] : pe[1];
    wire [8:0] max_pe = (pe01 > pe[2]) ? pe01 : pe[2];

    wire signed [SW:0] term [3];
    for (genvar i = 0; i < 3; ++i) begin : g_align
        wire [8:0]    sh    = max_pe - pe[i];
        wire [FW-1:0] field = ({{(FW-PW){1'b0}}, prod[i]} << GW) >> sh;
        wire signed [SW:0] fext = $signed({{(SW+1-FW){1'b0}}, field});
        assign term[i] = sign[i] ? -fext : fext;
    end

    wire signed [SW:0] s1_t0, s1_t1, s1_t2;
    wire [8:0]         s1_max_pe;
    VX_pipe_register #(.DATAW (3*(SW+1) + 9), .DEPTH (1)) p1 (
        .clk (clk), .reset (reset), .enable (enable),
        .data_in  ({term[0], term[1], term[2], max_pe}),
        .data_out ({s1_t0,   s1_t1,   s1_t2,   s1_max_pe})
    );

    // ── stage 2: signed sum ──────────────────────────────────────────────
    wire signed [SW:0] sum = s1_t0 + s1_t1 + s1_t2;

    wire signed [SW:0] s2_sum;
    wire [8:0]         s2_max_pe;
    VX_pipe_register #(.DATAW (SW+1 + 9), .DEPTH (1)) p2 (
        .clk (clk), .reset (reset), .enable (enable),
        .data_in  ({sum, s1_max_pe}),
        .data_out ({s2_sum, s2_max_pe})
    );

    // ── stage 3: sign/abs + leading-zero count ───────────────────────────
    wire          neg  = s2_sum[SW];
    wire [SW-1:0] absS = neg ? (~s2_sum[SW-1:0] + 1'b1) : s2_sum[SW-1:0];

    wire [LZW-1:0] lz;
    wire           lz_valid;
    VX_lzc #(.N (SW)) lzc_i (.data_in (absS), .data_out (lz), .valid_out (lz_valid));

    wire [SW-1:0]  s3_abs;
    wire [LZW-1:0] s3_lz;
    wire           s3_lzv, s3_neg;
    wire [8:0]     s3_max_pe;
    VX_pipe_register #(.DATAW (SW + LZW + 1 + 1 + 9), .DEPTH (1)) p3 (
        .clk (clk), .reset (reset), .enable (enable),
        .data_in  ({absS, lz, lz_valid, neg, s2_max_pe}),
        .data_out ({s3_abs, s3_lz, s3_lzv, s3_neg, s3_max_pe})
    );

    // ── stage 4: normalize, extract mantissa + GRS, base exponent ────────
    wire [SW-1:0] norm  = s3_abs << s3_lz;       // leading 1 at bit SW-1
    wire [23:0]   mant  = norm[SW-1 -: 24];
    wire          g_bit = norm[SW-1-24];
    wire          r_bit = norm[SW-1-25];
    wire          stky  = |norm[SW-1-26 : 0];
    wire          zero  = (s3_abs == '0);
    // biased exponent before any rounding carry
    wire signed [10:0] rexp_b = $signed({2'b0, s3_max_pe}) - 11'sd124
                              - $signed({{(11-LZW){1'b0}}, s3_lz});

    wire [23:0]        s4_mant;
    wire               s4_g, s4_r, s4_s, s4_zero, s4_lzv, s4_neg;
    wire signed [10:0] s4_rexp;
    VX_pipe_register #(.DATAW (24 + 6 + 11), .DEPTH (1)) p4 (
        .clk (clk), .reset (reset), .enable (enable),
        .data_in  ({mant, g_bit, r_bit, stky, zero, s3_lzv, s3_neg, rexp_b}),
        .data_out ({s4_mant, s4_g, s4_r, s4_s, s4_zero, s4_lzv, s4_neg, s4_rexp})
    );

    // ── stage 5: round (RNE) + pack ──────────────────────────────────────
    wire round_up = s4_g & (s4_r | s4_s | s4_mant[0]);
    wire [24:0] mant_r = s4_mant + round_up;
    wire        carry  = mant_r[24];
    wire [22:0] frac   = carry ? mant_r[23:1] : mant_r[22:0];
    wire signed [10:0] rexp = s4_rexp + $signed({10'd0, carry});

    reg [31:0] res;
    always @(*) begin
        if (s4_zero || !s4_lzv || rexp <= 11'sd0)
            res = {s4_neg, 31'd0};
        else if (rexp >= 11'sd255)
            res = {s4_neg, 8'hFF, 23'd0};
        else
            res = {s4_neg, rexp[7:0], frac};
    end

    VX_pipe_register #(.DATAW (32), .DEPTH (1)) p5 (
        .clk (clk), .reset (reset), .enable (enable),
        .data_in (res), .data_out (result)
    );

endmodule
