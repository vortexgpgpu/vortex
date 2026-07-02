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

// VX_rtu_recip — F32 reciprocal (1/x) for the ray-setup datapath (inv_d = 1/dir),
// with two interchangeable backends selected by BACKEND:
//
//   LUT_NR   : the portable default — a VX_fdiv_unit doing 1.0/x. Full
//              Newton-Raphson in LUTs, 0 DSP (the fabric-heavy baseline).
//   DSP_SEED : a seed-table reciprocal — a BlockRAM seed ROM gives ~10 bits of
//              1/significand, refined by two Newton-Raphson steps whose multiplies
//              map to DSP48. Trades ~2K LUT/unit onto the idle BRAM + DSP blocks.
//              ~9e-8 max relative error (well inside the RTU's 1e-4 tolerance).
//
// The input is presented combinationally and held stable for the whole setup
// span by the scheduler; the result is a fixed-latency pipeline output, valid
// after the backend's pipeline depth (<= the scheduler's SETUP_LAT wait).

`include "VX_define.vh"

module VX_rtu_recip import VX_gpu_pkg::*, VX_fpu_pkg::*; #(
    parameter LATENCY  = 17,                  // LUT_NR pipeline depth
    parameter DSP_SEED = 0                    // 0: LUT_NR (default) | 1: DSP_SEED
) (
    input  wire        clk,
    input  wire        reset,
    input  wire        enable,
    input  wire        mask,
    input  wire [31:0] x,                     // operand (dir component)
    output wire [31:0] result                 // 1 / x
);
    if (DSP_SEED != 0) begin : g_dsp_seed
        `UNUSED_VAR (mask)
        // ── seed ROM: 1/a for a = 1.fraction in [1,2), indexed by the top 10
        //    fraction bits, stored as Q1.31 (value = entry / 2^31, in (0.5,1]). ──
        localparam KIDX = 10;
        localparam SEED_N = 1 << KIDX;
        reg [31:0] seed_rom [0:SEED_N-1];
        initial begin
            for (int i = 0; i < SEED_N; i++) begin
                real a, y;
                a = 1.0 + (real'(i) + 0.5) / real'(SEED_N);
                y = (1.0 / a) * 2147483648.0;            // * 2^31
                seed_rom[i] = 32'($rtoi(y + 0.5));
            end
        end

        // S0: unpack the operand
        wire        s0_sign = x[31];
        wire [7:0]  s0_exp  = x[30:23];
        wire [22:0] s0_frac = x[22:0];
        wire [23:0] s0_A    = {1'b1, s0_frac};           // significand a*2^23
        wire [30:0] s0_afx  = {s0_A, 7'b0};              // a in Q2.30
        wire        s0_inf  = (s0_exp == 8'h00);         // 1/0 -> inf
        wire        s0_zero = (s0_exp == 8'hFF);         // 1/inf -> 0
        wire [KIDX-1:0] s0_idx = s0_frac[22 -: KIDX];

        reg        s1_sign, s1_inf, s1_zero;
        reg [7:0]  s1_exp;
        reg [30:0] s1_afx;
        reg [31:0] s1_y;                                 // seed, Q1.31
        always_ff @(posedge clk) if (enable) begin
            s1_sign <= s0_sign; s1_inf <= s0_inf; s1_zero <= s0_zero;
            s1_exp  <= s0_exp;  s1_afx <= s0_afx;
            s1_y    <= seed_rom[s0_idx];                 // registered ROM read -> BRAM
        end

        // NR step 1:  p = a*y (Q2.30);  t = 2 - p;  y = y*(2 - a*y)
        wire [62:0] s1_ay = s1_afx * s1_y;               // -> DSP (31b * 32b)
        wire [31:0] s1_p  = 32'(s1_ay >> 31);            // a*y  (Q2.30)
        wire [31:0] s1_t  = 32'h8000_0000 - s1_p;        // 2 - p  (2 == 2^31 in Q2.30)
        reg        s2_sign, s2_inf, s2_zero;
        reg [7:0]  s2_exp;
        reg [30:0] s2_afx;
        reg [31:0] s2_y0, s2_t;
        always_ff @(posedge clk) if (enable) begin
            s2_sign <= s1_sign; s2_inf <= s1_inf; s2_zero <= s1_zero;
            s2_exp  <= s1_exp;  s2_afx <= s1_afx;
            s2_y0   <= s1_y;    s2_t   <= s1_t;
        end
        wire [63:0] s2_yt = s2_y0 * s2_t;                // -> DSP (32b * 32b)
        wire [31:0] s2_y1 = 32'(s2_yt >> 30);            // y*(2-a*y)  (Q1.31)
        reg        s3_sign, s3_inf, s3_zero;
        reg [7:0]  s3_exp;
        reg [30:0] s3_afx;
        reg [31:0] s3_y1;
        always_ff @(posedge clk) if (enable) begin
            s3_sign <= s2_sign; s3_inf <= s2_inf; s3_zero <= s2_zero;
            s3_exp  <= s2_exp;  s3_afx <= s2_afx;  s3_y1 <= s2_y1;
        end

        // NR step 2
        wire [62:0] s3_ay = s3_afx * s3_y1;              // -> DSP (31b * 32b)
        wire [31:0] s3_p  = 32'(s3_ay >> 31);
        wire [31:0] s3_t  = 32'h8000_0000 - s3_p;
        reg        s4_sign, s4_inf, s4_zero;
        reg [7:0]  s4_exp;
        reg [31:0] s4_y1, s4_t;
        always_ff @(posedge clk) if (enable) begin
            s4_sign <= s3_sign; s4_inf <= s3_inf; s4_zero <= s3_zero;
            s4_exp  <= s3_exp;  s4_y1 <= s3_y1;   s4_t <= s3_t;
        end
        wire [63:0] s4_yt = s4_y1 * s4_t;                // -> DSP (32b * 32b)
        wire [31:0] s4_y2 = 32'(s4_yt >> 30);            // 1/a in Q1.31
        reg        s5_sign, s5_inf, s5_zero;
        reg [7:0]  s5_exp;
        reg [31:0] s5_y2;
        always_ff @(posedge clk) if (enable) begin
            s5_sign <= s4_sign; s5_inf <= s4_inf; s5_zero <= s4_zero;
            s5_exp  <= s4_exp;  s5_y2 <= s4_y2;
        end

        // Normalize + pack:  1/a in (0.5,1] -> significand 2/a in (1,2], exp-1.
        wire [32:0] s5_2y  = {s5_y2, 1'b0};              // 2y, value = s5_2y/2^31
        wire [31:0] s5_d   = s5_2y - 33'h8000_0000;      // (2y - 1)
        wire [23:0] s5_fr  = s5_d[31:8] + {23'b0, s5_d[7]};  // round to 23 frac bits
        wire [8:0]  s5_exf = 9'd253 - {1'b0, s5_exp};
        wire        s5_ovf = s5_fr[23];                  // 2y rounded up to 2.0
        wire [7:0]  s5_expf = s5_ovf ? (s5_exf[7:0] + 8'd1) : s5_exf[7:0];
        wire [22:0] s5_frac = s5_ovf ? 23'd0 : s5_fr[22:0];
        assign result = s5_inf  ? {s5_sign, 8'hFF, 23'd0}
                      : s5_zero ? {s5_sign, 8'h00, 23'd0}
                      :           {s5_sign, s5_expf, s5_frac};
        `UNUSED_PARAM (LATENCY)
    end else begin : g_lut_nr
        // portable baseline: 1.0 / x via the shared NR divide unit
        VX_fdiv_unit #(
            .LATENCY        (LATENCY),
            .SUBNORM_ENABLE (0),
            .EXCEPT_ENABLE  (0)
        ) u_recip (
            .clk     (clk),
            .reset   (reset),
            .enable  (enable),
            .mask    (mask),
            .fmt     ('0),
            .frm     (INST_FRM_RNE),
            .dataa   (32'h3F800000 /*1.0*/),
            .datab   (x),
            .result  (result),
            `UNUSED_PIN (fflags)
        );
    end

endmodule
