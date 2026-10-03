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

// VX_rtu_f32_round — rounds the positive value (mag + sticky) * 2^exp to F32,
// nearest even, subnormals and overflow to infinity included. `sticky` stands
// for a nonzero remainder below mag's LSB; callers keep at least two bits of
// mag below the result's LSB whenever it is set. Returns the magnitude bits
// {exponent, fraction}; mag == 0 gives +0. LATENCY 0 is combinational; 4
// registers the leading one, the alignment, the round increment and the
// result.

`include "VX_define.vh"

module VX_rtu_f32_round #(
    parameter WB      = 44,     // mag width
    parameter EW      = 11,     // signed exponent width
    parameter LATENCY = 0       // 0 or 4
) (
    input  wire                 clk,
    input  wire                 enable,
    input  wire [WB-1:0]        mag,
    input  wire signed [EW-1:0] exp,
    input  wire                 sticky,
    output wire [30:0]          result
);
    `STATIC_ASSERT(((LATENCY == 0) || (LATENCY == 4)), ("invalid LATENCY"))
    localparam IW = `CLOG2(WB);
    localparam BW = WB + 12;
    localparam EX = EW + 2;

    // ── stage 0: leading one, the LSB exponent and the shift ──────────
    reg [IW-1:0] msb;
    always @(*) begin
        msb = '0;
        for (integer i = 0; i < WB; ++i) begin
            if (mag[i]) msb = IW'(i);
        end
    end

    // the result LSB's exponent, clamped at the subnormal LSB 2^-149
    wire signed [EX-1:0] lsb_raw = EX'(exp) + EX'($signed({1'b0, msb})) - EX'(23);
    wire signed [EX-1:0] s0_lsb  = (lsb_raw < -EX'(149)) ? -EX'(149) : lsb_raw;
    wire signed [EX-1:0] s0_sh   = s0_lsb - EX'(exp);

    // ── stage 1: alignment, guard/sticky ──────────────────────────────
    wire [WB-1:0]        s1_mag;
    wire                 s1_sticky;
    wire signed [EX-1:0] lsb, sh;
    VX_pipe_register #(
        .DATAW (WB + 1 + 2 * EX),
        .DEPTH ((LATENCY != 0) ? 1 : 0)
    ) pipe0 (
        .clk      (clk),
        .reset    (1'b0),
        .enable   (enable),
        .data_in  ({mag, sticky, s0_lsb, s0_sh}),
        .data_out ({s1_mag, s1_sticky, lsb, sh})
    );

    // sh <= 0: exact, shifted up; sh > 0: drop sh bits, guard + sticky
    wire           exact = (sh <= 0);
    wire [IW-1:0]  ush   = exact ? IW'(-sh) : IW'(sh);
    wire [IW-1:0]  gi    = exact ? '0 : IW'(sh - EX'(1));
    wire [WB-1:0]  q     = s1_mag >> ush;
    wire [WB-1:0]  low   = s1_mag & ((WB'(1) << gi) - WB'(1));
    wire [BW-1:0]  s1_sig0 = exact ? (BW'(s1_mag) << ush) : BW'(q);
    wire           s1_inc  = !exact && s1_mag[gi] && ((low != '0) || s1_sticky || q[0]);
    wire [BW-1:0]  s1_ebits = BW'(lsb + EX'(149)) << 23;
    wire           s1_zero  = (s1_mag == '0);

    // ── stage 2: round increment ──────────────────────────────────────
    wire [BW-1:0] s2_sig0, s2_ebits;
    wire          s2_inc, s2_zero;
    VX_pipe_register #(
        .DATAW (2 * BW + 2),
        .DEPTH ((LATENCY != 0) ? 1 : 0)
    ) pipe1 (
        .clk      (clk),
        .reset    (1'b0),
        .enable   (enable),
        .data_in  ({s1_sig0, s1_ebits, s1_inc, s1_zero}),
        .data_out ({s2_sig0, s2_ebits, s2_inc, s2_zero})
    );
    wire [BW-1:0] s2_sig = s2_sig0 + BW'(s2_inc);

    // ── stage 3: {biased exponent of the LSB position, significand}: a
    // carry out of the significand bumps the exponent, a subnormal reaching
    // 2^23 becomes normal ─────────────────────────────────────────────
    wire [BW-1:0] s3_sig, s3_ebits;
    wire          s3_zero;
    VX_pipe_register #(
        .DATAW (2 * BW + 1),
        .DEPTH ((LATENCY != 0) ? 1 : 0)
    ) pipe2 (
        .clk      (clk),
        .reset    (1'b0),
        .enable   (enable),
        .data_in  ({s2_sig, s2_ebits, s2_zero}),
        .data_out ({s3_sig, s3_ebits, s3_zero})
    );
    wire [BW-1:0] s3_bits = s3_ebits + s3_sig;
    wire [30:0]   s3_res  = s3_zero ? 31'd0
                          : (s3_bits >= BW'(32'h7f800000)) ? 31'h7f800000
                          : s3_bits[30:0];

    VX_pipe_register #(
        .DATAW (31),
        .DEPTH ((LATENCY != 0) ? 1 : 0)
    ) pipe3 (
        .clk      (clk),
        .reset    (1'b0),
        .enable   (enable),
        .data_in  (s3_res),
        .data_out (result)
    );

endmodule
