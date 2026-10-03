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

// VX_rtu_near_t — the near-hit window bound t + |t| * 2^-19 in F32, both
// operations rounded to nearest even, subnormals included. Two opaque hits
// within it of each other may be ordered either way by the source BVH's F32
// box cull, so the walker settles them by its visit order.
//
// |t| * 2^-19 is exact unless it lands in the subnormal range, so the sum is
// formed as one integer W * 2^p (the scaled term pre-rounded where it is not
// exact) and rounded once. LATENCY 0 is combinational; 6 registers the scaled
// term, the sum and the four rounding steps.

`include "VX_define.vh"

module VX_rtu_near_t #(
    parameter LATENCY = 0       // 0 or 6
) (
    input  wire        clk,
    input  wire        enable,
    input  wire [31:0] t,
    output wire [31:0] result
);
    `STATIC_ASSERT(((LATENCY == 0) || (LATENCY == 6)), ("invalid LATENCY"))
    localparam WB = 44;
    localparam PD = (LATENCY != 0) ? 1 : 0;

    // ── stage 1: decode, the scaled term ──────────────────────────────
    wire        s = t[31];
    wire [7:0]  e = t[30:23];
    wire [22:0] f = t[22:0];

    // the result when t is not a finite nonzero value
    wire        s1_spec = (e == 8'hff) || ((e == 8'h00) && (f == 23'd0));
    wire [31:0] s1_sval = (e != 8'hff) ? 32'h00000000
                        : (f != 23'd0) ? (t | 32'h00400000)
                        : (s ? 32'hffc00000 : 32'h7f800000);

    // |t| = m * 2^q, q = e - 150 (normal) or -149 (subnormal)
    wire [23:0] m = {(e != 8'h00), f};
    // |t| * 2^-19 = (m / 2^d) * 2^p: d > 0 only when it is subnormal, where
    // its integer m / 2^d is rounded as the F32 multiply rounds it
    wire [4:0]  d = (e >= 8'd20) ? 5'd0 : ((e == 8'h00) ? 5'd19 : 5'(8'd20 - e));
    wire signed [10:0] s1_p = (d == 5'd0) ? (11'($signed({1'b0, e})) - 11'sd169) : -11'sd149;

    wire [23:0] q0  = m >> d;
    wire [4:0]  gi  = (d == 5'd0) ? 5'd0 : (d - 5'd1);
    wire        g   = (d != 5'd0) && m[gi];
    wire        st  = (m & ((24'd1 << gi) - 24'd1)) != 24'd0;
    wire [23:0] s1_y   = q0 + 24'(g && (st || q0[0]));
    wire [WB-1:0] s1_mal = WB'(m) << (5'd19 - d);

    // ── stage 2: the exact sum ────────────────────────────────────────
    wire          s2_s, s2_spec;
    wire [31:0]   s2_sval;
    wire signed [10:0] s2_p;
    wire [23:0]   s2_y;
    wire [WB-1:0] s2_mal;
    VX_pipe_register #(
        .DATAW (1 + 1 + 32 + 11 + 24 + WB),
        .DEPTH (PD)
    ) pipe1 (
        .clk      (clk),
        .reset    (1'b0),
        .enable   (enable),
        .data_in  ({s, s1_spec, s1_sval, s1_p, s1_y, s1_mal}),
        .data_out ({s2_s, s2_spec, s2_sval, s2_p, s2_y, s2_mal})
    );
    wire [WB-1:0] s2_w = s2_s ? (s2_mal - WB'(s2_y)) : (s2_mal + WB'(s2_y));

    wire          s3_s, s3_spec;
    wire [31:0]   s3_sval;
    wire signed [10:0] s3_p;
    wire [WB-1:0] s3_w;
    VX_pipe_register #(
        .DATAW (1 + 1 + 32 + 11 + WB),
        .DEPTH (PD)
    ) pipe2 (
        .clk      (clk),
        .reset    (1'b0),
        .enable   (enable),
        .data_in  ({s2_s, s2_spec, s2_sval, s2_p, s2_w}),
        .data_out ({s3_s, s3_spec, s3_sval, s3_p, s3_w})
    );

    // ── stages 3..6: the rounding, with the special result alongside ──
    wire [30:0] mag;
    VX_rtu_f32_round #(
        .WB      (WB),
        .EW      (11),
        .LATENCY ((LATENCY != 0) ? 4 : 0)
    ) round (
        .clk    (clk),
        .enable (enable),
        .mag    (s3_w),
        .exp    (s3_p),
        .sticky (1'b0),
        .result (mag)
    );

    wire        o_s, o_spec;
    wire [31:0] o_sval;
    VX_pipe_register #(
        .DATAW (1 + 1 + 32),
        .DEPTH ((LATENCY != 0) ? 4 : 0)
    ) pipe_side (
        .clk      (clk),
        .reset    (1'b0),
        .enable   (enable),
        .data_in  ({s3_s, s3_spec, s3_sval}),
        .data_out ({o_s, o_spec, o_sval})
    );

    assign result = o_spec ? o_sval : {o_s, mag};

endmodule
