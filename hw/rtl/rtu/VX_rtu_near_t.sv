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
// operations rounded to nearest even, subnormals included (combinational).
// Two opaque hits within it of each other may be ordered either way by the
// source BVH's F32 box cull, so the walker settles them by its visit order.
//
// |t| * 2^-19 is exact unless it lands in the subnormal range, so the sum is
// formed as one integer W * 2^p (the scaled term pre-rounded where it is not
// exact) and rounded once.

`include "VX_define.vh"

module VX_rtu_near_t (
    input  wire [31:0] t,
    output wire [31:0] result
);
    localparam WB = 44;

    wire        s = t[31];
    wire [7:0]  e = t[30:23];
    wire [22:0] f = t[22:0];

    wire is_nan  = (e == 8'hff) && (f != 23'd0);
    wire is_inf  = (e == 8'hff) && (f == 23'd0);
    wire is_zero = (e == 8'h00) && (f == 23'd0);

    // |t| = m * 2^q, q = e - 150 (normal) or -149 (subnormal)
    wire [23:0] m = {(e != 8'h00), f};
    // |t| * 2^-19 = (m / 2^d) * 2^p: d > 0 only when it is subnormal, where
    // its integer m / 2^d is rounded as the F32 multiply rounds it
    wire [4:0]  d = (e >= 8'd20) ? 5'd0 : ((e == 8'h00) ? 5'd19 : 5'(8'd20 - e));
    wire signed [10:0] p = (d == 5'd0) ? (11'($signed({1'b0, e})) - 11'sd169) : -11'sd149;

    wire [23:0] q0  = m >> d;
    wire [4:0]  gi  = (d == 5'd0) ? 5'd0 : (d - 5'd1);
    wire        g   = (d != 5'd0) && m[gi];
    wire        st  = (m & ((24'd1 << gi) - 24'd1)) != 24'd0;
    wire [23:0] y_int = q0 + 24'(g && (st || q0[0]));

    wire [WB-1:0] m_al = WB'(m) << (5'd19 - d);
    wire [WB-1:0] w    = s ? (m_al - WB'(y_int)) : (m_al + WB'(y_int));

    wire [30:0] mag;
    VX_rtu_f32_round #(
        .WB (WB),
        .EW (11)
    ) round (
        .mag    (w),
        .exp    (p),
        .sticky (1'b0),
        .result (mag)
    );

    assign result = is_nan  ? (t | 32'h00400000)
                  : is_inf  ? (s ? 32'hffc00000 : 32'h7f800000)
                  : is_zero ? 32'h00000000
                  :           {s, mag};

endmodule
