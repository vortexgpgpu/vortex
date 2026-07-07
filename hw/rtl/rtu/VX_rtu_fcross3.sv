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

// VX_rtu_fcross3 — fused fp32 3-vector cross product result = a × b.
//   result[i] = a[(i+1)%3]*b[(i+2)%3] - a[(i+2)%3]*b[(i+1)%3]
// Each axis forms its two lane products and sums them (the second negated) in a
// shared VX_rtu_fmac3, normalizing+rounding ONCE rather than per FMA. Inputs in
// the RTU geometry path are finite; subnormals flushed to zero. Latency padded
// to 2*LATENCY_FMA so the consuming PE keeps its side-band alignment.

`include "VX_define.vh"

module VX_rtu_fcross3 import VX_gpu_pkg::*, VX_fpu_pkg::*; #(
    parameter LATENCY_FMA = `VX_CFG_FMA_LATENCY,
    parameter LATENCY     = 2 * LATENCY_FMA
) (
    input  wire             clk,
    input  wire             reset,
    input  wire             enable,
    input  wire [2:0][31:0] a,
    input  wire [2:0][31:0] b,
    output wire [2:0][31:0] result
);
    for (genvar i = 0; i < 3; ++i) begin : g_axis
        localparam I1 = (i + 1) % 3;
        localparam I2 = (i + 2) % 3;

        // term0 = +a[I1]*b[I2], term1 = -a[I2]*b[I1]
        wire [7:0]  e0a = a[I1][30:23], e0b = b[I2][30:23];
        wire        z0a = (e0a == 8'd0), z0b = (e0b == 8'd0);
        wire [23:0] m0a = z0a ? 24'd0 : {1'b1, a[I1][22:0]};
        wire [23:0] m0b = z0b ? 24'd0 : {1'b1, b[I2][22:0]};

        wire [7:0]  e1a = a[I2][30:23], e1b = b[I1][30:23];
        wire        z1a = (e1a == 8'd0), z1b = (e1b == 8'd0);
        wire [23:0] m1a = z1a ? 24'd0 : {1'b1, a[I2][22:0]};
        wire [23:0] m1b = z1b ? 24'd0 : {1'b1, b[I1][22:0]};

        wire [2:0]       m_sign = {1'b0,
                                   ~(a[I2][31] ^ b[I1][31]),   // negated (subtraction)
                                     (a[I1][31] ^ b[I2][31])};
        wire [2:0][8:0]  m_pe   = {9'd0,
                                   (z1a | z1b) ? 9'd0 : ({1'b0, e1a} + {1'b0, e1b}),
                                   (z0a | z0b) ? 9'd0 : ({1'b0, e0a} + {1'b0, e0b})};
        wire [47:0]      pp0    = m0a * m0b;   // full 48-bit products
        wire [47:0]      pp1    = m1a * m1b;
        wire [2:0][47:0] m_prod = {48'd0, pp1, pp0};

        wire [2:0]       q_sign;
        wire [2:0][8:0]  q_pe;
        wire [2:0][47:0] q_prod;
        VX_pipe_register #(
            .DATAW (3 + 3*9 + 3*48),
            .DEPTH (1)
        ) p0 (
            .clk      (clk),
            .reset    (reset),
            .enable   (enable),
            .data_in  ({m_sign, m_pe, m_prod}),
            .data_out ({q_sign, q_pe, q_prod})
        );

        wire [31:0] crs;
        VX_rtu_fmac3 mac (
            .clk    (clk),
            .reset  (reset),
            .enable (enable),
            .sign   (q_sign),
            .pe     (q_pe),
            .prod   (q_prod),
            .result (crs)
        );

        VX_shift_register #(
            .DATAW (32),
            .DEPTH (LATENCY - 8)
        ) sr_pad (
            .clk      (clk),
            .reset    (reset),
            .enable   (enable),
            .data_in  (crs),
            .data_out (result[i])
        );
    end

endmodule
