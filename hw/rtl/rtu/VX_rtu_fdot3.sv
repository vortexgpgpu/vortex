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

// VX_rtu_fdot3 — fused fp32 3-vector dot product result = a·b. The three lane
// products feed VX_rtu_fmac3, which aligns and sums them, then normalizes and
// rounds ONCE — versus a chain of three FMAs that each normalize+round. Inputs
// in the RTU geometry path are finite; subnormals are flushed to zero. Latency
// padded to 3*LATENCY_FMA so the consuming PEs keep their side-band alignment.

`include "VX_define.vh"

module VX_rtu_fdot3 import VX_gpu_pkg::*, VX_fpu_pkg::*; #(
    parameter LATENCY_FMA = `VX_CFG_LATENCY_FMA,
    parameter LATENCY     = 3 * LATENCY_FMA
) (
    input  wire             clk,
    input  wire             reset,
    input  wire             enable,
    input  wire [2:0][31:0] a,
    input  wire [2:0][31:0] b,
    output wire [31:0]      result
);
    wire [2:0]       m_sign;
    wire [2:0][8:0]  m_pe;
    wire [2:0][47:0] m_prod;
    for (genvar i = 0; i < 3; ++i) begin : g_mul
        wire [7:0]  ea = a[i][30:23], eb = b[i][30:23];
        wire        az = (ea == 8'd0), bz = (eb == 8'd0);
        wire [23:0] ma = az ? 24'd0 : {1'b1, a[i][22:0]};
        wire [23:0] mb = bz ? 24'd0 : {1'b1, b[i][22:0]};
        assign m_sign[i] = a[i][31] ^ b[i][31];
        assign m_pe[i]   = (az | bz) ? 9'd0 : ({1'b0, ea} + {1'b0, eb});
        assign m_prod[i] = ma * mb;
    end

    wire [2:0]       q_sign;
    wire [2:0][8:0]  q_pe;
    wire [2:0][47:0] q_prod;
    VX_pipe_register #(.DATAW (3 + 3*9 + 3*48), .DEPTH (1)) p0 (
        .clk (clk), .reset (reset), .enable (enable),
        .data_in  ({m_sign, m_pe, m_prod}),
        .data_out ({q_sign, q_pe, q_prod})
    );

    wire [31:0] dot;
    VX_rtu_fmac3 mac (
        .clk (clk), .reset (reset), .enable (enable),
        .sign (q_sign), .pe (q_pe), .prod (q_prod), .result (dot)
    );

    VX_shift_register #(.DATAW (32), .DEPTH (LATENCY - 3)) sr_pad (
        .clk (clk), .reset (reset), .enable (enable),
        .data_in (dot), .data_out (result)
    );

endmodule
