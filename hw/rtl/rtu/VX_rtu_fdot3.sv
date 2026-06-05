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

// VX_rtu_fdot3 — pipelined fp32 3-vector dot product result = a·b. A fused
// multiply-accumulate chain (m0 = a.x*b.x, then +a.y*b.y, then +a.z*b.z) with
// the later operands delayed so each VX_fma_unit consumes its accumulator and
// its freshly-aligned inputs on the same cycle. Result latency = 3*LATENCY_FMA.

`include "VX_define.vh"

module VX_rtu_fdot3 import VX_gpu_pkg::*, VX_fpu_pkg::*; #(
    parameter LATENCY_FMA = `VX_CFG_LATENCY_FMA
) (
    input  wire             clk,
    input  wire             reset,
    input  wire             enable,
    input  wire [2:0][31:0] a,
    input  wire [2:0][31:0] b,
    output wire [31:0]      result
);
    localparam [INST_FMT_BITS-1:0] FMT_ADD = 2'b00;   // F32, a*b + c

    // m0 = a.x * b.x
    wire [31:0] m0;
    VX_fma_unit #(.LATENCY (LATENCY_FMA)) fma0 (
        .clk (clk), .reset (reset), .enable (enable), .mask (1'b1),
        .op_type (INST_FPU_MADD), .fmt (FMT_ADD), .frm (INST_FRM_RNE),
        .dataa (a[0]), .datab (b[0]), .datac (32'h0),
        .result (m0), `UNUSED_PIN (fflags)
    );

    // s1 = a.y * b.y + m0   (a.y/b.y aligned to m0)
    wire [31:0] a1_d, b1_d, s1;
    VX_shift_register #(.DATAW (64), .DEPTH (LATENCY_FMA)) sr1 (
        .clk (clk), .reset (reset), .enable (enable),
        .data_in ({a[1], b[1]}), .data_out ({a1_d, b1_d})
    );
    VX_fma_unit #(.LATENCY (LATENCY_FMA)) fma1 (
        .clk (clk), .reset (reset), .enable (enable), .mask (1'b1),
        .op_type (INST_FPU_MADD), .fmt (FMT_ADD), .frm (INST_FRM_RNE),
        .dataa (a1_d), .datab (b1_d), .datac (m0),
        .result (s1), `UNUSED_PIN (fflags)
    );

    // result = a.z * b.z + s1   (a.z/b.z aligned to s1)
    wire [31:0] a2_d, b2_d;
    VX_shift_register #(.DATAW (64), .DEPTH (2 * LATENCY_FMA)) sr2 (
        .clk (clk), .reset (reset), .enable (enable),
        .data_in ({a[2], b[2]}), .data_out ({a2_d, b2_d})
    );
    VX_fma_unit #(.LATENCY (LATENCY_FMA)) fma2 (
        .clk (clk), .reset (reset), .enable (enable), .mask (1'b1),
        .op_type (INST_FPU_MADD), .fmt (FMT_ADD), .frm (INST_FRM_RNE),
        .dataa (a2_d), .datab (b2_d), .datac (s1),
        .result (result), `UNUSED_PIN (fflags)
    );

endmodule
