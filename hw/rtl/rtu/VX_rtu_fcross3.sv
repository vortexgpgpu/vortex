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

// VX_rtu_fcross3 — pipelined fp32 3-vector cross product result = a × b.
//   result[i] = a[(i+1)%3]*b[(i+2)%3] - a[(i+2)%3]*b[(i+1)%3]
// The subtracted product is computed first, then fused into a single a*b-c
// per axis. Result latency = 2*LATENCY_FMA.

`include "VX_define.vh"

module VX_rtu_fcross3 import VX_gpu_pkg::*, VX_fpu_pkg::*; #(
    parameter LATENCY_FMA = `VX_CFG_LATENCY_FMA
) (
    input  wire             clk,
    input  wire             reset,
    input  wire             enable,
    input  wire [2:0][31:0] a,
    input  wire [2:0][31:0] b,
    output wire [2:0][31:0] result
);
    localparam [INST_FMT_BITS-1:0] FMT_ADD = 2'b00;   // F32, a*b + c
    localparam [INST_FMT_BITS-1:0] FMT_SUB = 2'b10;   // F32, a*b - c

    for (genvar i = 0; i < 3; ++i) begin : g_axis
        localparam I1 = (i + 1) % 3;
        localparam I2 = (i + 2) % 3;

        // prod = a[i2] * b[i1]
        wire [31:0] prod;
        VX_fma_unit #(.LATENCY (LATENCY_FMA)) fma_mul (
            .clk (clk), .reset (reset), .enable (enable), .mask (1'b1),
            .op_type (INST_FPU_MADD), .fmt (FMT_ADD), .frm (INST_FRM_RNE),
            .dataa (a[I2]), .datab (b[I1]), .datac (32'h0),
            .result (prod), `UNUSED_PIN (fflags)
        );

        // result[i] = a[i1] * b[i2] - prod   (a[i1]/b[i2] aligned to prod)
        wire [31:0] a1_d, b2_d;
        VX_shift_register #(.DATAW (64), .DEPTH (LATENCY_FMA)) sr (
            .clk (clk), .reset (reset), .enable (enable),
            .data_in ({a[I1], b[I2]}), .data_out ({a1_d, b2_d})
        );
        VX_fma_unit #(.LATENCY (LATENCY_FMA)) fma_sub (
            .clk (clk), .reset (reset), .enable (enable), .mask (1'b1),
            .op_type (INST_FPU_MADD), .fmt (FMT_SUB), .frm (INST_FRM_RNE),
            .dataa (a1_d), .datab (b2_d), .datac (prod),
            .result (result[i]), `UNUSED_PIN (fflags)
        );
    end

endmodule
