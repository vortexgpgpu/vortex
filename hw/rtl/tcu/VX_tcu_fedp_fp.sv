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

`include "VX_define.vh"

module VX_tcu_fedp_fp #(
    parameter DATAW = 32,
    parameter N     = 2
) (
    input  wire clk,
    input  wire reset,
    input  wire enable,

    input  wire[2:0] fmt_s,
    input  wire[2:0] fmt_d,

    input  wire [N-1:0][`XLEN-1:0] a_row,
    input  wire [N-1:0][`XLEN-1:0] b_col,
    input  wire [`XLEN-1:0] c_val,
    output wire [`XLEN-1:0] d_val
);
    wire [DATAW-1:0] a_row_p [0:N-1];
    wire [DATAW-1:0] b_col_p [0:N-1];

    wire [2:0] fmt_s_p [0:N-1];
    wire [2:0] fmt_d_p [0:N-1];

    `UNUSED_VAR ({a_row, b_col, c_val});

    for (genvar i = 0; i < N; i++) begin: g_pipe
        VX_pipe_register #(
            .DATAW (DATAW),
            .DEPTH (i * `LATENCY_FMA)
        ) a_pipe (
            .clk      (clk),
            .reset    (reset),
            .enable   (enable),
            .data_in  (a_row[i][DATAW-1:0]),
            .data_out (a_row_p[i])
        );
        VX_pipe_register #(
            .DATAW (DATAW),
            .DEPTH (i * `LATENCY_FMA)
        ) b_pipe (
            .clk      (clk),
            .reset    (reset),
            .enable   (enable),
            .data_in  (b_col[i][DATAW-1:0]),
            .data_out (b_col_p[i])
        );

        VX_pipe_register #(
            .DATAW (6),
            .DEPTH (i * `LATENCY_FMA)
        ) fmt_pipe (
            .clk      (clk),
            .reset    (reset),
            .enable   (enable),
            .data_in  ({fmt_d, fmt_s}),
            .data_out ({fmt_d_p[i], fmt_s_p[i]})
        );
    end

    wire [DATAW-1:0] fma_out [0:N];

    assign fma_out[0] = c_val[DATAW-1:0];

    for (genvar i = 0; i < N; i++) begin : g_fmas
    `ifdef TCU_DPI
        VX_tcu_fma_dpi #(
            .DATAW (DATAW)
        ) fma (
            .clk    (clk),
            .reset  (reset),
            .enable (enable),
            .fmt_s  (fmt_s_p[i]),
            .fmt_d  (fmt_d_p[i]),
            .a      (a_row_p[i]),
            .b      (b_col_p[i]),
            .c      (fma_out[i]),
            .y      (fma_out[i+1])
        );
    `elsif TCU_DSP
        VX_tcu_fma_dsp #(
            .DATAW (DATAW)
        ) fma (
            .clk    (clk),
            .reset  (reset),
            .enable (enable),
            .fmt_s  (fmt_s_p[i]),
            .fmt_d  (fmt_d_p[i]),
            .a      (a_row_p[i]),
            .b      (b_col_p[i]),
            .c      (fma_out[i]),
            .y      (fma_out[i+1])
        );
    `endif

    end

    assign d_val = `XLEN'(fma_out[N]);

endmodule
