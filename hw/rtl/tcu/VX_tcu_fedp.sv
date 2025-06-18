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

module VX_tcu_fedp #(
    parameter DATAW = 32,
    parameter N     = 2
) (
    input  wire clk,
    input  wire reset,
    input  wire enable,

    input  wire[3:0] fmt_s,
    input  wire[3:0] fmt_d,

    input  wire [N-1:0][DATAW-1:0] a_row,
    input  wire [N-1:0][DATAW-1:0] b_col,
    input  wire [DATAW-1:0] c_val,
    output wire [DATAW-1:0] d_val
);
    `UNUSED_VAR (fmt_s);
    `UNUSED_VAR (fmt_d);

    wire [DATAW-1:0] a_delayed [0:N-1];
    wire [DATAW-1:0] b_delayed [0:N-1];

    for (genvar i = 0; i < N; i++) begin: g_pipe
        VX_pipe_register #(
            .DATAW (DATAW),
            .DEPTH (i)
        ) a_pipe (
            .clk      (clk),
            .reset    (reset),
            .enable   (enable),
            .data_in  (a_row[i]),
            .data_out (a_delayed[i])
        );
        VX_pipe_register #(
            .DATAW (DATAW),
            .DEPTH (i)
        ) b_pipe (
            .clk      (clk),
            .reset    (reset),
            .enable   (enable),
            .data_in  (b_col[i]),
            .data_out (b_delayed[i])
        );
    end

    wire [DATAW-1:0] fma_out [0:N-1];

    for (genvar i = 0; i < N; i++) begin : g_fmas
        wire [DATAW-1:0] c_in = (i==0) ? c_val : fma_out[i-1];
        VX_tcu_fma_int #(
            .DATAW (DATAW)
        ) fma (
            .clk    (clk),
            .reset  (reset),
            .enable (enable),
            .a      (a_delayed[i]),
            .b      (b_delayed[i]),
            .c      (c_in),
            .y      (fma_out[i])
        );
    end

    assign d_val = fma_out[N-1];

endmodule
