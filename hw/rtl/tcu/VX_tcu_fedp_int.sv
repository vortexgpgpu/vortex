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

module VX_tcu_fedp_int #(
    parameter LATENCY = 2,
    parameter N = 2
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
    localparam LEVELS = $clog2(N);
    localparam RED_LATENCY = LEVELS;
    localparam ACC_LATENCY = RED_LATENCY + 1;
    `STATIC_ASSERT (LATENCY == (`LATENCY_IMUL+ACC_LATENCY), ("invalid parameter!"));

    `UNUSED_VAR ({a_row, b_col, c_val});
    `UNUSED_VAR (fmt_d);

    wire [31:0] nult_result [N];

    // multiplication stage
    for (genvar i = 0; i < N; i++) begin : g_prod
        reg signed [31:0] prod;
        always @(*) begin
            case (fmt_s)
                3'd0: begin // int32
                    prod = $signed(a_row[i][31:0]) * $signed(b_col[i][31:0]);
                end
                3'd1: begin // int16
                    prod = ($signed(a_row[i][15:0]) * $signed(b_col[i][15:0]))
                         + ($signed(a_row[i][31:16]) * $signed(b_col[i][31:16]));
                end
                3'd2: begin // int8
                    prod = ($signed(a_row[i][7:0]) * $signed(b_col[i][7:0])
                          + $signed(a_row[i][15:8]) * $signed(b_col[i][15:8]))
                         + ($signed(a_row[i][23:16]) * $signed(b_col[i][23:16])
                          + $signed(a_row[i][31:24]) * $signed(b_col[i][31:24]));
                end
                default: begin
                    prod = 'x;
                end
            endcase
        end
        VX_pipe_register #(
            .DATAW (32),
            .DEPTH (`LATENCY_IMUL)
        ) pipe_mult (
            .clk      (clk),
            .reset    (reset),
            .enable   (enable),
            .data_in  (prod),
            .data_out (nult_result[i])
        );
    end

    wire [31:0] red_in [LEVELS+1][N];

    for (genvar i = 0; i < N; i++) begin : g_red_inputs
        assign red_in[0][i] = nult_result[i];
    end

    // accumulate reduction tree
    for (genvar lvl = 0; lvl < LEVELS; lvl++) begin : g_red_tree
        localparam integer CURSZ = N >> lvl;
        localparam integer OUTSZ = CURSZ >> 1;
        for (genvar i = 0; i < OUTSZ; i++) begin : g_add
            VX_pipe_register #(
                .DATAW (32),
                .DEPTH (1)
            ) pipe_red (
                .clk      (clk),
                .reset    (reset),
                .enable   (enable),
                .data_in  (red_in[lvl][2*i+0] + red_in[lvl][2*i+1]),
                .data_out (red_in[lvl+1][i])
            );
        end
    end

    wire [31:0] delayed_c;

    VX_pipe_register #(
        .DATAW (32),
        .DEPTH (`LATENCY_IMUL + RED_LATENCY)
    ) pipe_c (
        .clk     (clk),
        .reset   (reset),
        .enable  (enable),
        .data_in (c_val[31:0]),
        .data_out(delayed_c)
    );

    wire [31:0] result;

    // final accumulation
    VX_pipe_register #(
        .DATAW (32),
        .DEPTH (1)
    ) pipe_acc (
        .clk     (clk),
        .reset   (reset),
        .enable  (enable),
        .data_in (red_in[LEVELS][0] + delayed_c),
        .data_out(result)
    );

    assign d_val = `XLEN'(result);

endmodule
