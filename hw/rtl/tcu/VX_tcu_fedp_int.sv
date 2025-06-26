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
    localparam MUL_LATENCY = 3;
    localparam ADD_LATENCY = 1;
    localparam RED_LATENCY = LEVELS * ADD_LATENCY;
    localparam ACC_LATENCY = RED_LATENCY + ADD_LATENCY;
    `STATIC_ASSERT (LATENCY == (MUL_LATENCY+ACC_LATENCY), ("invalid parameter!"));

    `UNUSED_VAR ({a_row, b_col, c_val});
    `UNUSED_VAR (fmt_d);

    wire [31:0] prod_i32 [N];
    wire [31:0] prod_i16 [N];
    wire [31:0] prod_i8 [N];

    // multiplication stage
    for (genvar i = 0; i < N; i++) begin : g_prod_i32
        reg [31:0] prod1, prod2, prod3;
        always @(posedge clk) begin
            if (enable) begin
                prod1 <= $signed(a_row[i][31:0]) * $signed(b_col[i][31:0]);
                prod2 <= prod1;
                prod3 <= prod2;
            end
        end
        assign prod_i32[i] = prod3;
    end

    for (genvar i = 0; i < N; i++) begin : g_prod_i16
        reg [31:0] prod1_0, prod1_1, prod2_0, prod2_1;
        reg [31:0] sum3;
        always @(posedge clk) begin
            if (enable) begin
                prod1_0 <= $signed(a_row[i][15:0]) * $signed(b_col[i][15:0]);
                prod1_1 <= $signed(a_row[i][31:16]) * $signed(b_col[i][31:16]);
                prod2_0 <= prod1_0;
                prod2_1 <= prod1_1;
                sum3    <= prod2_0 + prod2_1;
            end
        end
        assign prod_i16[i] = sum3;
    end

    for (genvar i = 0; i < N; i++) begin : g_prod_i8
        reg [16:0] prod1_0, prod1_1, prod1_2, prod1_3;
        reg [17:0] sum2_0, sum2_1;
        reg [18:0] sum3;
        always @(posedge clk) begin
            if (enable) begin
                prod1_0 <= $signed(a_row[i][7:0]) * $signed(b_col[i][7:0]);
                prod1_1 <= $signed(a_row[i][15:8]) * $signed(b_col[i][15:8]);
                prod1_2 <= $signed(a_row[i][23:16]) * $signed(b_col[i][23:16]);
                prod1_3 <= $signed(a_row[i][31:24]) * $signed(b_col[i][31:24]);
                sum2_0  <= prod1_0 + prod1_1;
                sum2_1  <= prod1_2 + prod1_3;
                sum3    <= sum2_0 + sum2_1;
            end
        end
        assign prod_i8[i] = 32'(sum3);
    end

    wire [2:0] delayed_fmt_s;
    VX_pipe_register #(
        .DATAW (3),
        .DEPTH (MUL_LATENCY)
    ) pipe_fmt_s (
        .clk     (clk),
        .reset   (reset),
        .enable  (enable),
        .data_in (fmt_s),
        .data_out(delayed_fmt_s)
    );

    wire [31:0] mult_result [N];
    for (genvar i = 0; i < N; i++) begin : g_mul_sel
        reg [31:0] mult_sel;
        always @(*) begin
            case (delayed_fmt_s)
            3'd0: mult_sel = prod_i32[i];
            3'd1: mult_sel = prod_i16[i];
            3'd2: mult_sel = prod_i8[i];
            default: mult_sel = 'x;
            endcase
        end
        assign mult_result[i] = mult_sel;
    end

    wire [31:0] red_in [LEVELS+1][N];

    for (genvar i = 0; i < N; i++) begin : g_red_inputs
        assign red_in[0][i] = mult_result[i];
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
        .DEPTH (MUL_LATENCY + RED_LATENCY)
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
