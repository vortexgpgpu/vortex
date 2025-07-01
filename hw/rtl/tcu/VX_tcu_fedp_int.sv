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
    parameter LATENCY = 1,
    parameter N = 1
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
    localparam PRODW = 18;
    localparam PSELW = PRODW + 1; // add unsinged guard bit
    localparam REDW  = `MAX(PRODW + LEVELS, PSELW);
    localparam MUL_LATENCY = 2;
    localparam ADD_LATENCY = 1;
    localparam RED_LATENCY = LEVELS * ADD_LATENCY;
    localparam ACC_LATENCY = RED_LATENCY + ADD_LATENCY;
    `STATIC_ASSERT (LATENCY == (MUL_LATENCY+ACC_LATENCY), ("invalid parameter!"));

    `UNUSED_VAR ({a_row, b_col, c_val});
    `UNUSED_VAR (fmt_d);

    wire [2:0] delayed_fmt_s;
    VX_pipe_register #(
        .DATAW (3),
        .DEPTH (MUL_LATENCY-1) // remove select stage
    ) pipe_fmt_s (
        .clk     (clk),
        .reset   (reset),
        .enable  (enable),
        .data_in (fmt_s),
        .data_out(delayed_fmt_s)
    );

    wire [PSELW-1:0] mult_result [N];

    // multiplication stage
    for (genvar i = 0; i < N; i++) begin : g_prod
        reg [16:0] prod_i8_1a, prod_i8_1b;
        reg [16:0] prod_u8_1a, prod_u8_1b;
        reg [9:0] prod_i4_1a, prod_i4_1b;
        reg [9:0] prod_u4_1a, prod_u4_1b;

        always @(posedge clk) begin
            if (enable) begin
                prod_i8_1a <= ($signed(a_row[i][7:0]) * $signed(b_col[i][7:0]))
                            + ($signed(a_row[i][15:8]) * $signed(b_col[i][15:8]));
                prod_i8_1b <= ($signed(a_row[i][23:16]) * $signed(b_col[i][23:16]))
                            + ($signed(a_row[i][31:24]) * $signed(b_col[i][31:24]));
            end
        end

        always @(posedge clk) begin
            if (enable) begin
                prod_u8_1a <= (a_row[i][7:0] * b_col[i][7:0])
                            + (a_row[i][15:8] * b_col[i][15:8]);
                prod_u8_1b <= (a_row[i][23:16] * b_col[i][23:16])
                            + (a_row[i][31:24] * b_col[i][31:24]);
            end
        end

        always @(posedge clk) begin
            if (enable) begin
                prod_i4_1a <= (($signed(a_row[i][3:0]) * $signed(b_col[i][3:0])) + ($signed(a_row[i][7:4]) * $signed(b_col[i][7:4])))
                            + (($signed(a_row[i][11:8]) * $signed(b_col[i][11:8])) + ($signed(a_row[i][15:12]) * $signed(b_col[i][15:12])));
                prod_i4_1b <= (($signed(a_row[i][19:16]) * $signed(b_col[i][19:16])) + ($signed(a_row[i][23:20]) * $signed(b_col[i][23:20])))
                            + (($signed(a_row[i][27:24]) * $signed(b_col[i][27:24])) + ($signed(a_row[i][31:28]) * $signed(b_col[i][31:28])));
            end
        end

        always @(posedge clk) begin
            if (enable) begin
                prod_u4_1a <= ((a_row[i][3:0] * b_col[i][3:0]) + (a_row[i][7:4] * b_col[i][7:4]))
                            + ((a_row[i][11:8] * b_col[i][11:8]) + (a_row[i][15:12] * b_col[i][15:12]));
                prod_u4_1b <= ((a_row[i][19:16] * b_col[i][19:16]) + (a_row[i][23:20] * b_col[i][23:20]))
                            + ((a_row[i][27:24] * b_col[i][27:24]) + (a_row[i][31:28] * b_col[i][31:28]));
            end
        end

        wire [17:0] sum_i8 = $signed(prod_i8_1a) + $signed(prod_i8_1b);
        wire [17:0] sum_u8 = prod_u8_1a + prod_u8_1b;
        wire [10:0] sum_i4 = $signed(prod_i4_1a) + $signed(prod_i4_1b);
        wire [10:0] sum_u4 = prod_u4_1a + prod_u4_1b;

        reg [PSELW-1:0] mult_sel;
        always @(*) begin
            case (delayed_fmt_s)
            3'd1: mult_sel = PSELW'($signed(sum_i8));
            3'd2: mult_sel = PSELW'(sum_u8);
            3'd3: mult_sel = PSELW'($signed(sum_i4));
            3'd4: mult_sel = PSELW'(sum_u4);
            default: mult_sel = 'x;
            endcase
        end

        VX_pipe_register #(
            .DATAW (PSELW),
            .DEPTH (1)
        ) pipe_sel (
            .clk      (clk),
            .reset    (reset),
            .enable   (enable),
            .data_in  (mult_sel),
            .data_out (mult_result[i])
        );
    end

    wire [REDW-1:0] red_in [LEVELS+1][N];
    for (genvar i = 0; i < N; i++) begin : g_red_inputs
        assign red_in[0][i] = REDW'($signed(mult_result[i]));
    end

    // accumulate reduction tree
    for (genvar lvl = 0; lvl < LEVELS; lvl++) begin : g_red_tree
        localparam integer CURSZ = N >> lvl;
        localparam integer OUTSZ = CURSZ >> 1;
        for (genvar i = 0; i < OUTSZ; i++) begin : g_add
            wire [REDW-1:0] sum = red_in[lvl][2*i+0] + red_in[lvl][2*i+1];
            VX_pipe_register #(
                .DATAW (REDW),
                .DEPTH (1)
            ) pipe_red (
                .clk      (clk),
                .reset    (reset),
                .enable   (enable),
                .data_in  (sum),
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
    wire [31:0] acc = 32'($signed(red_in[LEVELS][0])) + delayed_c;
    VX_pipe_register #(
        .DATAW (32),
        .DEPTH (1)
    ) pipe_acc (
        .clk     (clk),
        .reset   (reset),
        .enable  (enable),
        .data_in (acc),
        .data_out(result)
    );

    `ifdef XLEN_64
        // should nan-box when writing to FP registers
        assign d_val = {32'hffffffff, result};
    `else
        assign d_val = result;
    `endif

endmodule
