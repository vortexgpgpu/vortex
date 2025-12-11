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
`include "dpi_float.vh"

module VX_tcu_fedp_dpi #(
    parameter LATENCY = 0,
    parameter N = 1
) (
    input  wire clk,
    input  wire reset,
    input  wire enable,

    input  wire[3:0] fmt_s,
    input  wire[3:0] fmt_d,

    input  wire [N-1:0][`XLEN-1:0] a_row,
    input  wire [N-1:0][`XLEN-1:0] b_col,
    input  wire [`XLEN-1:0] c_val,
    output wire [`XLEN-1:0] d_val
);
    localparam FMUL_LATENCY = 2;
    localparam FACC_LATENCY = 2;
    localparam TOTAL_LATENCY= FMUL_LATENCY + FACC_LATENCY;
    `STATIC_ASSERT (LATENCY == 0 || LATENCY == TOTAL_LATENCY, ("invalid latency! expected=%0d, actual=%0d", TOTAL_LATENCY, LATENCY));

    `UNUSED_VAR ({fmt_d, c_val});

    wire [31:0] nult_result [N];

    wire is_int = fmt_s[3];
    wire is_int_r;

    // multiplication stage
    for (genvar i = 0; i < N; i++) begin : g_prod
        reg [63:0] a_f, b_f;
        reg [63:0] temp, prod;
        reg [4:0] fflags;

        `UNUSED_VAR({fflags, prod[63:32]});

        always_latch begin
            case (fmt_s)
            4'b0001: begin // fp16
                prod = 64'hffffffff00000000;
                for (int j = 0; j < 2; j++) begin
                    dpi_f2f(enable, int'(0), int'(2), {48'hffffffffffff, a_row[i][j * 16 +: 16]}, 3'b0, a_f, fflags);
                    dpi_f2f(enable, int'(0), int'(2), {48'hffffffffffff, b_col[i][j * 16 +: 16]}, 3'b0, b_f, fflags);
                    dpi_fmul(enable, int'(0), a_f, b_f, 3'b0, temp, fflags);
                    dpi_fadd(enable, int'(0), temp, prod, 3'b0, prod, fflags);
                end
            end
            4'b0010: begin // bf16
                prod = 64'hffffffff00000000;
                for (int j = 0; j < 2; j++) begin
                    dpi_f2f(enable, int'(0), int'(3), {48'hffffffffffff, a_row[i][j * 16 +: 16]}, 3'b0, a_f, fflags);
                    dpi_f2f(enable, int'(0), int'(3), {48'hffffffffffff, b_col[i][j * 16 +: 16]}, 3'b0, b_f, fflags);
                    dpi_fmul(enable, int'(0), a_f, b_f, 3'b0, temp, fflags);
                    dpi_fadd(enable, int'(0), temp, prod, 3'b0, prod, fflags);
                end
            end
            4'b0011: begin // fp8
                prod = 64'hffffffff00000000;
                for (int j = 0; j < 4; j++) begin
                    dpi_f2f(enable, int'(0), int'(4), {56'hffffffffffffff, a_row[i][j * 8 +: 8]}, 3'b0, a_f, fflags);
                    dpi_f2f(enable, int'(0), int'(4), {56'hffffffffffffff, b_col[i][j * 8 +: 8]}, 3'b0, b_f, fflags);
                    dpi_fmul(enable, int'(0), a_f, b_f, 3'b0, temp, fflags);
                    dpi_fadd(enable, int'(0), temp, prod, 3'b0, prod, fflags);
                end
            end
            4'b0100: begin // bf8
                prod = 64'hffffffff00000000;
                for (int j = 0; j < 4; j++) begin
                    dpi_f2f(enable, int'(0), int'(5), {56'hffffffffffffff, a_row[i][j * 8 +: 8]}, 3'b0, a_f, fflags);
                    dpi_f2f(enable, int'(0), int'(5), {56'hffffffffffffff, b_col[i][j * 8 +: 8]}, 3'b0, b_f, fflags);
                    dpi_fmul(enable, int'(0), a_f, b_f, 3'b0, temp, fflags);
                    dpi_fadd(enable, int'(0), temp, prod, 3'b0, prod, fflags);
                end
            end
            4'b1001: begin // int8
                prod = 0;
                for (int j = 0; j < 4; j++) begin
                    prod += $signed({{24{a_row[i][8 * j + 7]}}, a_row[i][8 * j +: 8]}) * $signed({{24{b_col[i][8 * j + 7]}}, b_col[i][8 * j +: 8]});
                end
            end
            4'b1010: begin // uint8
                prod = 0;
                for (int j = 0; j < 4; j++) begin
                    prod += a_row[i][8 * j +: 8] * b_col[i][8 * j +: 8];
                end
            end
            4'b1011: begin // int4
                prod = 0;
                for (int j = 0; j < 8; j++) begin
                    prod += $signed({{28{a_row[i][4 * j + 3]}}, a_row[i][4 * j +: 4]}) * $signed({{28{b_col[i][4 * j + 3]}}, b_col[i][4 * j +: 4]});
                end
            end
            4'b1100: begin // uint4
                prod = 0;
                for (int j = 0; j < 8; j++) begin
                    prod += a_row[i][4 * j +: 4] * b_col[i][4 * j +: 4];
                end
            end
            default: begin
                prod = 'x;
            end
            endcase
        end

        VX_pipe_register #(
            .DATAW (32),
            .DEPTH (FMUL_LATENCY)
        ) pipe_mult (
            .clk      (clk),
            .reset    (reset),
            .enable   (enable),
            .data_in  (prod[31:0]),
            .data_out (nult_result[i])
        );
    end

    wire [31:0] delayed_c;

    VX_pipe_register #(
        .DATAW (1 + 32),
        .DEPTH (FMUL_LATENCY)
    ) pipe_c (
        .clk     (clk),
        .reset   (reset),
        .enable  (enable),
        .data_in ({is_int,   c_val[31:0]}),
        .data_out({is_int_r, delayed_c})
    );

    reg [63:0] acc_f;
    reg [31:0] acc_i;
    reg [4:0] fflags;

    `UNUSED_VAR(fflags);
    `UNUSED_VAR(acc_f[63:32]);

    always_comb begin
        acc_f = 64'hffffffff00000000;
        for (int i = 0; i < N; ++i) begin
            dpi_fadd(enable, int'(0), {32'hffffffff, nult_result[i]}, acc_f, 3'b0, acc_f, fflags);
        end
        dpi_fadd(enable, int'(0), {32'hffffffff, delayed_c}, acc_f, 3'b0, acc_f, fflags);
    end

    always_comb begin
        acc_i = 0;
        for (int i = 0; i < N; ++i) begin
            acc_i = acc_i + nult_result[i];
        end
        acc_i += delayed_c;
    end

    wire [31:0] result;
    VX_pipe_register #(
        .DATAW (32),
        .DEPTH (FACC_LATENCY)
    ) pipe_acc (
        .clk     (clk),
        .reset   (reset),
        .enable  (enable),
        .data_in (is_int_r ? acc_i : acc_f[31:0]),
        .data_out(result)
    );

`ifdef XLEN_64
    // should nan-box when writing to FP registers
    assign d_val = {32'hffffffff, result};
`else
    assign d_val = result;
`endif

endmodule
