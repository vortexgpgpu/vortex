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

    wire [31:0] mult_result [N];

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
            4'b0101: begin // mxfp8 (basically the exact same as fp8 for fmul)
                prod = 64'hffffffff00000000;
                for (int j = 0; j < 4; j++) begin
                    dpi_f2f(enable, int'(0), int'(4), {56'hffffffffffffff, a_row[i][j * 8 +: 8]}, 3'b0, a_f, fflags);
                    dpi_f2f(enable, int'(0), int'(4), {56'hffffffffffffff, b_col[i][j * 8 +: 8]}, 3'b0, b_f, fflags);
                    dpi_fmul(enable, int'(0), a_f, b_f, 3'b0, temp, fflags);
                    dpi_fadd(enable, int'(0), temp, prod, 3'b0, prod, fflags);
                end
            end
            4'b0111: begin // nvfp4
                prod = 64'hffffffff00000000;
                for (int j = 0; j < 8; j++) begin
                    dpi_f2f(enable, int'(0), int'(6), {60'hfffffffffffffff, a_row[i][j * 4 +: 4]}, 3'b0, a_f, fflags);
                    dpi_f2f(enable, int'(0), int'(6), {60'hfffffffffffffff, b_col[i][j * 4 +: 4]}, 3'b0, b_f, fflags);
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
            4'b1101: begin // mxint8
                prod = 0;
                for (int j = 0; j < 4; j++) begin
                    prod += $signed({{24{a_row[i][8 * j + 7]}}, a_row[i][8 * j +: 8]}) * $signed({{24{b_col[i][8 * j + 7]}}, b_col[i][8 * j +: 8]});
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
            .data_out (mult_result[i])
        );
    end

    //TODO: temp hardcoded scale factors (input from module)
    wire [7:0] SCALE_FACTOR_E8M0_A = 8'd129;
    wire [7:0] SCALE_FACTOR_E8M0_B = 8'd131;
    wire [7:0] SCALE_FACTOR_E4M3_A = 8'h41;
    wire [7:0] SCALE_FACTOR_E4M3_B = 8'h33;

    reg [7:0] raw_sf_a, raw_sf_b, raw_sf;
    reg [63:0] a_sf, b_sf, temp_sf;
    reg [4:0] sfflags;
    `UNUSED_VAR(sfflags);
    always_comb begin
        case (fmt_s)
            4'b0101: begin // mxfp8
                raw_sf_a = SCALE_FACTOR_E8M0_A - 8'd127;
                raw_sf_b = SCALE_FACTOR_E8M0_B - 8'd127;
                raw_sf   = raw_sf_a + raw_sf_b;
            end
            4'b0111: begin // nvfp4
                dpi_f2f(enable, int'(0), int'(4), {56'hffffffffffffff, SCALE_FACTOR_E4M3_A}, 3'b0, a_sf, sfflags);
                dpi_f2f(enable, int'(0), int'(4), {56'hffffffffffffff, SCALE_FACTOR_E4M3_B}, 3'b0, b_sf, sfflags);
                dpi_fmul(enable, int'(0), a_sf, b_sf, 3'b0, temp_sf, sfflags);
            end
            4'b1101: begin // mxint8
                raw_sf_a = SCALE_FACTOR_E8M0_A - 8'd133;
                raw_sf_b = SCALE_FACTOR_E8M0_B - 8'd133;
                raw_sf   = raw_sf_a + raw_sf_b;
            end
            default: begin
                raw_sf = 8'd0;
            end
        endcase
    end

    wire [31:0] delayed_c;
    wire [7:0] delayed_raw_sf;
    wire [63:0] delayed_temp_sf;
    wire [3:0] delayed_fmt_s;

    VX_pipe_register #(
        .DATAW (4 + 32 + 8 + 64),
        .DEPTH (FMUL_LATENCY)
    ) pipe_c (
        .clk     (clk),
        .reset   (reset),
        .enable  (enable),
        .data_in ({fmt_s, c_val[31:0], raw_sf, temp_sf}),
        .data_out({delayed_fmt_s, delayed_c, delayed_raw_sf, delayed_temp_sf})
    );

    //floating point accumulator
    reg [63:0] acc_f;
    reg [4:0] fflags;
    `UNUSED_VAR(fflags);
    `UNUSED_VAR(acc_f[63:32]);
    always_comb begin
        acc_f = 64'hffffffff00000000;
        
        //adder chain
        for (int i = 0; i < N; ++i) begin
            dpi_fadd(enable, int'(0), {32'hffffffff, mult_result[i]}, acc_f, 3'b0, acc_f, fflags);
        end

        //multiply with scaling factor
        if (delayed_fmt_s == 4'b0111) begin // nvfp4
            dpi_fmul(enable, int'(0), acc_f, delayed_temp_sf, 3'b0, acc_f, fflags);
        end else begin
            acc_f[30:23] += delayed_raw_sf; // mxfp8
        end

        //addend addition
        dpi_fadd(enable, int'(0), {32'hffffffff, delayed_c}, acc_f, 3'b0, acc_f, fflags);
    end

    //integer accumulator
    reg [31:0] acc_i;
    always_comb begin
        acc_i = 0;
        for (int i = 0; i < N; ++i) begin
            acc_i = acc_i + mult_result[i];
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
        .data_in (delayed_fmt_s[3] ? acc_i : acc_f[31:0]),
        .data_out(result)
    );

`ifdef XLEN_64
    // should nan-box when writing to FP registers
    assign d_val = {32'hffffffff, result};
`else
    assign d_val = result;
`endif

endmodule
