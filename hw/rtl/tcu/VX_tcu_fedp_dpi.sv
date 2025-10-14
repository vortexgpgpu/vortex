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

module VX_tcu_fedp_dpi #(
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
    localparam FMUL_LATENCY = 2;
    localparam FACC_LATENCY = 2;
    localparam TOTAL_LATENCY= FMUL_LATENCY + FACC_LATENCY;
    `STATIC_ASSERT (LATENCY == 0 || LATENCY == TOTAL_LATENCY, ("invalid latency! expected=%0d, actual=%0d", TOTAL_LATENCY, LATENCY));

    `UNUSED_VAR ({fmt_d, c_val});

    wire [31:0] nult_result [N];

    // multiplication stage
    for (genvar i = 0; i < N; i++) begin : g_prod
        reg [63:0] a_f, b_f;
        reg [63:0] xprod;
        reg [4:0] fflags;

        `UNUSED_VAR({fflags, xprod[63:32]});

        always @(*) begin
            case (fmt_s)
            3'd1: begin // fp16
                xprod = 64'hffffffff00000000;
                for (int j = 0; j < 2; j++) begin
                    dpi_f2f(enable, int'(0), int'(2), {48'hffffffffffff, a_row[i][j * 16 +: 16]}, a_f);
                    dpi_f2f(enable, int'(0), int'(2), {48'hffffffffffff, b_col[i][j * 16 +: 16]}, b_f);
                    dpi_fmadd(enable, int'(0), a_f, b_f, xprod, 3'b0, xprod, fflags);
                end
            end
            3'd2: begin // bf16
                xprod = 64'hffffffff00000000;
                for (int j = 0; j < 2; j++) begin
                    dpi_f2f(enable, int'(0), int'(3), {48'hffffffffffff, a_row[i][j * 16 +: 16]}, a_f);
                    dpi_f2f(enable, int'(0), int'(3), {48'hffffffffffff, b_col[i][j * 16 +: 16]}, b_f);
                    dpi_fmadd(enable, int'(0), a_f, b_f, xprod, 3'b0, xprod, fflags);
                end
            end
            default: begin
                xprod = 'x;
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
            .data_in  (xprod[31:0]),
            .data_out (nult_result[i])
        );
    end

    wire [31:0] delayed_c;

    VX_pipe_register #(
        .DATAW (32),
        .DEPTH (FMUL_LATENCY)
    ) pipe_c (
        .clk     (clk),
        .reset   (reset),
        .enable  (enable),
        .data_in (c_val[31:0]),
        .data_out(delayed_c)
    );

    reg [63:0] xacc;
    reg [4:0] fflags;
    `UNUSED_VAR(fflags);

    always_comb begin
        xacc = 64'hffffffff00000000;
        for (int i = 0; i < N; ++i) begin
            dpi_fadd(enable, int'(0), {32'hffffffff, nult_result[i]}, xacc, 3'b0, xacc, fflags);
        end
        dpi_fadd(enable, int'(0), {32'hffffffff, delayed_c}, xacc, 3'b0, xacc, fflags);
    end

    wire [31:0] result;
    VX_pipe_register #(
        .DATAW (32),
        .DEPTH (FACC_LATENCY)
    ) pipe_acc (
        .clk     (clk),
        .reset   (reset),
        .enable  (enable),
        .data_in (xacc[31:0]),
        .data_out(result)
    );

    assign d_val = `XLEN'(result);

endmodule
