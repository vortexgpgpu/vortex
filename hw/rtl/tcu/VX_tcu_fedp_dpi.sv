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
    localparam TCK = 2 * N;
    localparam LEVELS = $clog2(TCK);
    localparam FMUL_LATENCY = 2;
    localparam FADD_LATENCY = 1;
    localparam FRND_LATENCY = 1;
    localparam RED_LATENCY  = LEVELS * FADD_LATENCY;
    localparam ACC_LATENCY  = RED_LATENCY + FADD_LATENCY;
    `STATIC_ASSERT (LATENCY == (FMUL_LATENCY+ACC_LATENCY+FRND_LATENCY), ("invalid parameter!"));

    `UNUSED_VAR (reset);
    `UNUSED_VAR ({a_row, b_col, c_val});
    `UNUSED_VAR (fmt_d);

    wire [TCK-1:0][15:0] a_row16;
    wire [TCK-1:0][15:0] b_col16;

    for (genvar i = 0; i < N; i++) begin : g_unpack
        assign a_row16[2*i]   = a_row[i][15:0];
        assign a_row16[2*i+1] = a_row[i][31:16];
        assign b_col16[2*i]   = b_col[i][15:0];
        assign b_col16[2*i+1] = b_col[i][31:16];
    end

    wire [63:0] nult_result [TCK];

    // multiplication stage
    for (genvar i = 0; i < TCK; i++) begin : g_prod
        reg [63:0] a_h, a_b;
        reg [63:0] b_h, b_b;
        reg [63:0] xprod;
        reg [4:0] fflags;

        `UNUSED_VAR(fflags);

        wire [63:0] xa = {48'hffffffffffff, a_row16[i]};
        wire [63:0] xb = {48'hffffffffffff, b_col16[i]};

        always @(*) begin
            case (fmt_s)
                3'd1: begin // fp16
                    dpi_f2f(enable, int'(0), int'(2), xa, a_h);
                    dpi_f2f(enable, int'(0), int'(2), xb, b_h);
                    dpi_fmul(enable, int'(0), a_h, b_h, 3'b0, xprod, fflags);
                end
                3'd2: begin // bf16
                    dpi_f2f(enable, int'(0), int'(3), xa, a_b);
                    dpi_f2f(enable, int'(0), int'(3), xb, b_b);
                    dpi_fmul(enable, int'(0), a_b, b_b, 3'b0, xprod, fflags);
                end
                default: begin
                    xprod = 'x;
                end
            endcase
        end
        VX_pipe_register #(
            .DATAW (64),
            .DEPTH (FMUL_LATENCY)
        ) pipe_mult (
            .clk      (clk),
            .reset    (reset),
            .enable   (enable),
            .data_in  (xprod),
            .data_out (nult_result[i])
        );
    end

    wire [63:0] red_in [LEVELS+1][TCK];

    for (genvar i = 0; i < TCK; i++) begin : g_red_inputs
      assign red_in[0][i] = nult_result[i];
    end

    // accumulate reduction tree
    for (genvar lvl = 0; lvl < LEVELS; lvl++) begin : g_red_tree
        localparam integer CURSZ = TCK >> lvl;
        localparam integer OUTSZ = CURSZ >> 1;
        for (genvar i = 0; i < OUTSZ; i++) begin : g_add
            reg [4:0] fflags;
            `UNUSED_VAR(fflags);

            reg [63:0] xsum;
            always @(*) begin
                dpi_fadd(enable, int'(0), red_in[lvl][2*i+0], red_in[lvl][2*i+1], 3'b0, xsum, fflags);
            end

            VX_pipe_register #(
                .DATAW (64),
                .DEPTH (FADD_LATENCY)
            ) pipe_red (
                .clk      (clk),
                .reset    (reset),
                .enable   (enable),
                .data_in  (xsum),
                .data_out (red_in[lvl+1][i])
            );
        end
    end

    wire [31:0] delayed_c;

    VX_pipe_register #(
        .DATAW (32),
        .DEPTH (FMUL_LATENCY + RED_LATENCY)
    ) pipe_c (
        .clk     (clk),
        .reset   (reset),
        .enable  (enable),
        .data_in (c_val[31:0]),
        .data_out(delayed_c)
    );

    wire [63:0] xc = {32'hffffffff, delayed_c};
    reg [63:0] xacc;
    wire [63:0] result;
    `UNUSED_VAR(result[63:32]);

    reg [4:0] fflags;
    `UNUSED_VAR(fflags);

    // final accumulation + rounding
    always @(*) begin
        dpi_fadd(enable, int'(0), red_in[LEVELS][0], xc, 3'b0, xacc, fflags);
    end

    VX_pipe_register #(
        .DATAW (64),
        .DEPTH (FADD_LATENCY + FRND_LATENCY)
    ) pipe_acc (
        .clk     (clk),
        .reset   (reset),
        .enable  (enable),
        .data_in (xacc),
        .data_out(result)
    );

    assign d_val = `XLEN'(result);

endmodule
