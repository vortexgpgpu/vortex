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
    localparam ACC_LATENCY  = 1;
    localparam FRND_LATENCY = 1;
    `STATIC_ASSERT (LATENCY == 0 || LATENCY == (FMUL_LATENCY+ACC_LATENCY+FRND_LATENCY), ("invalid parameter!"));

    `UNUSED_VAR (reset);
    `UNUSED_VAR ({fmt_d, c_val});

    wire [TCK-1:0][15:0] a_row16;
    wire [TCK-1:0][15:0] b_col16;

    for (genvar i = 0; i < N; i++) begin : g_unpack
        assign a_row16[2*i]   = a_row[i][15:0];
        assign a_row16[2*i+1] = a_row[i][31:16];
        assign b_col16[2*i]   = b_col[i][15:0];
        assign b_col16[2*i+1] = b_col[i][31:16];
    end

    wire [31:0] nult_result [TCK];

    // multiplication stage
    for (genvar i = 0; i < TCK; i++) begin : g_prod
        reg [63:0] a_f, b_f;
        reg [63:0] p1_f, p2_f, p_f;
        reg [4:0] fflags;

        `UNUSED_VAR({fflags, p1_f, p2_f, p_f[63:32]});

        always_latch begin
            case (fmt_s)
                3'd1: begin // fp16
                    dpi_f2f(enable, int'(0), int'(2), {48'hffffffffffff, a_row16[i]}, 3'b0, a_f, fflags);
                    dpi_f2f(enable, int'(0), int'(2), {48'hffffffffffff, b_col16[i]}, 3'b0, b_f, fflags);
                    dpi_fmul(enable, int'(0), a_f, b_f, 3'b0, p_f, fflags);
                end
                3'd2: begin // bf16
                    dpi_f2f(enable, int'(0), int'(3), {48'hffffffffffff, a_row16[i]}, 3'b0, a_f, fflags);
                    dpi_f2f(enable, int'(0), int'(3), {48'hffffffffffff, b_col16[i]}, 3'b0, b_f, fflags);
                    dpi_fmul(enable, int'(0), a_f, b_f, 3'b0, p_f, fflags);
                end
                3'd3: begin // fp8
                    dpi_f2f(enable, int'(0), int'(4), {56'hffffffffffff, a_row16[i][0 +: 8]}, 3'b0, a_f, fflags);
                    dpi_f2f(enable, int'(0), int'(4), {56'hffffffffffff, b_col16[i][0 +: 8]}, 3'b0, b_f, fflags);
                    dpi_fmul(enable, int'(0), a_f, b_f, 3'b0, p1_f, fflags);
                    dpi_f2f(enable, int'(0), int'(4), {56'hffffffffffff, a_row16[i][8 +: 8]}, 3'b0, a_f, fflags);
                    dpi_f2f(enable, int'(0), int'(4), {56'hffffffffffff, b_col16[i][8 +: 8]}, 3'b0, b_f, fflags);
                    dpi_fmul(enable, int'(0), a_f, b_f, 3'b0, p2_f, fflags);
                    dpi_fadd(enable, int'(0), p1_f, p2_f, 3'b0, p_f, fflags);
                end
                3'd4: begin // bf8
                    dpi_f2f(enable, int'(0), int'(5), {56'hffffffffffff, a_row16[i][0 +: 8]}, 3'b0, a_f, fflags);
                    dpi_f2f(enable, int'(0), int'(5), {56'hffffffffffff, b_col16[i][0 +: 8]}, 3'b0, b_f, fflags);
                    dpi_fmul(enable, int'(0), a_f, b_f, 3'b0, p1_f, fflags);
                    dpi_f2f(enable, int'(0), int'(5), {56'hffffffffffff, a_row16[i][8 +: 8]}, 3'b0, a_f, fflags);
                    dpi_f2f(enable, int'(0), int'(5), {56'hffffffffffff, b_col16[i][8 +: 8]}, 3'b0, b_f, fflags);
                    dpi_fmul(enable, int'(0), a_f, b_f, 3'b0, p2_f, fflags);
                    dpi_fadd(enable, int'(0), p1_f, p2_f, 3'b0, p_f, fflags);
                end
                default: begin
                    p_f = 'x;
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
            .data_in  (p_f[31:0]),
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

`IGNORE_UNOPTFLAT_BEGIN
    wire [63:0] red_in [LEVELS+1][TCK];
`IGNORE_UNOPTFLAT_END
    for (genvar i = 0; i < TCK; i++) begin : g_red_inputs
      assign red_in[0][i] = {32'hffffffff, nult_result[i]};
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
            assign red_in[lvl+1][i] = xsum;
        end
    end

    // final accumulation + rounding
    reg [63:0] xacc;
    reg [4:0] fflags;
    `UNUSED_VAR(fflags);
    `UNUSED_VAR(xacc[63:32]);
    always @(*) begin
        dpi_fadd(enable, int'(0), red_in[LEVELS][0], {32'hffffffff, delayed_c}, 3'b0, xacc, fflags);
    end

    wire [31:0] result;

    VX_pipe_register #(
        .DATAW (32),
        .DEPTH (ACC_LATENCY + FRND_LATENCY)
    ) pipe_acc (
        .clk     (clk),
        .reset   (reset),
        .enable  (enable),
        .data_in (xacc[31:0]),
        .data_out(result)
    );

    assign d_val = `XLEN'(result);

endmodule
