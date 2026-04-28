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

module VX_tcu_fedp_fpnew import VX_tcu_pkg::*; #(
    parameter `STRING INSTANCE_ID = "",
    parameter LATENCY = 0,
    parameter N = 1
) (
    input  wire clk,
    input  wire reset,
    input  wire enable,

    input  wire [3:0] fmt_s,
    input  wire [3:0] fmt_d,

    input  wire [N-1:0][31:0] a_row,
    input  wire [N-1:0][31:0] b_col,
    input  wire [31:0] c_val,
    output wire [31:0] d_val
);
    `UNUSED_SPARAM (INSTANCE_ID)

    localparam TCK = 2 * N;
    localparam LEVELS = $clog2(TCK);
    localparam FMUL_LATENCY = 2;
    localparam FADD_LATENCY = 2;
    localparam FRED_LATENCY = LEVELS * FADD_LATENCY;
    localparam TOTAL_LATENCY = FMUL_LATENCY + 1 + FRED_LATENCY + FADD_LATENCY;
    `STATIC_ASSERT (LATENCY == 0 || LATENCY == TOTAL_LATENCY, ("invalid latency! expected=%0d, actual=%0d", TOTAL_LATENCY, LATENCY));

    localparam FMT_DELAY = FMUL_LATENCY;
    localparam C_DELAY = FMUL_LATENCY + 1 + FRED_LATENCY;

    `UNUSED_VAR ({fmt_s[3], fmt_d, c_val});

    wire [15:0] a_row16 [TCK];
    wire [15:0] b_col16 [TCK];

    for (genvar i = 0; i < N; ++i) begin : g_unpack16
        assign a_row16[2*i]   = a_row[i][15:0];
        assign a_row16[2*i+1] = a_row[i][31:16];
        assign b_col16[2*i]   = b_col[i][15:0];
        assign b_col16[2*i+1] = b_col[i][31:16];
    end

    wire [2:0] fmt_s_delayed;
    VX_pipe_register #(
        .DATAW (3),
        .DEPTH (FMT_DELAY)
    ) pipe_fmt_s (
        .clk     (clk),
        .reset   (reset),
        .enable  (enable),
        .data_in (fmt_s[2:0]),
        .data_out(fmt_s_delayed)
    );

    wire [31:0] mult_result [TCK];

    for (genvar i = 0; i < TCK; ++i) begin : g_multiply
        wire [31:0] mult_result_fp16;

        VX_tcu_fpnew_mulfp32 #(
            .SRC_FMT       (fpnew_pkg::FP16),
            .FMT_CFG       (5'b10100),
            .NUM_PIPE_REGS (FMUL_LATENCY)
        ) fp16_mul (
            .clk    (clk),
            .reset  (reset),
            .enable (enable),
            .a      (a_row16[i]),
            .b      (b_col16[i]),
            .y      (mult_result_fp16)
        );

    `ifdef TCU_BF16_ENABLE
        wire [31:0] mult_result_bf16;

        VX_tcu_fpnew_mulfp32 #(
            .SRC_FMT       (fpnew_pkg::FP16ALT),
            .FMT_CFG       (5'b10001),
            .NUM_PIPE_REGS (FMUL_LATENCY)
        ) bf16_mul (
            .clk    (clk),
            .reset  (reset),
            .enable (enable),
            .a      (a_row16[i]),
            .b      (b_col16[i]),
            .y      (mult_result_bf16)
        );
    `endif

        logic [31:0] mult_result_mux;
        always_comb begin
            case ({1'b0, fmt_s_delayed})
                TCU_FP16_ID: mult_result_mux = mult_result_fp16;
            `ifdef TCU_BF16_ENABLE
                TCU_BF16_ID: mult_result_mux = mult_result_bf16;
            `endif
                default:     mult_result_mux = 'x;
            endcase
        end

        VX_pipe_register #(
            .DATAW (32),
            .DEPTH (1)
        ) pipe_mulsel (
            .clk      (clk),
            .reset    (reset),
            .enable   (enable),
            .data_in  (mult_result_mux),
            .data_out (mult_result[i])
        );
    end

    wire [31:0] red_in [LEVELS+1][TCK];

    for (genvar i = 0; i < TCK; ++i) begin : g_red_in
        assign red_in[0][i] = mult_result[i];
    end

    for (genvar lvl = 0; lvl < LEVELS; ++lvl) begin : g_accumulate
        localparam CURSZ = TCK >> lvl;
        localparam OUTSZ = CURSZ >> 1;
        for (genvar i = 0; i < OUTSZ; ++i) begin : g_add
            VX_tcu_fpnew_addfp32 #(
                .NUM_PIPE_REGS (FADD_LATENCY)
            ) reduce_add (
                .clk    (clk),
                .reset  (reset),
                .enable (enable),
                .a      (red_in[lvl][2*i+0]),
                .b      (red_in[lvl][2*i+1]),
                .y      (red_in[lvl+1][i])
            );
        end
    end

    wire [31:0] c_delayed;
    VX_pipe_register #(
        .DATAW (32),
        .DEPTH (C_DELAY)
    ) pipe_c (
        .clk     (clk),
        .reset   (reset),
        .enable  (enable),
        .data_in (c_val),
        .data_out(c_delayed)
    );

    VX_tcu_fpnew_addfp32 #(
        .NUM_PIPE_REGS (FADD_LATENCY)
    ) final_add (
        .clk    (clk),
        .reset  (reset),
        .enable (enable),
        .a      (red_in[LEVELS][0]),
        .b      (c_delayed),
        .y      (d_val)
    );
endmodule
