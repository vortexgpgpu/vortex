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

module VX_tcu_fedp_bhf #(
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
    localparam FADD_LATENCY = 2;
    localparam FRND_LATENCY = 1;
    localparam FRED_LATENCY = LEVELS * (FADD_LATENCY + FRND_LATENCY);
    localparam TOTAL_LATENCY= (FMUL_LATENCY + FRND_LATENCY) + 1 + FRED_LATENCY + (FADD_LATENCY + FRND_LATENCY);
    `STATIC_ASSERT (LATENCY == 0 || LATENCY == TOTAL_LATENCY, ("invalid latency! expected=%0d, actual=%0d", TOTAL_LATENCY, LATENCY));

    localparam FMT_DELAY = FMUL_LATENCY + FRND_LATENCY;
    localparam C_DELAY = (FMUL_LATENCY + FRND_LATENCY) + 1 + FRED_LATENCY;

    `UNUSED_VAR ({fmt_d, c_val});

    wire [2:0] frm = '0; // RNE rounding mode

    wire [TCK-1:0][15:0] a_row16;
    wire [TCK-1:0][15:0] b_col16;

    for (genvar i = 0; i < N; i++) begin : g_unpack
        assign a_row16[2*i]   = a_row[i][15:0];
        assign a_row16[2*i+1] = a_row[i][31:16];
        assign b_col16[2*i]   = b_col[i][15:0];
        assign b_col16[2*i+1] = b_col[i][31:16];
    end

    // Transprecision Multiply

    wire [2:0] fmt_s_delayed;

    VX_pipe_register #(
        .DATAW (3),
        .DEPTH (FMT_DELAY)
    ) pipe_fmt_s (
        .clk     (clk),
        .reset   (reset),
        .enable  (enable),
        .data_in (fmt_s),
        .data_out(fmt_s_delayed)
    );

    wire [32:0] mult_result [TCK];

    for (genvar i = 0; i < TCK; i++) begin : g_prod
        wire [32:0] mult_result_fp16;
        wire [32:0] mult_result_bf16;

        // FP16 multiplication
        VX_tcu_bhf_fmul #(
            .IN_EXPW (5),
            .IN_SIGW (10+1),
            .OUT_EXPW(8),
            .OUT_SIGW(24),
            .IN_REC  (0), // input in IEEE format
            .OUT_REC (1), // output in recoded format
            .MUL_LATENCY (FMUL_LATENCY),
            .RND_LATENCY (FRND_LATENCY)
        ) fp16_mul (
            .clk    (clk),
            .reset  (reset),
            .enable (enable),
            .frm    (frm),
            .a      (a_row16[i]),
            .b      (b_col16[i]),
            .y      (mult_result_fp16),
            `UNUSED_PIN(fflags)
        );

        // BF16 multiplication
        VX_tcu_bhf_fmul #(
            .IN_EXPW (8),
            .IN_SIGW (7+1),
            .OUT_EXPW(8),
            .OUT_SIGW(24),
            .IN_REC  (0), // input in IEEE format
            .OUT_REC (1), // output in recoded format
            .MUL_LATENCY (FMUL_LATENCY),
            .RND_LATENCY (FRND_LATENCY)
        ) bf16_mul (
            .clk    (clk),
            .reset  (reset),
            .enable (enable),
            .frm    (frm),
            .a      (a_row16[i]),
            .b      (b_col16[i]),
            .y      (mult_result_bf16),
            `UNUSED_PIN(fflags)
        );

        logic [32:0] mult_result_mux;
        always_comb begin
            case(fmt_s_delayed)
                3'd1: mult_result_mux = mult_result_fp16;
                3'd2: mult_result_mux = mult_result_bf16;
                default: mult_result_mux = 'x;
            endcase
        end

        VX_pipe_register #(
            .DATAW (33),
            .DEPTH (1) // select latency
        ) pipe_mult (
            .clk      (clk),
            .reset    (reset),
            .enable   (enable),
            .data_in  (mult_result_mux),
            .data_out (mult_result[i])
        );
    end

    wire [32:0] red_in [0:LEVELS] [TCK];

    for (genvar i = 0; i < TCK; i++) begin : g_red_inputs
        assign red_in[0][i] = mult_result[i];
    end

    // Accumulate reduction tree
    for (genvar lvl = 0; lvl < LEVELS; lvl++) begin : g_red_tree
        localparam CURSZ = TCK >> lvl;
        localparam OUTSZ = CURSZ >> 1;

        for (genvar i = 0; i < OUTSZ; i++) begin : g_add
            VX_tcu_bhf_fadd #(
                .IN_EXPW (8),
                .IN_SIGW (23+1),
                .IN_REC  (1), // input in recoded format
                .OUT_REC (1), // output in recoded format
                .ADD_LATENCY (FADD_LATENCY),
                .RND_LATENCY (FRND_LATENCY)
            ) reduce_add (
                .clk    (clk),
                .reset  (reset),
                .enable (enable),
                .frm    (frm),
                .a      (red_in[lvl][2*i+0]),
                .b      (red_in[lvl][2*i+1]),
                .y      (red_in[lvl+1][i]),
                `UNUSED_PIN(fflags)
            );
        end
    end

    // Accumulation input C recoding and delay handling

    wire [32:0] c_rec, c_delayed;
    wire [31:0] result;

    fNToRecFN #(
        .expWidth (8),
        .sigWidth (24)
    ) conv_c (
        .in  (c_val[31:0]),
        .out (c_rec)
    );

    VX_pipe_register #(
        .DATAW (33),
        .DEPTH (C_DELAY)
    ) pipe_c (
        .clk     (clk),
        .reset   (reset),
        .enable  (enable),
        .data_in (c_rec),
        .data_out(c_delayed)
    );

    // Final accumulation
    VX_tcu_bhf_fadd #(
        .IN_EXPW (8),
        .IN_SIGW (23+1),
        .IN_REC  (1), // input in recoded format
        .OUT_REC (0), // output in IEEE format
        .ADD_LATENCY (FADD_LATENCY),
        .RND_LATENCY (FRND_LATENCY)
    ) final_add (
        .clk    (clk),
        .reset  (reset),
        .enable (enable),
        .frm    (frm),
        .a      (red_in[LEVELS][0]),
        .b      (c_delayed),
        .y      (result),
        `UNUSED_PIN(fflags)
    );

    assign d_val = `XLEN'(result);

endmodule
