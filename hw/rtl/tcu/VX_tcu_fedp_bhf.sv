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
`include "HardFloat_consts.vi"

module VX_tcu_fedp_bhf import VX_tcu_pkg::*; #(
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

`ifdef XLEN_64
    `UNUSED_VAR (c_val[63:32]);
`endif
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
            case (fmt_s_delayed)
                TCU_FP16_ID: mult_result_mux = mult_result_fp16;
                TCU_BF16_ID: mult_result_mux = mult_result_bf16;
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

    wire [16:0] c_fp16_rec, c_bf16_rec;
    wire [32:0] c_fp32_rec, c_fp16_to_fp32_rec, c_bf16_to_fp32_rec;
    logic [32:0] c_rec;
    wire [32:0] c_delayed;

    fNToRecFN #(
        .expWidth (8),
        .sigWidth (24)
    ) conv_c_fp32 (
        .in  (c_val[31:0]),
        .out (c_fp32_rec)
    );

    fNToRecFN #(
        .expWidth (5),
        .sigWidth (11)
    ) conv_c_fp16 (
        .in  (c_val[15:0]),
        .out (c_fp16_rec)
    );

    // Match the BHF fadd/fmul HardFloat tininess policy.
    wire control = `flControl_tininessAfterRounding; // IEEE 754-2008

    recFNToRecFN #(
        .inExpWidth  (5),
        .inSigWidth  (11),
        .outExpWidth (8),
        .outSigWidth (24)
    ) widen_c_fp16 (
        .control       (control),
        .in            (c_fp16_rec),
        .roundingMode  (frm),
        .out           (c_fp16_to_fp32_rec),
        `UNUSED_PIN (exceptionFlags)
    );

    fNToRecFN #(
        .expWidth (8),
        .sigWidth (8)
    ) conv_c_bf16 (
        .in  (c_val[15:0]),
        .out (c_bf16_rec)
    );

    recFNToRecFN #(
        .inExpWidth  (8),
        .inSigWidth  (8),
        .outExpWidth (8),
        .outSigWidth (24)
    ) widen_c_bf16 (
        .control       (control),
        .in            (c_bf16_rec),
        .roundingMode  (frm),
        .out           (c_bf16_to_fp32_rec),
        `UNUSED_PIN (exceptionFlags)
    );

    always_comb begin
        case (fmt_d)
            TCU_FP32_ID: c_rec = c_fp32_rec;
            TCU_FP16_ID: c_rec = c_fp16_to_fp32_rec;
            TCU_BF16_ID: c_rec = c_bf16_to_fp32_rec;
            default:     c_rec = 'x;
        endcase
    end

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

    wire [2:0] fmt_d_delayed;

    VX_pipe_register #(
        .DATAW (3),
        .DEPTH (TOTAL_LATENCY)
    ) pipe_fmt_d (
        .clk     (clk),
        .reset   (reset),
        .enable  (enable),
        .data_in (fmt_d),
        .data_out(fmt_d_delayed)
    );

    // Final accumulation

    wire [32:0] result_rec;

    VX_tcu_bhf_fadd #(
        .IN_EXPW (8),
        .IN_SIGW (23+1),
        .IN_REC  (1), // input in recoded format
        .OUT_REC (1), // output in recoded format
        .ADD_LATENCY (FADD_LATENCY),
        .RND_LATENCY (FRND_LATENCY)
    ) final_add (
        .clk    (clk),
        .reset  (reset),
        .enable (enable),
        .frm    (frm),
        .a      (red_in[LEVELS][0]),
        .b      (c_delayed),
        .y      (result_rec),
        `UNUSED_PIN(fflags)
    );

    wire [31:0] result_fp32;
    wire [16:0] result_fp16_rec, result_bf16_rec;
    wire [15:0] result_fp16, result_bf16;

    recFNToFN #(
        .expWidth (8),
        .sigWidth (24)
    ) to_fp32 (
        .in  (result_rec),
        .out (result_fp32)
    );

    recFNToRecFN #(
        .inExpWidth  (8),
        .inSigWidth  (24),
        .outExpWidth (5),
        .outSigWidth (11)
    ) narrow_result_fp16 (
        .control       (control),
        .in            (result_rec),
        .roundingMode  (frm),
        .out           (result_fp16_rec),
        `UNUSED_PIN (exceptionFlags)
    );

    recFNToFN #(
        .expWidth (5),
        .sigWidth (11)
    ) to_fp16 (
        .in  (result_fp16_rec),
        .out (result_fp16)
    );

    recFNToRecFN #(
        .inExpWidth  (8),
        .inSigWidth  (24),
        .outExpWidth (8),
        .outSigWidth (8)
    ) narrow_result_bf16 (
        .control       (control),
        .in            (result_rec),
        .roundingMode  (frm),
        .out           (result_bf16_rec),
        `UNUSED_PIN (exceptionFlags)
    );

    recFNToFN #(
        .expWidth (8),
        .sigWidth (8)
    ) to_bf16 (
        .in  (result_bf16_rec),
        .out (result_bf16)
    );

    logic [31:0] result;

    always_comb begin
        case (fmt_d_delayed)
            TCU_FP32_ID: result = result_fp32;
            TCU_FP16_ID: result = {16'b0, result_fp16};
            TCU_BF16_ID: result = {16'b0, result_bf16};
            default:     result = 'x;
        endcase
    end

`ifdef XLEN_64
    assign d_val = {32'hffffffff, result};
`else
    assign d_val = result;
`endif

endmodule
