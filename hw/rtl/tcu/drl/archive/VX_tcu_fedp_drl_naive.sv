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

module VX_tcu_fedp_drl_naive #(
    parameter LATENCY = 1,
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

    localparam TCK = 2 * N;
    localparam LEVELS = $clog2(TCK);
    localparam FMUL_LATENCY = 2;
    localparam FADD_LATENCY = 1;
    localparam FRND_LATENCY = 1;
    localparam RED_LATENCY  = LEVELS * FADD_LATENCY;
    localparam ACC_LATENCY  = RED_LATENCY + FADD_LATENCY;
    `STATIC_ASSERT (LATENCY == (FMUL_LATENCY+ACC_LATENCY+FRND_LATENCY), ("invalid parameter!"));

    `UNUSED_VAR (reset);
    `UNUSED_VAR (fmt_d);

    wire [TCK-1:0][15:0] a_row16;
    wire [TCK-1:0][15:0] b_col16;

    for (genvar i = 0; i < N; i++) begin : g_unpack
        assign a_row16[2*i]   = a_row[i][15:0];
        assign a_row16[2*i+1] = a_row[i][31:16];
        assign b_col16[2*i]   = b_col[i][15:0];
        assign b_col16[2*i+1] = b_col[i][31:16];
    end

    wire [31:0] mult_result_fp16 [TCK];
    wire [31:0] mult_result_bf16 [TCK];
    logic [31:0] mult_result_mux [TCK];
    wire [31:0] mult_result [TCK];

    //Transprecision Multiplication stage
    for (genvar i = 0; i < TCK; i++) begin : g_prod
        // FP16 multiplication
        VX_tcu_drl_fp16mul fp16mul (
            .enable  (enable),
            .a       (a_row16[i]),
            .b       (b_col16[i]),
            .y       (mult_result_fp16[i])
        );

        // BF16 multiplication
        VX_tcu_drl_bf16mul bf16mul (
            .enable  (enable),
            .a       (a_row16[i]),
            .b       (b_col16[i]),
            .y       (mult_result_bf16[i])
        );

        always_comb begin
            case(fmt_s)
                3'd1: mult_result_mux[i] = mult_result_fp16[i];
                3'd2: mult_result_mux[i] = mult_result_bf16[i];
                default: mult_result_mux[i] = 32'hxxxxxxxx;
            endcase
        end

        VX_pipe_register #(
            .DATAW (32),
            .DEPTH (FMUL_LATENCY)
        ) pipe_mult (
            .clk      (clk),
            .reset    (reset),
            .enable   (enable),
            .data_in  (mult_result_mux[i]),
            .data_out (mult_result[i])
        );
    end

    //Accumulate reduction tree
    wire [31:0] red_in [0:LEVELS] [TCK];

    for (genvar i = 0; i < TCK; i++) begin : g_red_init_inputs
        assign red_in[0][i] = mult_result[i];
    end

    for (genvar lvl = 0; lvl < LEVELS; lvl++) begin : g_red_tree
        localparam integer CURSZ = TCK >> lvl;
        localparam integer OUTSZ = CURSZ >> 1;
        
        wire [31:0] red_comb [OUTSZ];
        
        for (genvar i = 0; i < OUTSZ; i++) begin : g_add
            VX_tcu_drl_fp32add fp32add (
                .enable  (enable),
                .a       (red_in[lvl][2*i+0]),
                .b       (red_in[lvl][2*i+1]),
                .y       (red_comb[i])
            );

            VX_pipe_register #(
                .DATAW (32),
                .DEPTH (FADD_LATENCY)
            ) pipe_red (
                .clk      (clk),
                .reset    (reset),
                .enable   (enable),
                .data_in  (red_comb[i]),
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

    wire [31:0] final_add_result;
    wire [31:0] result;

    // final accumulation
    VX_tcu_drl_fp32add final_fp32add (
        .enable  (enable),
        .a       (red_in[LEVELS][0]),
        .b       (delayed_c),
        .y       (final_add_result)
    );

    VX_pipe_register #(
        .DATAW (32),
        .DEPTH (FADD_LATENCY + FRND_LATENCY)
    ) pipe_acc (
        .clk     (clk),
        .reset   (reset),
        .enable  (enable),
        .data_in (final_add_result),
        .data_out(result)
    );

    assign d_val = `XLEN'(result);

endmodule
