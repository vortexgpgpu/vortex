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

module VX_tcu_fedp_drl #(
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
    localparam FMUL_LATENCY = 1;
    localparam FADD_LATENCY = 1;
    localparam FRND_LATENCY = 1;
    localparam RED_LATENCY  = LEVELS * FADD_LATENCY;
    localparam ACC_LATENCY  = RED_LATENCY + FADD_LATENCY;
    `STATIC_ASSERT (LATENCY == (FMUL_LATENCY+ACC_LATENCY+FRND_LATENCY), ("invalid parameter!"));

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
    wire acc_sign;
    wire [7:0] max_exp;
    wire [24+$clog2(TCK)-1:0] acc_sig;    //23 mantissa + 1 hidden + log2(N) bits

    VX_tcu_drl_acc #(
        .N(TCK)
    ) csa_acc (
        .fp32operands(mult_result),
        .signOut(acc_sign),
        .expOut(max_exp),
        .sigOut(acc_sig)
    );

    wire pipe_result_sign;
    wire [7:0] pipe_max_exp;
    wire [24+$clog2(TCK)-1:0] pipe_acc_sig;
    
    VX_pipe_register #(
        .DATAW (1+8+24+$clog2(TCK)),
        .DEPTH (RED_LATENCY)
    ) pipe_acc (
        .clk     (clk),
        .reset   (reset),
        .enable  (enable),
        .data_in ({acc_sign, max_exp, acc_sig}),
        .data_out({pipe_result_sign, pipe_max_exp, pipe_acc_sig})
    );

    //Normalization of accumulated significand before final add
    //Leading zero counter
    wire [$clog2(24+$clog2(TCK))-1:0] lz_count;
    VX_lzc #(
        .N (24+$clog2(TCK))
    ) lzc (
        .data_in   (pipe_acc_sig),
        .data_out  (lz_count),
        `UNUSED_PIN(valid_out)
    );

    wire [7:0] shift_amount = 8'($clog2(TCK)) - 8'(lz_count);
    wire [7:0] norm_exp = pipe_max_exp + shift_amount;
    wire [24+$clog2(TCK)-1:0] shifted_acc_sig = pipe_acc_sig << lz_count;
    wire [22:0] norm_sig = shifted_acc_sig[24+$clog2(TCK)-2 : 24+$clog2(TCK)-2-22];
   `UNUSED_VAR (shifted_acc_sig)

    wire [31:0] norm_result;
    VX_pipe_register #(
        .DATAW (32),
        .DEPTH (FRND_LATENCY)
    ) pipe_norm (
        .clk     (clk),
        .reset   (reset),
        .enable  (enable),
        .data_in ({pipe_result_sign, norm_exp, norm_sig}),
        .data_out(norm_result)
    );

    wire [31:0] delayed_c;
    VX_pipe_register #(
        .DATAW (32),
        .DEPTH (FMUL_LATENCY + RED_LATENCY + FRND_LATENCY)
    ) pipe_c (
        .clk     (clk),
        .reset   (reset),
        .enable  (enable),
        .data_in (c_val[31:0]),
        .data_out(delayed_c)
    );    

    wire [31:0] final_add_result;
    VX_tcu_drl_fp32add final_fp32add (
        .enable  (enable),
        .a       (norm_result),
        .b       (delayed_c),
        .y       (final_add_result)
    );    

    wire [31:0] fedp_result;
    VX_pipe_register #(
        .DATAW (32),
        .DEPTH (FRND_LATENCY)
    ) pipe_final (
        .clk     (clk),
        .reset   (reset),
        .enable  (enable),
        .data_in (final_add_result),
        .data_out(fedp_result)
    );

    assign d_val = `XLEN'(fedp_result);
endmodule
