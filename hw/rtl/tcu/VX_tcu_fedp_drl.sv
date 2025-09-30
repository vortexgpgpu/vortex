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
    parameter LATENCY = 0,
    parameter N = 2
) (
    input  wire clk,
    input  wire reset,
    input  wire enable,

    input  wire [3:0] fmt_s,
    input  wire [3:0] fmt_d,

    input  wire [N-1:0][`XLEN-1:0] a_row,
    input  wire [N-1:0][`XLEN-1:0] b_col,
    input  wire [`XLEN-1:0] c_val,
    output wire [`XLEN-1:0] d_val
);

    localparam TCK = 2 * N;
    localparam FMUL_LATENCY = 1;
    localparam ALN_LATENCY  = 1;
    localparam ACC_LATENCY  = 1;
    localparam FRND_LATENCY = 1;
    localparam TOTAL_LATENCY= FMUL_LATENCY + ALN_LATENCY + ACC_LATENCY + FRND_LATENCY;
    `STATIC_ASSERT (LATENCY == 0 || LATENCY == TOTAL_LATENCY, ("invalid latency! expected=%0d, actual=%0d", TOTAL_LATENCY, LATENCY));

    `UNUSED_VAR ({fmt_d, c_val});

    //Unpack two 16-bit elements from every 32-bit register
    wire [TCK-1:0][15:0] a_row16;
    wire [TCK-1:0][15:0] b_col16;

    for (genvar i = 0; i < N; i++) begin : g_unpack
        assign a_row16[2*i]   = a_row[i][15:0];
        assign a_row16[2*i+1] = a_row[i][31:16];
        assign b_col16[2*i]   = b_col[i][15:0];
        assign b_col16[2*i+1] = b_col[i][31:16];
    end

    //Transprecision Mul & Max Exp
    wire [6:0] hi_c = c_val[31:25];   //c_val[24:0] acc is taken care of in acc stage
    wire fmt_sel = fmt_s[3];
    wire [7:0] raw_max_exp;
    wire [TCK:0][7:0] shift_amounts;
    wire [TCK:0][24:0] raw_sigs;

    VX_tcu_drl_mul_exp #(
        .N(TCK+1)
    ) mul_exp (
        .enable        (enable),
        .fmt_s         (fmt_s),
        .a_rows        (a_row16),
        .b_cols        (b_col16),
        .c_val         (c_val[31:0]),
        .raw_max_exp   (raw_max_exp),
        .shift_amounts (shift_amounts),
        .raw_sigs      (raw_sigs)
    );

    //Stage 1 pipeline reg
    wire [7:0] pipe_raw_max_exp;
    wire [TCK:0][7:0] pipe_shift_amounts;
    wire [TCK:0][24:0] pipe_raw_sigs;
    wire [6:0] pipe_hi_c;
    wire pipe_fmt_sel;
    VX_pipe_register #(
        .DATAW (8+((TCK+1)*8)+((TCK+1)*25)+7+1),
        .DEPTH (FMUL_LATENCY)
    ) pipe_fmul (
        .clk     (clk),
        .reset   (reset),
        .enable  (enable),
        .data_in ({raw_max_exp, shift_amounts, raw_sigs, hi_c, fmt_sel}),
        .data_out({pipe_raw_max_exp, pipe_shift_amounts, pipe_raw_sigs, pipe_hi_c, pipe_fmt_sel})
    );

    //Significand Alignment
    wire [TCK:0][24:0] aln_sigs;
    wire [7:0] aln_max_exp = pipe_raw_max_exp;
    wire [6:0] aln_hi_c = pipe_hi_c;
    wire aln_fmt_sel = pipe_fmt_sel;

    VX_tcu_drl_align #(
        .N(TCK+1)
    ) sigs_aln (
        .shift_amounts (pipe_shift_amounts),
        .sigs_in       (pipe_raw_sigs),
        .fmt_sel       (pipe_fmt_sel),
        .sigs_out      (aln_sigs)
    );

    //Stage 2 pipeline reg
    wire [7:0] pipe_aln_max_exp;
    wire [TCK:0][24:0] pipe_aln_sigs;
    wire [6:0] pipe_aln_hi_c;
    wire pipe_aln_fmt_sel;
    VX_pipe_register #(
        .DATAW (8+((TCK+1)*25)+7+1),
        .DEPTH (ALN_LATENCY)
    ) pipe_aln (
        .clk     (clk),
        .reset   (reset),
        .enable  (enable),
        .data_in ({aln_max_exp, aln_sigs, aln_hi_c, aln_fmt_sel}),
        .data_out({pipe_aln_max_exp, pipe_aln_sigs, pipe_aln_hi_c, pipe_aln_fmt_sel})
    );

    //Accumulate CSA reduction tree
    wire [7:0] acc_max_exp = pipe_aln_max_exp;
    wire [6:0] acc_hi_c = pipe_aln_hi_c;
    wire acc_fmt_sel = pipe_aln_fmt_sel;
    wire [25+$clog2(TCK+1):0] acc_sig;    //23 mantissa + 1 hidden + 1 sign + log2(N) bits
    wire [TCK-1:0] sigs_sign;    //sign bits of all operands (for int math)

    VX_tcu_drl_acc #(
        .N(TCK+1)
    ) csa_acc (
        .sigsIn   (pipe_aln_sigs),
        .fmt_sel  (pipe_aln_fmt_sel),
        .sigOut   (acc_sig),
        .signOuts (sigs_sign)
    );

    //Stage 3 pipeline reg
    wire [7:0] pipe_acc_max_exp;
    wire [6:0] pipe_acc_hi_c;
    wire pipe_acc_fmt_sel;
    wire [25+$clog2(TCK+1):0] pipe_acc_sig;
    wire [TCK-1:0] pipe_sigs_sign;
    VX_pipe_register #(
        .DATAW (8+25+$clog2(TCK+1)+1+7+1+TCK),
        .DEPTH (ACC_LATENCY)
    ) pipe_acc (
        .clk     (clk),
        .reset   (reset),
        .enable  (enable),
        .data_in ({acc_max_exp, acc_sig, acc_hi_c, acc_fmt_sel, sigs_sign}),
        .data_out({pipe_acc_max_exp, pipe_acc_sig, pipe_acc_hi_c, pipe_acc_fmt_sel, pipe_sigs_sign})
    );

    //Normalization and RNE of accumulated significand for FP
    //Final upper 7-bit addition for INT
    wire [31:0] final_result;

    VX_tcu_drl_norm_round #(
        .N(TCK+1)
    ) norm_round (
        .max_exp     (pipe_acc_max_exp),
        .acc_sig     (pipe_acc_sig),
        .hi_c        (pipe_acc_hi_c),
        .sigSigns    (pipe_sigs_sign),
        .fmt_sel     (pipe_acc_fmt_sel),
        .result      (final_result)
    );

    //Stage 4 pipeline reg
    wire [31:0] fedp_result;
    VX_pipe_register #(
        .DATAW (32),
        .DEPTH (FRND_LATENCY)
    ) pipe_norm_round (
        .clk     (clk),
        .reset   (reset),
        .enable  (enable),
        .data_in (final_result),
        .data_out(fedp_result)
    );

    assign d_val = `XLEN'(fedp_result);
endmodule
