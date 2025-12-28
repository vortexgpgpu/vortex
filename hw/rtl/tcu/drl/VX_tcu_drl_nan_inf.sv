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

module VX_tcu_drl_nan_inf #(
    parameter N = 5  //includes c_val count
) (
    input wire [2:0] fmt_s,
    input wire [N-2:0][15:0] a_rows,
    input wire [N-2:0][15:0] b_cols,
    input wire [31:0] c_val,
    output logic [2:0] exceptions    //{sign, nan, inf}
);

    wire [N-2:0] sign_a_fp16, sign_b_fp16, nan_a_fp16, nan_b_fp16, inf_a_fp16, inf_b_fp16, zero_a_fp16, zero_b_fp16;
    wire [N-2:0] sign_a_bf16, sign_b_bf16, nan_a_bf16, nan_b_bf16, inf_a_bf16, inf_b_bf16, zero_a_bf16, zero_b_bf16;

    for (genvar i = 0; i < N-1; i++) begin : g_exc_flags    
        //FP16
        assign sign_a_fp16[i] = a_rows[i][15];
        assign sign_b_fp16[i] = b_cols[i][15];
        assign nan_a_fp16[i]  = (&a_rows[i][14:10]) & (|a_rows[i][9:0]);
        assign nan_b_fp16[i]  = (&b_cols[i][14:10]) & (|b_cols[i][9:0]);
        assign inf_a_fp16[i]  = (&a_rows[i][14:10]) & (~|a_rows[i][9:0]);
        assign inf_b_fp16[i]  = (&b_cols[i][14:10]) & (~|b_cols[i][9:0]);
        assign zero_a_fp16[i] = ~|a_rows[i][14:0];
        assign zero_b_fp16[i] = ~|b_cols[i][14:0];

        //BF16
        assign sign_a_bf16[i] = a_rows[i][15];
        assign sign_b_bf16[i] = b_cols[i][15];
        assign nan_a_bf16[i]  = (&a_rows[i][14:7]) & (|a_rows[i][6:0]);
        assign nan_b_bf16[i]  = (&b_cols[i][14:7]) & (|b_cols[i][6:0]);
        assign inf_a_bf16[i]  = (&a_rows[i][14:7]) & (~|a_rows[i][6:0]);
        assign inf_b_bf16[i]  = (&b_cols[i][14:7]) & (~|b_cols[i][6:0]);
        assign zero_a_bf16[i] = ~|a_rows[i][14:0];
        assign zero_b_bf16[i] = ~|b_cols[i][14:0];

    end

    //c_val exceptions
    wire sign_c = c_val[31];
    wire nan_c  = (&c_val[30:22]) & (|c_val[21:0]);
    wire inf_c  = (&c_val[30:23]) & (~|c_val[22:0]);

    //format mux
    logic [N-2:0] sign_a, sign_b, nan_a, nan_b, inf_a, inf_b, zero_a, zero_b;
    always_comb begin
        case (fmt_s)
            3'b001: begin
                sign_a = sign_a_fp16;
                sign_b = sign_b_fp16;
                nan_a  = nan_a_fp16;
                nan_b  = nan_b_fp16;
                inf_a  = inf_a_fp16;
                inf_b  = inf_b_fp16;
                zero_a = zero_a_fp16;
                zero_b = zero_b_fp16;
            end
            3'b010: begin
                sign_a = sign_a_bf16;
                sign_b = sign_b_bf16;
                nan_a  = nan_a_bf16;
                nan_b  = nan_b_bf16;
                inf_a  = inf_a_bf16;
                inf_b  = inf_b_bf16;
                zero_a = zero_a_bf16;
                zero_b = zero_b_bf16;
            end
            default: begin
                sign_a = {(N-1){1'b0}};
                sign_b = {(N-1){1'b0}};
                nan_a  = {(N-1){1'b0}};
                nan_b  = {(N-1){1'b0}};
                inf_a  = {(N-1){1'b0}};
                inf_b  = {(N-1){1'b0}};
                zero_a = {(N-1){1'b0}};
                zero_b = {(N-1){1'b0}};
            end
        endcase
    end

    //Input NaN
    wire input_has_nan = |nan_a | |nan_b | nan_c;

    //Multiply NaN
    wire [N-2:0] inf_times_zero;
    for (genvar i = 0; i < N-1; i++) begin : g_mult_nan
        assign inf_times_zero[i] = (inf_a[i] & zero_b[i]) | (zero_a[i] & inf_b[i]);
    end
    wire mult_generates_nan = |inf_times_zero;

    //Multiply Inf
    wire [N-2:0] prod_inf;
    wire [N-2:0] prod_sign;
    for (genvar i = 0; i < N-1; i++) begin : g_prod_inf
        assign prod_inf[i] = (inf_a[i] | inf_b[i]) & ~inf_times_zero[i];
        assign prod_sign[i] = sign_a[i] ^ sign_b[i];
    end

    //Addition NaN
    wire [N-2:0] prod_pos_inf = prod_inf & ~prod_sign;
    wire [N-2:0] prod_neg_inf = prod_inf & prod_sign;
    wire c_pos_inf = inf_c & ~sign_c;
    wire c_neg_inf = inf_c & sign_c;
    wire has_pos_inf = |prod_pos_inf | c_pos_inf;
    wire has_neg_inf = |prod_neg_inf | c_neg_inf;
    wire add_generates_nan = has_pos_inf & has_neg_inf;

    //Final exception flags
    wire result_nan = input_has_nan | mult_generates_nan | add_generates_nan;
    wire result_inf = (|prod_inf | inf_c) & ~result_nan;
    wire result_sign = has_neg_inf & ~has_pos_inf;
    assign exceptions = {result_sign, result_nan, result_inf};

endmodule
