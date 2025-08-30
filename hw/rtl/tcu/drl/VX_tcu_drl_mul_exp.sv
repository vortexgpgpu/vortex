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

module VX_tcu_drl_mul_exp #(
    parameter N = 5  //includes c_val count
) (
    input wire enable,
    input wire [3:0] fmt_s,
    input wire [N-2:0][15:0] a_rows,
    input wire [N-2:0][15:0] b_cols,
    input wire [31:0] c_val,
    output logic [7:0] raw_max_exp,
    output logic [N-1:0][24:0] sigs_out
);

    //raw fp signals
    wire [N-2:0] mul_sign_fp16, mul_sign_bf16, mul_sign_fp8, mul_sign_bf8;
    wire [N-2:0][7:0] mul_exp_fp16, mul_exp_bf16, mul_exp_fp8, mul_exp_bf8;
    wire [N-2:0][23:0] mul_sig_fp16, mul_sig_bf16, mul_sig_fp8, mul_sig_bf8;

    //raw int signals
    wire [N-2:0][16:0] mul_sig_int8;
    wire [N-2:0][9:0]  mul_sig_uint4;

    //muxed signals
    logic [N-1:0] mul_sign_mux;
    logic [N-1:0][7:0] mul_exp_mux;
    logic [N-1:0][23:0] mul_sig_mux;
    logic [N-1:0][24:0] int_vals_mux;

    for (genvar i = 0; i < N-1; i++) begin : g_prod
        // FP16 multiplication
        VX_tcu_drl_fp16mul fp16mul (
            .enable    (enable),
            .a         (a_rows[i]),
            .b         (b_cols[i]),
            .sign_y    (mul_sign_fp16[i]),
            .raw_exp_y (mul_exp_fp16[i]),
            .raw_sig_y (mul_sig_fp16[i])
        );

        // BF16 multiplication
        VX_tcu_drl_bf16mul bf16mul (
            .enable    (enable),
            .a         (a_rows[i]),
            .b         (b_cols[i]),
            .sign_y    (mul_sign_bf16[i]),
            .raw_exp_y (mul_exp_bf16[i]),
            .raw_sig_y (mul_sig_bf16[i])
        );

        // FP8 E4M3 multiplication
        VX_tcu_drl_fp8mul fp8mul (
            .enable    (enable),
            .a         (a_rows[i]),
            .b         (b_cols[i]),
            .sign_y    (mul_sign_fp8[i]),
            .raw_exp_y (mul_exp_fp8[i]),
            .raw_sig_y (mul_sig_fp8[i])
        );

        // FP8 E5M2 multiplication
        VX_tcu_drl_bf8mul bf8mul (
            .enable    (enable),
            .a         (a_rows[i]),
            .b         (b_cols[i]),
            .sign_y    (mul_sign_bf8[i]),
            .raw_exp_y (mul_exp_bf8[i]),
            .raw_sig_y (mul_sig_bf8[i])
        );

        // INT8 multiplication
        VX_tcu_drl_int8mul int8mul (
            .enable (enable),
            .a      (a_rows[i]),
            .b      (b_cols[i]),
            .signed_y (mul_sig_int8[i])
        );

        // UINT4 multiplication
        VX_tcu_drl_uint4mul uint4mul (
            .enable (enable),
            .a      (a_rows[i]),
            .b      (b_cols[i]),
            .unsigned_y (mul_sig_uint4[i])
        );

        //FP Format selection
        always_comb begin
            case(fmt_s[2:0])
                3'd1: begin
                    mul_sign_mux[i] = mul_sign_fp16[i];
                    mul_exp_mux[i]  = mul_exp_fp16[i];                 
                    mul_sig_mux[i]  = mul_sig_fp16[i];
                end
                3'd2: begin
                    mul_sign_mux[i] = mul_sign_bf16[i];
                    mul_exp_mux[i]  = mul_exp_bf16[i];                    
                    mul_sig_mux[i]  = mul_sig_bf16[i];
                end
                3'd3: begin
                    mul_sign_mux[i] = mul_sign_fp8[i];
                    mul_exp_mux[i]  = mul_exp_fp8[i];                    
                    mul_sig_mux[i]  = mul_sig_fp8[i];
                end
                3'd4: begin
                    mul_sign_mux[i] = mul_sign_bf8[i];
                    mul_exp_mux[i]  = mul_exp_bf8[i];                    
                    mul_sig_mux[i]  = mul_sig_bf8[i];
                end
                default: begin
                    mul_sign_mux[i] = 1'bx;
                    mul_exp_mux[i]  = 8'hxx;
                    mul_sig_mux[i]  = 24'hxxxxxx;
                end
            endcase
        end

        //INT Format selection (sign extend)
        always_comb begin
            case(fmt_s[2:0])
                3'd1: begin
                    int_vals_mux[i] = 25'($signed(mul_sig_int8[i]));
                end
                3'd4: begin
                    int_vals_mux[i] = {15'd0, mul_sig_uint4[i]};
                end
                default: begin
                    int_vals_mux[i] = 25'hxxxxxxx;
                end
            endcase
        end
    end
    
    //FP c_val integration
    always_comb begin
        mul_sign_mux[N-1] = c_val[31];
        mul_exp_mux[N-1]  = c_val[30:23];
        mul_sig_mux[N-1]  = {1'b1, c_val[22:0]};
    end

    //INT c_val integration
    assign int_vals_mux[N-1] = c_val[24:0];

    //Raw maximum exponent finder (in parallel to mul) and shift amounts
    wire [N-1:0][7:0] shift_amounts;
    VX_tcu_drl_max_exp #(
        .N(N)
    ) find_max_exp (
        .exponents (mul_exp_mux),
        .max_exp   (raw_max_exp),
        .shift_amounts (shift_amounts)
    );

    //Aligned + signed significands
    for (genvar i = 0; i < N; i++) begin : g_align_signed
        wire [23:0] adj_sig = mul_sig_mux[i] >> shift_amounts[i];
        wire [24:0] fp_val = mul_sign_mux[i] ? -adj_sig : {1'b0, adj_sig};
        assign sigs_out[i] = (fmt_s[3]) ? int_vals_mux[i] : fp_val;
    end

endmodule
