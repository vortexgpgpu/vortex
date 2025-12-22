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
    output logic [N-1:0][7:0] shift_amounts,
    output logic [N-1:0][24:0] raw_sigs,
    output logic [2:0] exceptions
);

    //muxed signals
    logic [N-1:0][7:0] raw_exps;

    for (genvar i = 0; i < N-1; i++) begin : g_prod
        wire exp_low_larger;
        wire [6:0] raw_exp_diff;
        
        //shared significand multiplier
        VX_tcu_drl_shared_mul shared_mul_inst (
            .enable         (enable),
            .fmt_s          (fmt_s),
            .a              (a_rows[i]),
            .b              (b_cols[i]),
            .exp_low_larger (exp_low_larger),
            .raw_exp_diff   (raw_exp_diff),
            .y              (raw_sigs[i])
        );

        //exponent add and bias
        VX_tcu_drl_exp_bias exp_bias_inst (
            .enable         (enable),
            .fmt_s          (fmt_s[2:0]),
            .a              (a_rows[i]),
            .b              (b_cols[i]),
            .raw_exp_y      (raw_exps[i]),
            .exp_low_larger (exp_low_larger),
            .raw_exp_diff   (raw_exp_diff)
        );
    end
    
    //c_val integration
    assign raw_exps[N-1] = c_val[30:23];
    assign raw_sigs[N-1] = fmt_s[3] ? c_val[24:0] : {c_val[31], 1'b1, c_val[22:0]};

    //Raw maximum exponent finder (in parallel to mul) and shift amounts
    VX_tcu_drl_max_exp #(
        .N(N)
    ) find_max_exp (
        .exponents     (raw_exps),
        .max_exp       (raw_max_exp),
        .shift_amounts (shift_amounts)
    );

    //NaN/Inf exception flag generation
    VX_tcu_drl_nan_inf #(
        .N(N)
    ) nan_inf_exc (
        .enable     (enable),
        .fmt_s      (fmt_s[2:0]),
        .a_rows     (a_rows),
        .b_cols     (b_cols),
        .c_val      (c_val),
        .exceptions (exceptions)
    );
    
endmodule
