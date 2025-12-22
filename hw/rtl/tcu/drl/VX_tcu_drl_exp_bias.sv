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

module VX_tcu_drl_exp_bias (
    input wire enable,
    input wire [2:0] fmt_s,
    input wire [15:0] a,
    input wire [15:0] b,
    output logic [7:0] raw_exp_y,
    output logic exp_low_larger,
    output logic [6:0] raw_exp_diff
);
    `UNUSED_VAR({a, b, enable});

    //FP16 exponent addition and bias
    wire [7:0] raw_exp_fp16;
    wire [7:0] fp16_32_conv_bias = 8'd98;    //127-30 + 1
    VX_csa_tree #(
        .N(3),
        .W(8),
        .S(8)
    ) biasexp_fp16(
        .operands({{3'd0, a[14:10]}, {3'd0, b[14:10]}, fp16_32_conv_bias}),
        .sum     (raw_exp_fp16),
        `UNUSED_PIN (cout)
    );

    //BF16 exponent addition and bias
    wire [7:0] raw_exp_bf16;
    wire [9:0] neg_bias = 10'b1110000010; //-127+1
    wire [9:0] raw_exp_bf16_signed;
    `UNUSED_VAR(raw_exp_bf16_signed);
    VX_csa_tree #(
        .N(3),
        .W(10),    //8 + log2(3) extend for sign handling
        .S(10)
    ) biasexp_bf16(
        .operands({{2'd0, a[14:7]}, {2'd0, b[14:7]}, neg_bias}),
        .sum     (raw_exp_bf16_signed),
        `UNUSED_PIN (cout)
    );
    assign raw_exp_bf16 = raw_exp_bf16_signed[9] ? -raw_exp_bf16_signed[7:0] : raw_exp_bf16_signed[7:0];

    //FP8 (E4M3) exponent addition and bias
    wire [7:0] raw_exp_fp8;
    wire [1:0][4:0] raw_exp_fp8_sub;
    for (genvar i = 0; i < 2; i++) begin  :  g_fp8_sub
        VX_ks_adder #(
            .N(4)
        ) raw_exp_fp8_sub_add (
            .dataa (a[(i*8)+6 -: 4]),
            .datab (b[(i*8)+6 -: 4]),
            .sum   (raw_exp_fp8_sub[i][3:0]),
            .cout  (raw_exp_fp8_sub[i][4])
        );
    end
    wire [5:0] raw_exp_fp8_diff = {1'b0, raw_exp_fp8_sub[1]} - {1'b0, raw_exp_fp8_sub[0]};
    wire fp8_exp_low_larger = raw_exp_fp8_diff[5];
    wire [4:0] raw_exp_fp8_unbiased = fp8_exp_low_larger ? raw_exp_fp8_sub[0] : raw_exp_fp8_sub[1];
    wire [7:0] fp8_conv_bias_fp32 = 8'd115;    //127-14+2
    VX_ks_adder #(
        .N(8)
    ) biasexp_fp8 (
        .dataa ({3'd0, raw_exp_fp8_unbiased}),
        .datab (fp8_conv_bias_fp32),
        .sum   (raw_exp_fp8),
        `UNUSED_PIN (cout)
    );

    //BF8 (E5M2) exponent addition and bias
    wire [7:0] raw_exp_bf8;
    wire [1:0][5:0] raw_exp_bf8_sub;
    for (genvar j = 0; j < 2; j++) begin  :  g_bf8_sub
        VX_ks_adder #(
            .N(5)
        ) raw_exp_bf8_sub_add (
            .dataa (a[(j*8)+6 -: 5]),
            .datab (b[(j*8)+6 -: 5]),
            .sum   (raw_exp_bf8_sub[j][4:0]),
            .cout  (raw_exp_bf8_sub[j][5])
        );
    end
    wire [6:0] raw_exp_bf8_diff = {1'b0, raw_exp_bf8_sub[1]} - {1'b0, raw_exp_bf8_sub[0]};
    wire bf8_exp_low_larger = raw_exp_bf8_diff[6];
    wire [5:0] raw_exp_bf8_unbiased = bf8_exp_low_larger ? raw_exp_bf8_sub[0] : raw_exp_bf8_sub[1];
    wire [7:0] bf8_conv_bias_fp32 = 8'd99;    //127-30+2
    VX_ks_adder #(
        .N(8)
    ) biasexp_bf8 (
        .dataa ({2'd0, raw_exp_bf8_unbiased}),
        .datab (bf8_conv_bias_fp32),
        .sum   (raw_exp_bf8),
        `UNUSED_PIN (cout)
    );

    //Select exp out based on datatype
    always_comb begin
        case(fmt_s[2:0])
            3'd1: begin
                raw_exp_y      = raw_exp_fp16;
                exp_low_larger = 1'bx;
                raw_exp_diff   = 7'dx;                 
            end
            3'd2: begin
                raw_exp_y      = raw_exp_bf16;
                exp_low_larger = 1'bx;
                raw_exp_diff   = 7'dx;                    
            end
            3'd3: begin
                raw_exp_y      = raw_exp_fp8;
                exp_low_larger = fp8_exp_low_larger;
                raw_exp_diff   = {raw_exp_fp8_diff[5], raw_exp_fp8_diff};                    
            end
            3'd4: begin
                raw_exp_y      = raw_exp_bf8;
                exp_low_larger = bf8_exp_low_larger;
                raw_exp_diff   = raw_exp_bf8_diff;                    
            end
            default: begin
                raw_exp_y      = 8'dx;
                exp_low_larger = 1'bx;
                raw_exp_diff   = 7'dx;
            end
        endcase
    end

endmodule
