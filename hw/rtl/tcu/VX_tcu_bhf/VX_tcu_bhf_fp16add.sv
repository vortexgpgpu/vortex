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

module VX_tcu_bhf_fp16add (
    input wire enable,
    input  wire [16:0] a,           //recoded FP16 a
    input  wire [16:0] b,           //recoded FP16 b
    output logic [16:0] y           //recoded FP16 y
);

    `UNUSED_VAR(enable);

    //FP16 format constants
    localparam EXP_WIDTH = 5;
    localparam SIG_WIDTH = 11;
    localparam CONTROL = 1'b1;          //Default (tininess after rounding)
    localparam [2:0] RNE = 3'b000;      //Round Near Even mode

    wire [EXP_WIDTH+SIG_WIDTH:0] sum_recoded;      //33-bit recoded result
    wire [4:0] exception_flags;
    `UNUSED_VAR(exception_flags)           

    //Performing addition in recoded format
    addRecFN #(
        .expWidth(EXP_WIDTH),
        .sigWidth(SIG_WIDTH)
    ) adder (
        .control(CONTROL),
        .subOp(1'b0),               // Addition SubOpcode
        .a(a),
        .b(b),
        .roundingMode(RNE),
        .out(sum_recoded),
        .exceptionFlags(exception_flags)
    );

    assign y = sum_recoded;

    /*
    //Final result exception handling
    wire result_is_inf = exception_flags[3];
    wire result_is_nan = exception_flags[4] | (|exception_flags[2:0]);

    always_comb begin
        casez({result_is_nan, result_is_inf})
            2'b1?: y = 32'h7FC00000;                                //Canonical FP32 quiet NaN
            2'b01: y = y_wo_exp[31] ? 32'hFF800000 : 32'h7F800000;  //Signed FP32 infinity
            default: y = y_wo_exp;
        endcase
    end
    */

endmodule
