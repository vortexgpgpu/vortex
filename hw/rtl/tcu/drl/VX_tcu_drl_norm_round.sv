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
    
module VX_tcu_drl_norm_round #(
    parameter N = 5    //includes c_val
) (
    input wire [7:0] max_exp,
    input wire [24+$clog2(N)-1:0] acc_sig,
    output wire [7:0] norm_exp,
    output wire [22:0] rounded_sig
);

    localparam ACC_WIDTH = 24 + $clog2(N);

    //Leading zero counter
    wire [$clog2(ACC_WIDTH)-1:0] lz_count;
    VX_lzc #(
        .N (ACC_WIDTH)
    ) lzc (
        .data_in   (acc_sig),
        .data_out  (lz_count),
        `UNUSED_PIN(valid_out)
    );

    wire [7:0] shift_amount = 8'(lz_count) - 8'($clog2(N));
    assign norm_exp = max_exp - shift_amount;
    
    //Move leading 1 to MSB (mantissa norm)
    wire [ACC_WIDTH-1:0] shifted_acc_sig = acc_sig << lz_count;
    //RNE rounding
    wire lsb = shifted_acc_sig[ACC_WIDTH-2-22];
    wire guard_bit = shifted_acc_sig[ACC_WIDTH-2-23];
    wire round_bit = shifted_acc_sig[ACC_WIDTH-2-24];
    wire sticky_bit = |shifted_acc_sig[ACC_WIDTH-2-25:0];
    //wire round_up = guard_bit & (round_bit | sticky_bit | lsb);   //TODO: standard RNE should've worked but doesnt?
    wire round_up = guard_bit | (round_bit | sticky_bit | lsb);
    
    //Index [ACC_WIDTH-1] becomes the hidden 1
    assign rounded_sig = shifted_acc_sig[ACC_WIDTH-2 : ACC_WIDTH-2-22] + 23'(round_up);

endmodule 
