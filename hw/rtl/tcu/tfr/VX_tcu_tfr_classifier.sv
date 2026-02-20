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

module VX_tcu_tfr_classifier import VX_tcu_pkg::*; #(
    parameter EXP_W = 8,
    parameter MAN_W = 23
) (
    input wire [EXP_W-1:0] exp,
    input wire [MAN_W-1:0] man,
    input wire [EXP_W-1:0] max_exp,
    output fedp_class_t    cls
);
    wire exp_zero = ~|exp;
    wire exp_ones = (exp == max_exp);

    wire man_non_zero = |man;
    wire man_zero     = ~man_non_zero;

    assign cls.is_zero = exp_zero & man_zero;
    assign cls.is_sub  = exp_zero & man_non_zero;
    assign cls.is_inf  = exp_ones & man_zero;
    assign cls.is_nan  = exp_ones & man_non_zero;

endmodule
