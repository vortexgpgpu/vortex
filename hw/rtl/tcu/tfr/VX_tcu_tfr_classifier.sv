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
    parameter N     = 1,
    parameter WIDTH = 32,
    parameter FMT   = 0
) (
    input wire [N-1:0][WIDTH-1:0] val,
    output fedp_class_t [N-1:0]   cls
);
    localparam EXP_BITS = VX_tcu_pkg::exp_bits(FMT);
    localparam MAN_BITS = VX_tcu_pkg::sig_bits(FMT);
    localparam SIGN_POS = VX_tcu_pkg::sign_pos(FMT);

    localparam EXP_START = SIGN_POS - 1;
    localparam EXP_END   = EXP_START - EXP_BITS + 1;

    localparam MAN_START = EXP_END - 1;
    localparam MAN_END   = MAN_START - MAN_BITS + 1;

    for (genvar i = 0; i < N; ++i) begin : g_cls
        wire sign = val[i][SIGN_POS];
        wire [EXP_BITS-1:0] exp  = val[i][EXP_START : EXP_END];
        wire [MAN_BITS-1:0] man  = val[i][MAN_START : MAN_END];

        wire exp_zero = ~|exp;
        wire exp_ones = &exp;

        wire man_non_zero = |man;
        wire man_zero     = ~man_non_zero;

        assign cls[i].sign    = sign;
        assign cls[i].is_zero = exp_zero & man_zero;
        assign cls[i].is_sub  = exp_zero & man_non_zero;
        assign cls[i].is_inf  = exp_ones & man_zero;
        assign cls[i].is_nan  = exp_ones & man_non_zero;
    end

endmodule
