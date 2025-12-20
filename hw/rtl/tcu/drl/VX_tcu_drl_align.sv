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

module VX_tcu_drl_align #(
    parameter N = 5,    //includes c_val
    parameter W = 53    //prod align bitwidth (use 25 for approx)
) (
    input wire [N-1:0][7:0] shift_amounts,
    input wire [N-1:0][24:0] sigs_in,
    input wire fmt_sel,
    output logic [N-1:0][W-1:0] sigs_out
);

    //extend + align + sign significands
    for (genvar i = 0; i < N; i++) begin : g_align
        wire [W-1:0] ext_sigs_in = {sigs_in[i], {W-25{1'b0}}};
        wire fp_sign = ext_sigs_in[W-1];
        wire [W-2:0] fp_sig = ext_sigs_in[W-2:0];
        wire [W-2:0] adj_sig = fp_sig >> shift_amounts[i];
        wire [W-1:0] fp_val = fp_sign ? -adj_sig : {1'b0, adj_sig};
        assign sigs_out[i] = fmt_sel ? ext_sigs_in : fp_val;
    end

endmodule

/*
        wire [23:0] adj_sig = shift_amount[3] ? 24'd0 : full_sig[i] >> shift_amount;      //reducing switching activity (power) by clamping to 0 if
                                                                                        //input won't make a significant impact on accumulated value
*/
