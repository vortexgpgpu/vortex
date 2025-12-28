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
    input wire [N-2:0] sparse_mask,
    output logic [N-1:0][W-1:0] sigs_out
);

    //input power gating
    wire [N-1:0][7:0] gated_shift_amounts;
    wire [N-1:0][24:0] gated_sigs_in;
    for (genvar i = 0; i < N-1; i++) begin : g_power_gating
        assign gated_sigs_in[i] = ({25{sparse_mask[i]}} & sigs_in[i]);
        assign gated_shift_amounts[i] = ({8{sparse_mask[i]}} & shift_amounts[i]);
    end
    assign gated_sigs_in[N-1] = sigs_in[N-1];               //c_val
    assign gated_shift_amounts[N-1] = shift_amounts[N-1];

    //extend + align + sign significands
    for (genvar i = 0; i < N; i++) begin : g_align
        wire [W-1:0] ext_sigs_in = {gated_sigs_in[i], {W-25{1'b0}}};
        wire fp_sign = ext_sigs_in[W-1];
        wire [W-2:0] fp_sig = ext_sigs_in[W-2:0];
        wire [W-2:0] adj_sig = fp_sig >> gated_shift_amounts[i];
        wire [W-1:0] fp_val = fp_sign ? -adj_sig : {1'b0, adj_sig};
        assign sigs_out[i] = fmt_sel ? ext_sigs_in : fp_val;
    end

endmodule
