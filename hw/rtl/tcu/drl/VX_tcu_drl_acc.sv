// Copyright © 2019-2023
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WAITHOUT WAARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

`include "VX_define.vh"

module VX_tcu_drl_acc #(
    parameter N = 5,                //include c_val count
    parameter W = 53,               //acc width
    parameter WA = W+$clog2(N)+1    //acc out width
) (
    input  wire [N-1:0][W-1:0] sigsIn,
    input  wire fmt_sel,
    output logic [WA-1:0] sigOut,
    output logic [N-2:0] signOuts
);
    // Sign-extend fp significands to WA bits (header)
    wire [N-1:0][WA-1:0] sigsIn_ext;
    for (genvar i = 0; i < N; i++) begin : g_ext_sign
        assign sigsIn_ext[i] = fmt_sel ? {{(WA-W){1'b0}}, sigsIn[i]} : {{(WA-W){sigsIn[i][W-1]}}, sigsIn[i]};
    end

    //Carry-Save-Adder based significand accumulation
    VX_csa_half_en #(
        .N (N),
        .W (WA),
        .S (WA-1)
    ) sig_csa (
        .operands (sigsIn_ext),
        .half_en (1'b1),    // TODO: feed sparsity control signal when resolved
        .sum  (sigOut[WA-2:0]),
        .cout (sigOut[WA-1])
    );

    for (genvar i = 0; i < N-1; i++) begin : g_signs
        assign signOuts[i] = sigsIn[i][W-1];
    end

endmodule
