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

module VX_tcu_drl_acc #(
    parameter N = 5,         //include c_val count
    parameter W = 25+$clog2(N)+1
) (
    input  wire [N-1:0][24:0] sigsIn,
    input  wire fmt_sel,
    output logic [W-1:0] sigOut,
    output logic [N-2:0] signOuts
);
    // Sign-extend fp significands to W bits
    wire [N-1:0][W-1:0] sigsIn_ext;
    for (genvar i = 0; i < N; i++) begin : g_ext_sign
        assign sigsIn_ext[i] = fmt_sel ? {{(W-25){1'b0}}, sigsIn[i]} : {{(W-25){sigsIn[i][24]}}, sigsIn[i]};
    end

    //Carry-Save-Adder based significand accumulation
    VX_csa_mod4 #(
        .N (N),
        .W (W),
        .S (W-1)
    ) sig_csa (
        .operands (sigsIn_ext),
        .sum  (sigOut[W-2:0]),
        .cout (sigOut[W-1])
    );

    for (genvar i = 0; i < N-1; i++) begin : g_signs
        assign signOuts[i] = sigsIn[i][24];
    end

endmodule
