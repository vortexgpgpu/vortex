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
    parameter N = 5         //include c_val count
) (
    input  wire [N-1:0][24:0] sigsIn,
    output logic [25+$clog2(N):0] sigOut,
    output logic [N-2:0] signOuts
);

    //Carry-Save-Adder based significand accumulation
    VX_csa_tree #(
        .N (N),
        .W (25)
    ) sig_csa (
        .operands (sigsIn),
        .sum  (sigOut[25+$clog2(N)-1:0]),
        .cout (sigOut[25+$clog2(N)])
    );

    for (genvar i = 0; i < N-1; i++) begin : g_signs
        assign signOuts[i] = sigsIn[i][24];
    end
    
endmodule
