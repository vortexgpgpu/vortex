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
    output logic signOut,
    output logic [24+$clog2(N)-1:0] sigOut
);

    //Carry-Save-Adder based significand accumulation
    wire [25+$clog2(N)-1:0] signed_sum_sig;
    VX_csa_tree #(
        .N (N),
        .W (25)
    ) sig_csa (
        .operands (sigsIn),
        .sum (signed_sum_sig)
    );

    //Extracting magnitude from signed result
    wire sum_sign = signed_sum_sig[25+$clog2(N)-1];
    wire [24+$clog2(N)-1:0] abs_sum;
    assign abs_sum = sum_sign ? -signed_sum_sig[24+$clog2(N)-1:0] : signed_sum_sig[24+$clog2(N)-1:0];

    assign signOut = sum_sign;
    assign sigOut = abs_sum;
endmodule
