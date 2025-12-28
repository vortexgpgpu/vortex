// Copyright © 2019-2023
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WAARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
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
    input  wire [N-2:0] sparse_mask,
    output logic [WA-1:0] sigOut,
    output logic [N-2:0] signOuts
);

    //input power gating
    wire [N-1:0][W-1:0] gated_sigsIn;
    for (genvar i = 0; i < N-1; i++) begin : g_power_gating
        assign gated_sigsIn[i] = ({W{sparse_mask[i]}} & sigsIn[i]);
    end
    assign gated_sigsIn[N-1] = sigsIn[N-1];  //c_val

    //Sign-extend fp significands to WA bits (header)
    wire [N-1:0][WA-1:0] sigsIn_ext;
    for (genvar i = 0; i < N; i++) begin : g_ext_sign
        assign sigsIn_ext[i] = fmt_sel ? {{(WA-W){1'b0}}, gated_sigsIn[i]} : {{(WA-W){gated_sigsIn[i][W-1]}}, gated_sigsIn[i]};
    end

    //Carry-Save-Adder based significand accumulation
    if (N >= 7) begin : g_large_acc
        VX_csa_mod4 #(
            .N (N),
            .W (WA),
            .S (WA-1)
        ) sig_csa (
            .operands (sigsIn_ext),
            .sum      (sigOut[WA-2:0]),
            .cout     (sigOut[WA-1])
        );
    end else begin : g_small_acc
        VX_csa_tree #(
            .N (N),
            .W (WA),
            .S (WA-1)
        ) sig_csa (
            .operands (sigsIn_ext),
            .sum      (sigOut[WA-2:0]),
            .cout     (sigOut[WA-1])
        );
    end

    //Extract prod sigs signs for INT
    for (genvar i = 0; i < N-1; i++) begin : g_signs
        assign signOuts[i] = sigsIn[i][W-1];
    end

endmodule
