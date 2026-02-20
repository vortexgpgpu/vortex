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

module VX_wallace_mul #(
    parameter N = 8,
    parameter P = 2 * N,
    parameter CPA_KS = 1 // Use Kogge-Stone CPA
) (
    input wire [N-1:0]  a,
    input wire [N-1:0]  b,
    output wire [P-1:0] p
);
    wire [N-1:0][2*N-1:0] pp;    //partial products, double width (shifted)

    for (genvar g = 0; g < N; g++) begin: g_pp_loop
        for (genvar h = 0; h < N; h++) begin: g_and_loop
            assign pp[g][h+g] = a[h] & b[g];
        end
        if (g != 0) begin : g_bit_fill
            assign pp[g][g-1:0] = {g{1'b0}};    //fill lower bits with zeros
        end
        assign pp[g][2*N-1:N+g] = {(N-g){1'b0}};    //fill upper bits with zeros
    end

    wire [P-1:0] sum_vec, carry_vec;
    VX_csa_tree #(
        .N (N),
        .W (2*N),
        .S (P)
    ) pp_acc (
        .operands (pp),
        .sum  (sum_vec),
        .carry(carry_vec)
    );

    // Final CPA stage
    VX_ks_adder #(
        .N(P),
        .BYPASS(CPA_KS == 0)
    ) final_add (
        .dataa(sum_vec),
        .datab(carry_vec),
        .cin(1'b0),
        .sum(p),
        `UNUSED_PIN(cout)
    );

endmodule
