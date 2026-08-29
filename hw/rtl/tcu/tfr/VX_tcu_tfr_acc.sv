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

// Reduces the pre-conditioned accumulator operands (masked, negated and
// sign-extended by the align stage). The +1 completions of the fp
// ones-complement negations enter the tree as a single popcount operand.
module VX_tcu_tfr_acc import VX_tcu_pkg::*; #(
    parameter `STRING INSTANCE_ID = "",
    parameter N  = 5,
    parameter WO = 30
) (
    input  wire                 clk,
    input  wire                 valid_in,
    input  wire [31:0]          req_id,
    input  wire [N-1:0][WO-1:0] sigs_in,
    input  wire [N-1:0]         sticky_in,
    input  wire [N-1:0]         fp_negs,
    output wire [WO-1:0]        sig_out,   // Sum magnitude
    output wire                 sign_out,  // Sum sign
    output wire                 sticky_out
);
    `UNUSED_SPARAM (INSTANCE_ID)
    `UNUSED_VAR ({clk, valid_in, req_id})

    // Count number of negative FP terms for final correction
    wire [`CLOG2(N+1)-1:0] fp_neg_count;
    VX_popcount #(
        .N (N)
    ) fp_neg_popcount (
        .data_in  (fp_negs),
        .data_out (fp_neg_count)
    );

    wire [N:0][WO-1:0] sig_operands;
    for (genvar i = 0; i < N; ++i) begin : g_ops
        assign sig_operands[i] = sigs_in[i];
    end
    assign sig_operands[N] = WO'(fp_neg_count);

    // Fast Reduction (CSA Tree)
    wire [WO-1:0] sum_vec;
    wire [WO-1:0] carry_vec;

    VX_csa_tree #(
        .N (N+1),
        .W (WO),
        .S (WO)
    ) sig_csa (
        .operands (sig_operands),
        .sum      (sum_vec),
        .carry    (carry_vec)
    );

    // Resolve the sum in sign-magnitude form with two parallel adders, so the
    // norm stage can run its LZC directly on the exact magnitude.
    wire [WO-1:0] sig_pos;
    VX_ks_adder #(
        .N (WO),
        .BYPASS (`FORCE_BUILTIN_ADDER(WO))
    ) final_adder (
        .cin   (1'b0),
        .dataa (sum_vec),
        .datab (carry_vec),
        .sum   (sig_pos),
        `UNUSED_PIN (cout)
    );

    // -(a+b) = ~a + ~b + 2: cin=1 yields ~(a+b), the +1 completes the negate.
    wire [WO-1:0] sig_neg_raw;
    VX_ks_adder #(
        .N (WO),
        .BYPASS (`FORCE_BUILTIN_ADDER(WO))
    ) neg_adder (
        .cin   (1'b1),
        .dataa (~sum_vec),
        .datab (~carry_vec),
        .sum   (sig_neg_raw),
        `UNUSED_PIN (cout)
    );
    wire [WO-1:0] sig_neg = sig_neg_raw + WO'(1);

    assign sign_out = sig_pos[WO-1];
    assign sig_out  = sign_out ? sig_neg : sig_pos;

    // Sticky bit aggregation
    assign sticky_out = |sticky_in;

`ifdef DBG_TRACE_TCU
    always_ff @(posedge clk) begin
        if (valid_in) begin
            `TRACE(4, ("%t: %s FEDP-ACC(%0d): sigs_in=", $time, INSTANCE_ID, req_id));
            `TRACE_ARRAY1D(4, "0x%0h", sigs_in, N)
            `TRACE(4, (", sticky_in="));
            `TRACE_ARRAY1D(4, "%0d", sticky_in, N)
            `TRACE(4, (", fp_negs="));
            `TRACE_ARRAY1D(4, "%0d", fp_negs, N)
            `TRACE(4, (", sig_out=0x%0h, sign_out=%0d, sticky_out=%0d\n", sig_out, sign_out, sticky_out));
        end
    end
`endif

endmodule
