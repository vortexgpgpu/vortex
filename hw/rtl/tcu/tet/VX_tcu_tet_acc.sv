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

module VX_tcu_tet_acc import VX_tcu_pkg::*; #(
    parameter `STRING INSTANCE_ID = "",
    parameter N  = 5,
    parameter WI = 26,
    parameter WO = 30
) (
    input  wire                 clk,
    input  wire                 reset,
    input  wire                 enable,
    input  wire                 valid_in,
    input  wire [31:0]          req_id,
    input  wire [N-2:0]         lane_mask,
    input  wire                 is_int,
    input  wire [N-1:0][WI-1:0] sigs_in,
    input  wire [N-1:0]         sticky_in,
    output wire [WO-1:0]        sig_out,
    output wire                 sticky_out
);
    `UNUSED_SPARAM (INSTANCE_ID)

    wire [N-1:0][WI-1:0] masked_sigs;
    wire [N-1:0]         masked_sticky;
    for (genvar i = 0; i < N-1; ++i) begin : g_mask
        assign masked_sigs[i]   = sigs_in[i] & {WI{lane_mask[i]}};
        assign masked_sticky[i] = sticky_in[i] & lane_mask[i];
    end

    assign masked_sigs[N-1]   = sigs_in[N-1];
    assign masked_sticky[N-1] = sticky_in[N-1];

    wire [N-1:0] fp_neg_terms;
    wire [N:0][WO-1:0] sig_operands;

    for (genvar i = 0; i < N; ++i) begin : g_ext
        wire [WO-1:0] int_sig = $signed({{(WO-WI){masked_sigs[i][WI-1]}}, masked_sigs[i]});
        wire [WO-1:0] fp_mag  = WO'(masked_sigs[i][WI-2:0]);
        assign fp_neg_terms[i] = ~is_int & masked_sigs[i][WI-1];
        assign sig_operands[i] = is_int ? int_sig : (fp_neg_terms[i] ? ~fp_mag : fp_mag);
    end

    wire [`CLOG2(N+1)-1:0] fp_neg_count;
    VX_popcount #(
        .N (N)
    ) fp_neg_popcount (
        .data_in  (fp_neg_terms),
        .data_out (fp_neg_count)
    );

    assign sig_operands[N] = WO'(fp_neg_count);

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

    wire sticky_w = |masked_sticky;

    wire [WO-1:0] s1_sum_vec;
    wire [WO-1:0] s1_carry_vec;
    wire          s1_sticky;
    wire          s1_valid;
    wire [31:0]   s1_req_id;

    VX_tcu_tet_register #(
        .DATAW ((2 * WO) + 1 + 1 + 32),
        .DEPTH (1)
    ) pipe_acc0 (
        .clk      (clk),
        .reset    (reset),
        .enable   (enable),
        .data_in  ({sum_vec,    carry_vec,    sticky_w,   valid_in,  req_id}),
        .data_out ({s1_sum_vec, s1_carry_vec, s1_sticky, s1_valid, s1_req_id})
    );

`ifndef DBG_TRACE_TCU
    `UNUSED_VAR ({s1_valid, s1_req_id})
`endif

    VX_ks_adder #(
        .N (WO),
        .BYPASS (`FORCE_BUILTIN_ADDER(WO))
    ) final_adder (
        .cin   (1'b0),
        .dataa (s1_sum_vec),
        .datab (s1_carry_vec),
        .sum   (sig_out),
        `UNUSED_PIN (cout)
    );

    assign sticky_out = s1_sticky;

`ifdef DBG_TRACE_TCU
    always_ff @(posedge clk) begin
        if (s1_valid) begin
            `TRACE(4, ("%t: %s FEDP-ACC(%0d): lane_mask=%b, sigs_in=", $time, INSTANCE_ID, s1_req_id, lane_mask));
            `TRACE_ARRAY1D(4, "0x%0h", sigs_in, N)
            `TRACE(4, (", masked_sigs="));
            `TRACE_ARRAY1D(4, "0x%0h", masked_sigs, N)
            `TRACE(4, (", sticky_in="));
            `TRACE_ARRAY1D(4, "%0d", sticky_in, N)
            `TRACE(4, (", sig_out=0x%0h, sticky_out=%0d\n", sig_out, sticky_out));
        end
    end
`endif

endmodule
