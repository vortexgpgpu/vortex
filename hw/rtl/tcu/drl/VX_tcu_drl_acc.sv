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

module VX_tcu_drl_acc import VX_tcu_pkg::*; #(
    parameter `STRING INSTANCE_ID = "",
    parameter N  = 5,
    parameter WI = 26,
    parameter WO = 30
) (
    input  wire                 clk,
    input  wire                 valid_in,
    input  wire [31:0]          req_id,
    input  wire [N-2:0]         lane_mask,
    input  wire [N-1:0][WI-1:0] sigs_in,
    input  wire [N-1:0]         sticky_in,
    output wire [WO-1:0]        sig_out,
    output wire                 sticky_out
);
    `UNUSED_SPARAM (INSTANCE_ID)
    `UNUSED_VAR ({clk, valid_in, req_id})

    // ----------------------------------------------------------------------
    // Input Masking
    // ----------------------------------------------------------------------
    wire [N-1:0][WI-1:0] masked_sigs;
    wire [N-1:0]        masked_sticky;

    // Mask vector lanes (0 to N-2)
    for (genvar i = 0; i < N-1; ++i) begin : g_mask
        assign masked_sigs[i]   = sigs_in[i] & {WI{lane_mask[i]}};
        assign masked_sticky[i] = sticky_in[i] & lane_mask[i];
    end

    // Pass C-term (N-1) unmasked
    assign masked_sigs[N-1]   = sigs_in[N-1];
    assign masked_sticky[N-1] = sticky_in[N-1];

    // ----------------------------------------------------------------------
    // Sign Extension
    // ----------------------------------------------------------------------

    wire [N-1:0][WO-1:0] sigs_in_packed;
    for (genvar i = 0; i < N; ++i) begin : g_ext
        assign sigs_in_packed[i] = $signed({{(WO-WI){masked_sigs[i][WI-1]}}, masked_sigs[i]});
    end

    // ----------------------------------------------------------------------
    // Fast Accumulation (CSA Tree)
    // ----------------------------------------------------------------------

    VX_csa_tree #(
        .N (N),
        .W (WO),
        .S (WO),
        .CPA_KS (!`FORCE_BUILTIN_ADDER(WO))
    ) sig_csa (
        .operands (sigs_in_packed),
        .sum      (sig_out),
        `UNUSED_PIN(cout)
    );

    // ----------------------------------------------------------------------
    // Sticky aggregation
    // ----------------------------------------------------------------------

    assign sticky_out = |masked_sticky;

    // ----------------------------------------------------------------------
    // Debug
    // ----------------------------------------------------------------------

`ifdef DBG_TRACE_TCU
    always_ff @(posedge clk) begin
        if (valid_in) begin
            `TRACE(4, ("%t: %s FEDP-ACC(%0d): lane_mask=%b, sigs_in=", $time, INSTANCE_ID, req_id, lane_mask));
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
