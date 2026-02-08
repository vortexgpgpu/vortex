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

`TRACING_OFF

// Carry-Save Adder Tree (4:2 based, falls back to 3:2 if needed)
module VX_csa_tree #(
    parameter N = 4,  // Number of operands
    parameter W = 8,  // Bit-width of each operand
    parameter S = W + $clog2(N),  // Output width
    parameter CPA_KS = 1          // Use Kogge-Stone CPA
) (
    input  wire [N-1:0][W-1:0] operands,  // Input operands
    output wire [S-1:0] sum,  // Final sum output
    output wire cout
);
    `STATIC_ASSERT (N >= 3, ("N must be at least 3"));

    function automatic integer calc_4to2_levels(integer n);
        integer remaining = n;
        integer levels_4to2 = 0;
        while (remaining >= 4) begin
            levels_4to2 = levels_4to2 + 1;
            remaining = remaining - 2;    // 4->2 reduction
        end
        return levels_4to2;
    endfunction

    localparam LEVELS_4TO2 = calc_4to2_levels(N);
    localparam TOTAL_LEVELS = LEVELS_4TO2 + ((N - LEVELS_4TO2 * 2) == 3 ? 1 : 0);
    localparam WN = W + TOTAL_LEVELS + 2;

    // Intermediate signals
    wire [WN-1:0] St [0:TOTAL_LEVELS];
    wire [WN-1:0] Ct [0:TOTAL_LEVELS];

    // Initialize first two operands
    assign St[0] = WN'(operands[0]);
    assign Ct[0] = WN'(operands[1]);

    // Generate 4:2 compressor levels first
    for (genvar i = 0; i < LEVELS_4TO2; i++) begin : g_4to2_levels
        localparam WI = W + i;
        localparam WO = WI + 2;
        localparam OP_A_IDX = 2 + (i * 2);
        localparam OP_B_IDX = 3 + (i * 2);

        wire [WO-1:0] st, ct;
        VX_csa_42 #(
            .N       (WI),
            .WIDTH_O (WO)
        ) csa_42 (
            .a     (WI'(St[i])),
            .b     (WI'(Ct[i])),
            .c     (WI'(operands[OP_A_IDX])),
            .d     (WI'(operands[OP_B_IDX])),
            .sum   (st),
            .carry (ct)
        );
        assign St[i+1] = WN'(st);
        assign Ct[i+1] = WN'(ct);
    end

    // If final 3:2 compressor level is needed
    if ((N - LEVELS_4TO2 * 2) == 3) begin : g_final_3to2
        // exactly 3 operands left
        localparam FINAL_OP_IDX = 2 + (LEVELS_4TO2 * 2);
        localparam WI = W + LEVELS_4TO2;
        localparam WO = WI + 2;

        wire [WO-1:0] st, ct;
        VX_csa_32 #(
            .N       (WI),
            .WIDTH_O (WO)
        ) csa_32 (
            .a     (WI'(St[LEVELS_4TO2])),
            .b     (WI'(Ct[LEVELS_4TO2])),
            .c     (WI'(operands[FINAL_OP_IDX])),
            .sum   (st),
            .carry (ct)
        );
        assign St[LEVELS_4TO2+1] = WN'(st);
        assign Ct[LEVELS_4TO2+1] = WN'(ct);
    end
    else begin : g_no_final_3to2
        // exactly 2 operands left
        if (LEVELS_4TO2 < TOTAL_LEVELS) begin : g_pass_through
            assign St[LEVELS_4TO2+1] = St[LEVELS_4TO2];
            assign Ct[LEVELS_4TO2+1] = Ct[LEVELS_4TO2];
        end
    end

    // Final Kogge-Stone addition
    VX_ks_adder #(
        .N (S),
        .BYPASS (!CPA_KS)
    ) KSA (
        .cin(0),
        .dataa(St[TOTAL_LEVELS][S-1:0]),
        .datab(Ct[TOTAL_LEVELS][S-1:0]),
        .sum(sum),
        .cout(cout)
    );
endmodule

`TRACING_ON
