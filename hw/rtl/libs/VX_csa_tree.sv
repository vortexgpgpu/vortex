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
    parameter N = 4,              // Number of operands
    parameter W = 8,              // Bit-width of each operand
    parameter S = W + $clog2(N)   // Output width
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

    // Clamp internal width to S.
    localparam MAX_WN = W + TOTAL_LEVELS + 2;
    localparam WN = (S < MAX_WN) ? S : MAX_WN;

    wire [WN-1:0] St [0:TOTAL_LEVELS];
    wire [WN-1:0] Ct [0:TOTAL_LEVELS];

    // Initialize inputs (Extend to WN)
    assign St[0] = WN'(operands[0]);
    assign Ct[0] = WN'(operands[1]);

    // --------------------------------------------------------------
    // 4:2 COMPRESSOR LEVELS
    // --------------------------------------------------------------

    for (genvar i = 0; i < LEVELS_4TO2; i++) begin : g_4to2_levels
        // We set the input width (WI) to match the accumulation of 2 bits/stage.
        localparam _WI = W + i;
        localparam _WO = _WI + 2;
        localparam WI = (_WI > WN) ? WN : _WI;
        localparam WO = (_WO > WN) ? WN : _WO;

        localparam OP_A_IDX = 2 + (i * 2);
        localparam OP_B_IDX = 3 + (i * 2);

        // Cast inputs to WI
        wire [WI-1:0] op_a = WI'(St[i]);
        wire [WI-1:0] op_b = WI'(Ct[i]);
        wire [WI-1:0] op_c = WI'(operands[OP_A_IDX]);
        wire [WI-1:0] op_d = WI'(operands[OP_B_IDX]);

        wire [WO-1:0] st, ct;

        VX_csa_42 #(
            .N       (WI),
            .WIDTH_O (WO)
        ) csa_42 (
            .a     (op_a),
            .b     (op_b),
            .c     (op_c),
            .d     (op_d),
            .sum   (st),
            .carry (ct)
        );

        // Store back in WN-wide container
        assign St[i+1] = WN'(st);
        assign Ct[i+1] = WN'(ct);
    end

    // --------------------------------------------------------------
    // FINAL 3:2 LEVEL
    // --------------------------------------------------------------

    if ((N - LEVELS_4TO2 * 2) == 3) begin : g_final_3to2
        localparam FINAL_OP_IDX = 2 + (LEVELS_4TO2 * 2);

        // Input width must match the accumulated growth (W + Levels*2)
        localparam _WI = W + LEVELS_4TO2;
        localparam _WO = _WI + 2;
        localparam WI = (_WI > WN) ? WN : _WI;
        localparam WO = (_WO > WN) ? WN : _WO;

        // Cast inputs to WI
        wire [WI-1:0] op_a = WI'(St[LEVELS_4TO2]);
        wire [WI-1:0] op_b = WI'(Ct[LEVELS_4TO2]);
        wire [WI-1:0] op_c = operands[FINAL_OP_IDX];

        wire [WO-1:0] st, ct;

        VX_csa_32 #(
            .N       (WI),
            .WIDTH_O (WO)
        ) csa_32 (
            .a     (op_a),
            .b     (op_b),
            .c     (op_c),
            .sum   (st),
            .carry (ct)
        );

        assign St[LEVELS_4TO2+1] = WN'(st);
        assign Ct[LEVELS_4TO2+1] = WN'(ct);
    end
    else begin : g_no_final_3to2
        if (LEVELS_4TO2 < TOTAL_LEVELS) begin : g_pass_through
            assign St[LEVELS_4TO2+1] = St[LEVELS_4TO2];
            assign Ct[LEVELS_4TO2+1] = Ct[LEVELS_4TO2];
        end
    end

    // --------------------------------------------------------------
    // FINAL ADDER
    // --------------------------------------------------------------

    wire [WN-1:0] raw_sum;

    VX_ks_adder #(
        .N (WN)
    ) KSA (
        .cin   (0),
        .dataa (St[TOTAL_LEVELS]),
        .datab (Ct[TOTAL_LEVELS]),
        .sum   (raw_sum),
        .cout  (cout)
    );

    // Final Output Expansion
    assign sum = S'(raw_sum);

endmodule

`TRACING_ON
