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

// -------------------------------------------------------------------------
// Recursive Reduction Module
// -------------------------------------------------------------------------

module VX_csa_reducer #(
    parameter N = 1,
    parameter W = 1
) (
    input  wire [N-1:0][W-1:0] operands,
    output wire [W-1:0]        sum_out,
    output wire [W-1:0]        carry_out
);

    // -----------------------------------------------------------------
    // STRATEGY:
    // - Use as many 4:2 compressors as possible (Groups of 4).
    // - If 3 remain, use a 3:2 compressor.
    // - Pass any remaining inputs (1 or 2) directly to the next level.
    // -----------------------------------------------------------------

    localparam NUM_42 = N / 4;
    localparam REM_AFTER_42 = N % 4;

    // We only use a 3:2 if exactly 3 are left. If 1 or 2 left, we pass them.
    localparam USE_32 = (REM_AFTER_42 == 3);

    // Pass-through count
    localparam NUM_PASS = USE_32 ? 0 : REM_AFTER_42;

    // Calculate the number of signals going to the NEXT level
    localparam N_NEXT = (NUM_42 * 2) + (USE_32 * 2) + NUM_PASS;

    // Intermediate wires for the output of this layer
    wire [N_NEXT-1:0][W-1:0] next_operands;

    // 1. Instantiate 4:2 Compressors
    for (genvar i = 0; i < NUM_42; i++) begin : g_csa42
        wire [W-1:0] s, c;
        VX_csa_42 #(
            .N      (W),
            .WIDTH_O(W)
        ) csa42_inst (
            .a      (operands[4*i + 0]),
            .b      (operands[4*i + 1]),
            .c      (operands[4*i + 2]),
            .d      (operands[4*i + 3]),
            .sum    (s),
            .carry  (c)
        );
        assign next_operands[2*i + 0] = s;
        assign next_operands[2*i + 1] = c;
    end

    // 2. Instantiate 3:2 Compressor (if needed)
    if (USE_32) begin : g_csa32
        wire [W-1:0] s, c;
        VX_csa_32 #(
            .N      (W),
            .WIDTH_O(W)
        ) csa32_inst (
            .a      (operands[4*NUM_42 + 0]),
            .b      (operands[4*NUM_42 + 1]),
            .c      (operands[4*NUM_42 + 2]),
            .sum    (s),
            .carry  (c)
        );
        // Map to the next available slots after the 4:2 outputs
        assign next_operands[(2*NUM_42) + 0] = s;
        assign next_operands[(2*NUM_42) + 1] = c;
    end

    // 3. Pass-through remaining signals
    if (NUM_PASS > 0) begin : g_pass
        for (genvar i = 0; i < NUM_PASS; i++) begin : g_loop
            assign next_operands[(2*NUM_42) + (USE_32 ? 2 : 0) + i] = operands[(4*NUM_42) + i];
        end
    end

    // 4. Recursive Step
    if (N_NEXT > 2) begin : g_recurse
        VX_csa_reducer #(
            .N (N_NEXT),
            .W (W)
        ) next_step (
            .operands  (next_operands),
            .sum_out   (sum_out),
            .carry_out (carry_out)
        );
    end else begin : g_done
        assign sum_out   = next_operands[0];
        assign carry_out = next_operands[1];
    end

endmodule

// -------------------------------------------------------------------------
// Top Level Tree
// -------------------------------------------------------------------------

module VX_csa_tree #(
    parameter N = 4,              // Number of operands
    parameter W = 8,              // Bit-width of each operand
    parameter S = W + $clog2(N)   // Output width
) (
    input  wire [N-1:0][W-1:0] operands,
    output wire [S-1:0]        sum,
    output wire                cout
);
    `STATIC_ASSERT(N >= 2, ("N must be at least 2"));

    // 1. Determine Internal Operands Width
    localparam WN = S;

    // 2. Extend Inputs to Internal Width
    wire [N-1:0][WN-1:0] operands_extended;
    for (genvar i = 0; i < N; i++) begin : g_extend
        assign operands_extended[i] = WN'(operands[i]);
    end

    // 3. Instantiate Recursive Reduction Tree
    wire [WN-1:0] final_sum_vec;
    wire [WN-1:0] final_carry_vec;
    VX_csa_reducer #(
        .N (N),
        .W (WN)
    ) tree_core (
        .operands  (operands_extended),
        .sum_out   (final_sum_vec),
        .carry_out (final_carry_vec)
    );

    // 4. Final Carry Propagate Adder
    VX_ks_adder #(
        .N (WN)
    ) KSA (
        .cin   (1'b0),
        .dataa (final_sum_vec),
        .datab (final_carry_vec),
        .sum   (sum),
        .cout  (cout)
    );

endmodule

`TRACING_ON
