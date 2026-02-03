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

`include "VX_platform.vh"

`TRACING_OFF

// Fast Kogge-Stone adder

module VX_ks_adder #(
    parameter N = 16,    // inputs width
    parameter SIGNED = 0 // 0 = Unsigned, 1 = Signed
) (
    input  wire [N-1:0] dataa,
    input  wire [N-1:0] datab,
    output wire [N-1:0] sum,
    output wire         cout // Overflow flag (if SIGNED=1) or Carry out
);
    localparam LEVELS = $clog2(N);

`IGNORE_UNOPTFLAT_BEGIN
    wire [N-1:0] G [LEVELS+1];
    wire [N-1:0] P [LEVELS+1];
`IGNORE_UNOPTFLAT_END

    // level 0: initial generate & propagate
    for (genvar i = 0; i < N; i++) begin : g_initial_gp
        assign G[0][i] = dataa[i] & datab[i];
        assign P[0][i] = dataa[i] ^ datab[i];
    end

    // Kogge-Stone tree levels
    for (genvar k = 1; k <= LEVELS; k++) begin : g_ks_levels
        localparam STEP = 1 << (k - 1);
        for (genvar i = 0; i < N; i++) begin : g_ks_nodes
            if (i >= STEP) begin : g_compute_gp
                assign G[k][i] = G[k-1][i] | (P[k-1][i] & G[k-1][i-STEP]);
                assign P[k][i] = P[k-1][i] & P[k-1][i-STEP];
            end else begin : g_passthrough_gp
                assign G[k][i] = G[k-1][i];
                assign P[k][i] = P[k-1][i];
            end
        end
    end

    // final sum bits
    assign sum[0] = P[0][0];
    for (genvar i = 1; i < N; i++) begin : g_sum
        assign sum[i] = P[0][i] ^ G[LEVELS][i-1];
    end

    if (SIGNED) begin : g_signed_logic
        if (N > 1) begin : g_ovf
            // Signed Overflow = Carry_Out_MSB XOR Carry_In_MSB
            assign cout = G[LEVELS][N-1] ^ G[LEVELS][N-2];
        end else begin : g_ovf_1bit
            // For 1-bit signed, Overflow = Carry_Out ^ 0
            assign cout = G[LEVELS][0];
        end
    end else begin : g_unsigned_logic
        // Unsigned Carry Out
        assign cout = G[LEVELS][N-1];
    end

endmodule

`TRACING_ON
