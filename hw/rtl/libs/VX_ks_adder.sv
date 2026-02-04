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
    parameter N = 16
) (
    input  wire [N-1:0] dataa,
    input  wire [N-1:0] datab,
    input  wire         cin,
    output wire [N-1:0] sum,
    output wire         cout
);
    localparam LEVELS = $clog2(N);

`IGNORE_UNOPTFLAT_BEGIN
    wire [N-1:0] G [LEVELS+1];
    wire [N-1:0] P [LEVELS+1];
`IGNORE_UNOPTFLAT_END

    // Initial generate & propagate
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

    // Final sum with carry-in
    assign sum[0] = P[0][0] ^ cin;
    for (genvar i = 1; i < N; i++) begin : g_sum
        wire carry_in_i = G[LEVELS][i-1] | (P[LEVELS][i-1] & cin);
        assign sum[i] = P[0][i] ^ carry_in_i;
    end

    // Carry Out
    assign cout = G[LEVELS][N-1] | (P[LEVELS][N-1] & cin);

endmodule

`TRACING_ON
