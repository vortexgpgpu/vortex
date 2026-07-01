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

module VX_tcu_tet_max_exp import VX_tcu_pkg::*; #(
    parameter N     = 5,
    parameter WIDTH = 8
) (
    input  wire [N-1:0][WIDTH-1:0] exponents,
    output wire [N-1:0]            sel_exp,
    output wire [N-2:0][N-2:0][WIDTH:0] diff_mat
);

    // Signed subtractor matrix.
    wire [N-2:0] sign_mat[N-2:0] /* verilator split_var */;

    for (genvar i = 0; i < N-1; i++) begin : g_row
        for (genvar j = 0; j < N-1; j++) begin : g_col
            if (j < i) begin : g_lower
                assign sign_mat[i][j] = ~sign_mat[j][i];
                assign diff_mat[i][j] = '0;
            end else begin : g_upper
                assign diff_mat[i][j] = {1'b0, exponents[i]} - {1'b0, exponents[j+1]};
                assign sign_mat[i][j] = diff_mat[i][j][WIDTH];
            end
        end
    end

    // Find maximum exponent index based on the sign matrix
    for (genvar i = 0; i < N; i++) begin : g_index
        wire and_left, or_right;
        if (i == 0) begin : g_first
            assign and_left = 1'b1;
        end else begin : g_and_left
            wire [i-1:0] left_signals;
            for (genvar jl = 0; jl < i; jl++) begin : g_left
                assign left_signals[jl] = sign_mat[jl][i-1];
            end
            assign and_left = &left_signals;
        end

        if (i == N-1) begin : g_last
            assign or_right = 1'b0;
        end else begin : g_or_right
            wire [N-2-i:0] right_signals;
            for (genvar jr = i+1; jr < N; jr++) begin : g_right
                assign right_signals[jr-i-1] = sign_mat[i][jr-1];
            end
            assign or_right = |right_signals;
        end

        assign sel_exp[i] = and_left & (~or_right);
    end

endmodule
