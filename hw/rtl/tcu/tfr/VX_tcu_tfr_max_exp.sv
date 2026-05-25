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

module VX_tcu_tfr_max_exp import VX_tcu_pkg::*; #(
    parameter N     = 5,
    parameter WIDTH = 8
) (
    input  wire [N-1:0][WIDTH-1:0] exponents,
    output logic [WIDTH-1:0]       max_exp,
    output wire [N-1:0][7:0]       shift_amts
);

    // Signed subtractor matrix.
    wire [N-2:0] sign_mat[N-2:0] /* verilator split_var */;
    wire signed [WIDTH:0] diff_mat[N-2:0][N-2:0];

    for (genvar i = 0; i < N-1; i++) begin : g_row
        for (genvar j = 0; j < N-1; j++) begin : g_col
            if (j < i) begin : g_lower
                assign sign_mat[i][j] = ~sign_mat[j][i];
            end else begin : g_upper
                assign diff_mat[i][j] = {1'b0, exponents[i]} - {1'b0, exponents[j+1]};
                assign sign_mat[i][j] = diff_mat[i][j][WIDTH];
            end
        end
    end

    // One-hot max exponent select.
    wire [N-1:0] sel_exp;

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

    // Parallel max exponent mux.
    wire [WIDTH-1:0] or_red[N:0] /* verilator split_var */;

    assign or_red[0] = {WIDTH{1'b0}};
    for (genvar i = 0; i < N; i++) begin : g_or_red
        assign or_red[i+1] = or_red[i] | (sel_exp[i] ? exponents[i] : {WIDTH{1'b0}});
    end
    assign max_exp = or_red[N];

    // Reuse the comparison subtractors to produce max_exp - exponents[i]
    for (genvar i = 0; i < N; i++) begin : g_shift
        wire [WIDTH-1:0] sh_or [N:0] /* verilator split_var */;

        assign sh_or[0] = {WIDTH{1'b0}};
        for (genvar k = 0; k < N; k++) begin : g_sh_mux
            if (k == i) begin : g_self
                assign sh_or[k+1] = sh_or[k];
            end else if (k < i) begin : g_direct
                wire sel_k = sel_exp[k];
                wire [WIDTH-1:0] diff_lane = diff_mat[k][i-1][WIDTH-1:0];
                assign sh_or[k+1] = sh_or[k] | (sel_k ? diff_lane : {WIDTH{1'b0}});
            end else begin : g_invert
                wire sel_k = sel_exp[k];
                wire [WIDTH-1:0] diff_lane = diff_mat[i][k-1][WIDTH-1:0];
                assign sh_or[k+1] = sh_or[k] | (sel_k ? ~diff_lane : {WIDTH{1'b0}});
            end
        end

        wire needs_inc;
        if (i == N-1) begin : g_no_inc
            assign needs_inc = 1'b0;
        end else begin : g_calc_inc
            wire [N-2-i:0] inc_sel;
            for (genvar k = i+1; k < N; k++) begin : g_inc_sel
                wire sel_k = sel_exp[k];
                assign inc_sel[k-i-1] = sel_k;
            end
            assign needs_inc = |inc_sel;
        end

        wire [WIDTH-1:0] shift_full = sh_or[N] + WIDTH'(needs_inc);
        if (WIDTH > 8) begin : g_sat
            assign shift_amts[i] = (|shift_full[WIDTH-1:8]) ? 8'hFF : shift_full[7:0];
        end else begin : g_no_sat
            assign shift_amts[i] = 8'(shift_full);
        end
    end

endmodule
