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

module VX_tcu_drl_acc #(
    parameter N = 4
) (
    input wire [31:0] fp32operands[N-1:0],
    output logic signOut,
    output logic [7:0] expOut,
    output logic [24+$clog2(N)-1:0] sigOut
);
    localparam TREE_LEVELS = $clog2(N);

    //Extracting FP32 input operands' fields
    wire [N-1:0] op_sign;
    wire [7:0] op_exp[N-1:0];
    wire [23:0] op_sig[N-1:0];  //includes implied hidden 1

    for (genvar i = 0; i < N; i++) begin : g_field_extract
        assign op_sign[i] = fp32operands[i][31];
        assign op_exp[i] = fp32operands[i][30:23];
        assign op_sig[i] = {1'b1, fp32operands[i][22:0]};
    end

    //Subtractor-based exponent compare tree max finder
    //Generate exponent subtract sign matrix AND store differences
    wire [N-1:0] sign_mat[N-1:0];
    wire signed [8:0] diff_mat[N-1:0][N-1:0];
    for (genvar i = 0; i < N; i = i + 1) begin : g_row
        for (genvar j = 0; j < N; j = j + 1) begin : g_col
            if (i == j) begin : g_diag
                assign sign_mat[i][j] = 1'b0;
                assign diff_mat[i][j] = 9'd0;
            end else begin : g_comp
                assign diff_mat[i][j] = {1'b0, op_exp[i]} - {1'b0, op_exp[j]};
                assign sign_mat[i][j] = diff_mat[i][j][8];
            end
        end
    end

    //Finding Max exp one-hot encoded index
    wire [N-1:0] sel_exp;
    for (genvar i = 0; i < N; i = i + 1) begin : g_sel
        wire and_left, or_right;

        if (i == 0) begin : g_first
            assign and_left = 1'b1;
        end else begin : g_and_left
            wire [i-1:0] left_signals;
            for (genvar jl = 0; jl < i; jl = jl + 1) begin : g_left
               assign left_signals[jl] = sign_mat[jl][i];
            end
            assign and_left = &left_signals;
        end

        if (i == N-1) begin : g_last
            assign or_right = 1'b0;
        end else begin : g_or_right
            wire [N-2-i:0] right_signals;
            for (genvar jr = i+1; jr < N; jr = jr + 1) begin : g_right
                assign right_signals[jr-i-1] = sign_mat[i][jr];
            end
            assign or_right = |right_signals;
        end

        assign sel_exp[i] = and_left & (~or_right);
    end

    //Reduction OR (Explicit MUX)
    wire [7:0] or_red[N:0];
    assign or_red[0] = 8'd0;
    for (genvar i = 0; i < N; i++) begin : g_or_red
        assign or_red[i+1] = or_red[i] | (sel_exp[i] ? op_exp[i] : 8'd0);
    end
    wire [7:0] max_exp = or_red[N];

    //Reusing shift amounts directly from difference matrix
    wire [7:0] shift_amounts[N-1:0];
    for (genvar i = 0; i < N; i++) begin : g_shift_extract
        wire [7:0] shift_op[N:0];
        assign shift_op[0] = 8'd0;

        for (genvar j = 0; j < N; j++) begin : g_shift_mux
            //For case operand j is max
            wire [7:0] shift_sel = sel_exp[j] ? (-diff_mat[i][j][7:0]) : 8'd0;
            assign shift_op[j+1] = shift_op[j] | shift_sel;
        end

        assign shift_amounts[i] = shift_op[N];
    end

    //Aligned + signed significands
    wire signed [24:0] signed_sig[N-1:0];
    for (genvar i = 0; i < N; i++) begin : g_align_signed
        wire [23:0] adj_sig = op_sig[i] >> shift_amounts[i];
        assign signed_sig[i] = op_sign[i] ? -adj_sig : adj_sig;
    end

    //Carry-Save-Adder based significand accumulation
    wire [25+$clog2(N)-1:0] signed_sum_sig;
    VX_csa_tree #(
        .N (N),
        .W (25)
    ) sig_csa (
        .operands (signed_sig),
        .sum (signed_sum_sig)
    );

    //Extracting magnitude from signed result
    wire sum_sign = signed_sum_sig[25+$clog2(N)-1];
    wire [24+$clog2(N)-1:0] abs_sum;
    assign abs_sum = sum_sign ? -signed_sum_sig[24+$clog2(N)-1:0] : signed_sum_sig[24+$clog2(N)-1:0];

    assign signOut = sum_sign;
    assign expOut = max_exp;
    assign sigOut = abs_sum;
endmodule

/*
        wire [23:0] adj_sig = shift_amount[3] ? 24'd0 : op_sig[i] >> shift_amount;      //reducing switching activity (power) by clamping to 0 if
                                                                                        //input won't make a significant impact on accumulated value
*/

/*
    //Low area exponential fmax dropoff exp max finder version

    // Subtractor-based exponent compare tree max finder
    wire [7:0] tree_exp [0:TREE_LEVELS][N-1:0];
    for (genvar i = 0; i < N; i++) begin : g_init_exp_tree
        assign tree_exp[0][i] = op_exp[i];
    end
    for (genvar lvl = 0; lvl < TREE_LEVELS; lvl++) begin : g_exp_tree_levels
        localparam integer CURSZ = N >> lvl;
        localparam integer OUTSZ = (CURSZ >> 1);
        for (genvar i = 0; i < OUTSZ; i++) begin : g_exp_tree_cmp
            wire [7:0] expA = tree_exp[lvl][2*i];
            wire [7:0] expB = tree_exp[lvl][2*i+1];
            wire signed [8:0] diff = {1'b0, expA} - {1'b0, expB};
            assign tree_exp[lvl+1][i] = diff[8] ? expB : expA;
        end
        // Pass-through odd one (if any)
        if (CURSZ % 2 == 1) begin : g_exp_odd_handle
            assign tree_exp[lvl+1][OUTSZ] = tree_exp[lvl][CURSZ-1];
        end
    end
    assign max_exp = tree_exp[TREE_LEVELS][0];
*/

/*
    //Alternate orignial significand accumulation version
    //Signed Tree-based reduction adder
    //Each level has wider bit-widths than required to prevent overflow
    wire signed [25+TREE_LEVELS-1:0] tree_sum[0:TREE_LEVELS] [N-1:0];
    for (genvar i = 0; i < N; i++) begin : g_init_sum_tree
        assign tree_sum[0][i] = {{TREE_LEVELS{signed_sig[i][25-1]}}, signed_sig[i]};
    end
    for (genvar lvl = 0; lvl < TREE_LEVELS; lvl++) begin : g_sum_tree_levels
        localparam integer CURSZ = N >> lvl;
        localparam integer OUTSZ = CURSZ >> 1;

        for (genvar i = 0; i < OUTSZ; i++) begin : g_sum_tree_add
            //add signed pairs of operands at current level
            assign tree_sum[lvl+1][i] = tree_sum[lvl][2*i+0] + tree_sum[lvl][2*i+1];
        end

        //odd number operand handling - carry forward to next stage
        if (CURSZ % 2 == 1) begin : g_sum_odd_handle
            assign tree_sum[lvl+1][OUTSZ] = tree_sum[lvl][CURSZ-1];
        end
    end
    //Signed result of summation
    wire signed [25+$clog2(N)-1:0] signed_sum_sig;
    assign signed_sum_sig = tree_sum[TREE_LEVELS][0];
*/
