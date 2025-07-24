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

module VX_tcu_drl_acc_red_tree #(
    parameter N = 4
) (
    input wire [31:0] fp32operands[N-1:0],
    output logic signOut,
    output logic [7:0] expOut,
    output logic [24+$clog2(N)-1:0] sigOut
);
    localparam TREE_LEVELS = $clog2(N);

    wire [N-1:0] op_sign;
    wire [7:0] op_exp[N-1:0];
    wire [23:0] op_sig[N-1:0];  //includes implied hidden 1

    for (genvar i = 0; i < N; i++) begin : g_field_extract
        assign op_sign[i] = fp32operands[i][31];
        assign op_exp[i] = fp32operands[i][30:23];
        assign op_sig[i] = {1'b1, fp32operands[i][22:0]};
    end

    //Low area exponential fmax dropoff with increase in N exp max finder version
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

    //Aligning significands based on max_exp
    wire [23:0] adj_sig[N-1:0];
    for (genvar i = 0; i < N; i++) begin : g_sig_adjust
        assign adj_sig[i] = op_sig[i] >> (max_exp - op_exp[i]);
    end

    // NOTE: could add pipeline register here? (will be pretty expensive)

    //Converting significands to signed for direct result summation sign
    wire signed [24:0] signed_sig[N-1:0];
    for (genvar i = 0; i < N; i++) begin : g_signed_conv
        assign signed_sig[i] = op_sign[i] ? -adj_sig[i] : adj_sig[i];
    end

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

    //Extracting magnitude from signed result
    wire sum_sign = signed_sum_sig[25+$clog2(N)-1];
    wire [24+$clog2(N)-1:0] abs_sum;
    assign abs_sum = sum_sign ? -signed_sum_sig[24+$clog2(N)-1:0] : signed_sum_sig[24+$clog2(N)-1:0];

    assign signOut = sum_sign;
    assign expOut = max_exp;
    assign sigOut = abs_sum;
endmodule
