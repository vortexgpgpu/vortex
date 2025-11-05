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

module VX_tcu_drl_max_exp #(
    parameter N = 5     //include c_val count
) (
    input wire [N-1:0][7:0] exponents,
    output logic [7:0] max_exp,
    output logic [N-1:0][7:0] shift_amounts
);

    //Subtractor-based compare exponent tree max finder
    //Generate exponent subtract sign matrix and store differences
    wire [N-1:0] sign_mat[N-1:0];
    wire signed [8:0] diff_mat[N-1:0][N-1:0];
    for (genvar i = 0; i < N; i++) begin : g_row
        for (genvar j = 0; j < N; j++) begin : g_col
            if (i == j) begin : g_diag
                assign sign_mat[i][j] = 1'b0;
                assign diff_mat[i][j] = 9'd0;
            end else begin : g_comp
                assign diff_mat[i][j] = {1'b0, exponents[i]} - {1'b0, exponents[j]};
                assign sign_mat[i][j] = diff_mat[i][j][8];
            end
        end
    end

    //Finding max exp one-hot encoded index
    wire [N-1:0] sel_exp;
    for (genvar i = 0; i < N; i++) begin : g_index
        wire [N-1:0] col_i;
        for (genvar j = 0; j < N; j++) begin : g_col_extract
            assign col_i[j] = sign_mat[j][i];
        end
        assign sel_exp[i] = ~(|col_i);    //Column with all zeros is max
    end

    //Reduction OR (Explicit MUX)
`IGNORE_UNOPTFLAT_BEGIN
    wire [7:0] or_red[N:0];
`IGNORE_UNOPTFLAT_END
    assign or_red[0] = 8'd0;
    for (genvar i = 0; i < N; i++) begin : g_or_red
        assign or_red[i+1] = or_red[i] | (sel_exp[i] ? exponents[i] : 8'd0);
    end
    assign max_exp = or_red[N];

    //Reusing shift amounts directly from difference matrix
    for (genvar i = 0; i < N; i++) begin : g_shift_extract

`IGNORE_UNOPTFLAT_BEGIN
        wire [7:0] shift_op[N:0];
`IGNORE_UNOPTFLAT_END

        assign shift_op[0] = 8'd0;
        for (genvar j = 0; j < N; j++) begin : g_shift_mux
            //For case operand j is max
            wire [7:0] shift_sel = sel_exp[j] ? (-diff_mat[i][j][7:0]) : 8'd0;
            assign shift_op[j+1] = shift_op[j] | shift_sel;
        end

        assign shift_amounts[i] = shift_op[N];
    end

endmodule

