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

module VX_tcu_drl_align #(
    parameter N = 5    //includes c_val
) (
    input wire [N-1:0] signsIn,
    input wire [N-1:0][7:0] expsIn,
    input wire [7:0] rawMaxExpIn,
    input wire [N-1:0][23:0] sigsIn,    //includes hidden 1
    output logic [N-1:0][24:0] sigsOut
);

    //Find Shift amount, Align & Sign Convert significands
    for (genvar i = 0; i < N; i++) begin : g_align_signed
        wire [7:0] shift_amount = rawMaxExpIn - expsIn[i];
        wire [23:0] adj_sig = sigsIn[i] >> shift_amount;
        assign sigsOut[i] = signsIn[i] ? -adj_sig : {1'b0, adj_sig};
    end

endmodule

/*
    //Final shift amounts reusing diff matrix and adding norm_shift differences
    for (genvar i = 0; i < N; i++) begin : g_final_shift
`IGNORE_UNOPTFLAT_BEGIN
        wire [7:0] shift_op[N:0];
`IGNORE_UNOPTFLAT_END

        assign shift_op[0] = 8'd0;
        for (genvar j = 0; j < N; j++) begin : g_shift_mux
            //norm_shift values for adjustment
            wire norm_shift_j, norm_shift_i;
            
            if (j == N-1) begin : g_j_c_val
                assign norm_shift_j = 1'b0;
            end else begin : g_j_mult
                assign norm_shift_j = fmt ? bf16_norm_shifts[j] : fp16_norm_shifts[j];
            end
            
            if (i == N-1) begin : g_i_c_val
                assign norm_shift_i = 1'b0;
            end else begin : g_i_mult
                assign norm_shift_i = fmt ? bf16_norm_shifts[i] : fp16_norm_shifts[i];
            end
            
            // Adjust the base difference with normalization deltas
            wire norm_delta = norm_shift_j - norm_shift_i;
            wire [7:0] adjusted_diff = (-diff_mat[i][j][7:0]) + (8'norm_delta);
            wire [7:0] shift_sel = sel_exp[j] ? adjusted_diff : 8'd0;
            assign shift_op[j+1] = shift_op[j] | shift_sel;
        end

        assign shift_amounts[i] = shift_op[N];
    end


        wire [23:0] adj_sig = shift_amount[3] ? 24'd0 : full_sig[i] >> shift_amount;      //reducing switching activity (power) by clamping to 0 if
                                                                                        //input won't make a significant impact on accumulated value
*/
