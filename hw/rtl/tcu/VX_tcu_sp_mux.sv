// Copyright 2019-2023
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

`ifdef VX_CFG_TCU_SPARSE_ENABLE

// 2:4 structured sparsity gather mux of one (i, j) FEDP cell.
// Output nibble p of B word k takes one of six fixed nibbles of the two
// candidate words b_col_in1[k] and b_col_in2[k] (the union over the element
// widths); VX_tcu_sp_sel decodes which, one-hot, once for the FEDP's row,
// from the metadata and fmt_s.

module VX_tcu_sp_mux import VX_gpu_pkg::*, VX_tcu_pkg::*; #(
    parameter `STRING INSTANCE_ID = ""
) (
    input  wire [TCU_TC_K-1:0][7:0][5:0]   sel,
    input  wire [TCU_TC_K-1:0][31:0]       b_col_in1,
    input  wire [TCU_TC_K-1:0][31:0]       b_col_in2,
    output logic [TCU_TC_K-1:0][31:0]      b_col_out
);
    `UNUSED_SPARAM (INSTANCE_ID);

    for (genvar k = 0; k < TCU_TC_K; ++k) begin : g_k
        for (genvar p = 0; p < 8; ++p) begin : g_p
            // Candidate c of output nibble p: a fixed nibble of in1[k] or in2[k].
            wire [5:0][3:0] cand;
            if (p == 0) begin : g_p0
                assign cand[0] = b_col_in1[k][3:0];
                assign cand[1] = b_col_in2[k][3:0];
                assign cand[2] = b_col_in1[k][19:16];
                assign cand[3] = b_col_in2[k][19:16];
                assign cand[4] = b_col_in1[k][11:8];
                assign cand[5] = b_col_in1[k][7:4];
            end
            if (p == 1) begin : g_p1
                assign cand[0] = b_col_in1[k][7:4];
                assign cand[1] = b_col_in2[k][7:4];
                assign cand[2] = b_col_in1[k][23:20];
                assign cand[3] = b_col_in2[k][23:20];
                assign cand[4] = b_col_in1[k][15:12];
                assign cand[5] = b_col_in1[k][11:8];
            end
            if (p == 2) begin : g_p2
                assign cand[0] = b_col_in1[k][11:8];
                assign cand[1] = b_col_in2[k][11:8];
                assign cand[2] = b_col_in1[k][27:24];
                assign cand[3] = b_col_in2[k][27:24];
                assign cand[4] = b_col_in1[k][19:16];
                assign cand[5] = b_col_in1[k][23:20];
            end
            if (p == 3) begin : g_p3
                assign cand[0] = b_col_in1[k][15:12];
                assign cand[1] = b_col_in2[k][15:12];
                assign cand[2] = b_col_in1[k][31:28];
                assign cand[3] = b_col_in2[k][31:28];
                assign cand[4] = b_col_in1[k][23:20];
                assign cand[5] = b_col_in1[k][27:24];
            end
            if (p == 4) begin : g_p4
                assign cand[0] = b_col_in1[k][19:16];
                assign cand[1] = b_col_in2[k][19:16];
                assign cand[2] = b_col_in2[k][3:0];
                assign cand[3] = b_col_in1[k][3:0];
                assign cand[4] = b_col_in2[k][11:8];
                assign cand[5] = b_col_in2[k][7:4];
            end
            if (p == 5) begin : g_p5
                assign cand[0] = b_col_in1[k][23:20];
                assign cand[1] = b_col_in2[k][23:20];
                assign cand[2] = b_col_in2[k][7:4];
                assign cand[3] = b_col_in1[k][7:4];
                assign cand[4] = b_col_in2[k][15:12];
                assign cand[5] = b_col_in2[k][11:8];
            end
            if (p == 6) begin : g_p6
                assign cand[0] = b_col_in1[k][27:24];
                assign cand[1] = b_col_in2[k][27:24];
                assign cand[2] = b_col_in2[k][11:8];
                assign cand[3] = b_col_in1[k][11:8];
                assign cand[4] = b_col_in2[k][19:16];
                assign cand[5] = b_col_in2[k][23:20];
            end
            if (p == 7) begin : g_p7
                assign cand[0] = b_col_in1[k][31:28];
                assign cand[1] = b_col_in2[k][31:28];
                assign cand[2] = b_col_in2[k][15:12];
                assign cand[3] = b_col_in1[k][15:12];
                assign cand[4] = b_col_in2[k][23:20];
                assign cand[5] = b_col_in2[k][27:24];
            end
            always_comb begin
                b_col_out[k][p * 4 +: 4] = '0;
                for (int c = 0; c < 6; ++c) begin
                    b_col_out[k][p * 4 +: 4] |= cand[c] & {4{sel[k][p][c]}};
                end
            end
        end
    end

endmodule

`endif // VX_CFG_TCU_SPARSE_ENABLE
