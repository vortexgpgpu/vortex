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

module VX_tcu_dsm import VX_tcu_pkg::*; #(
    parameter N = 2
) (
    input  wire [4:0]         fmt_s,
    input  wire [N-1:0][31:0] a_row,
    input  wire [N-1:0][31:0] b_col,
    output logic [TCU_MAX_INPUTS-1:0] vld_mask
);

    logic [N-1:0][7:0] vld_mask_per_k;

    assign vld_mask = vld_mask_per_k;

    for (genvar k = 0; k < N; ++k) begin : g_k

        wire a_tf32_nz = |a_row[k][17:0];
        wire b_tf32_nz = |b_col[k][17:0];

        wire [1:0] a_f16_nz;
        wire [1:0] b_f16_nz;
        for (genvar e = 0; e < 2; ++e) begin : g_f16
            assign a_f16_nz[e] = |a_row[k][e * 16 +: 15];
            assign b_f16_nz[e] = |b_col[k][e * 16 +: 15];
        end

        wire [3:0] a_i8_nz;
        wire [3:0] b_i8_nz;
        for (genvar e = 0; e < 4; ++e) begin : g_i8
            assign a_i8_nz[e] = |a_row[k][e * 8 +: 8];
            assign b_i8_nz[e] = |b_col[k][e * 8 +: 8];
        end

        wire [7:0] a_i4_nz;
        wire [7:0] b_i4_nz;
        for (genvar e = 0; e < 8; ++e) begin : g_i4
            assign a_i4_nz[e] = |a_row[k][e * 4 +: 4];
            assign b_i4_nz[e] = |b_col[k][e * 4 +: 4];
        end
        `UNUSED_VAR({a_tf32_nz, b_tf32_nz, a_f16_nz, b_f16_nz, a_i8_nz, b_i8_nz, a_i4_nz, b_i4_nz})

        always_comb begin
            vld_mask_per_k[k] = '1;
            case (fmt_s)
            `ifdef VX_CFG_TCU_TF32_ENABLE
                TCU_TF32_ID: begin
                    vld_mask_per_k[k] = {8{a_tf32_nz && b_tf32_nz}};
                end
            `endif
            `ifdef VX_CFG_TCU_FP16_ENABLE
                TCU_FP16_ID,
                TCU_BF16_ID: begin
                    vld_mask_per_k[k][0 +: 4] = {4{a_f16_nz[0] && b_f16_nz[0]}};
                    vld_mask_per_k[k][4 +: 4] = {4{a_f16_nz[1] && b_f16_nz[1]}};
                end
            `endif
            `ifdef VX_CFG_TCU_FP8_ENABLE
                TCU_FP8_ID,
                TCU_BF8_ID: begin
                    for (int e = 0; e < 4; ++e) begin
                        vld_mask_per_k[k][e * 2 +: 2] = {2{a_i8_nz[e] && b_i8_nz[e]}};
                    end
                end
            `ifdef VX_CFG_TCU_MX_ENABLE
                TCU_MXFP8_ID,
                TCU_MXBF8_ID: begin
                    for (int e = 0; e < 4; ++e) begin
                        vld_mask_per_k[k][e * 2 +: 2] = {2{a_i8_nz[e] && b_i8_nz[e]}};
                    end
                end
            `ifdef VX_CFG_TCU_FP4_ENABLE
            `ifdef VX_CFG_TCU_MXFP4_ENABLE
                TCU_MXFP4_ID: begin
                    for (int e = 0; e < 8; ++e) begin
                        vld_mask_per_k[k][e] = a_i4_nz[e] && b_i4_nz[e];
                    end
                end
            `endif
            `ifdef VX_CFG_TCU_NVFP4_ENABLE
                TCU_NVFP4_ID: begin
                    for (int e = 0; e < 8; ++e) begin
                        vld_mask_per_k[k][e] = a_i4_nz[e] && b_i4_nz[e];
                    end
                end
            `endif
            `endif
            `endif
            `endif
            `ifdef VX_CFG_TCU_INT8_ENABLE
                TCU_I8_ID,
                TCU_U8_ID: begin
                    for (int e = 0; e < 4; ++e) begin
                        vld_mask_per_k[k][e * 2 +: 2] = {2{a_i8_nz[e] && b_i8_nz[e]}};
                    end
                end
            `ifdef VX_CFG_TCU_MX_ENABLE
                TCU_MXI8_ID: begin
                    for (int e = 0; e < 4; ++e) begin
                        vld_mask_per_k[k][e * 2 +: 2] = {2{a_i8_nz[e] && b_i8_nz[e]}};
                    end
                end
            `endif
            `endif
            `ifdef VX_CFG_TCU_INT4_ENABLE
                TCU_I4_ID,
                TCU_U4_ID: begin
                    for (int e = 0; e < 8; ++e) begin
                        vld_mask_per_k[k][e] = a_i4_nz[e] && b_i4_nz[e];
                    end
                end
            `endif
                default: begin
                    vld_mask_per_k[k] = '1;
                end
            endcase
        end
    end

endmodule
