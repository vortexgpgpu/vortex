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

module VX_tcu_tet_lane_mask import VX_tcu_pkg::*; #(
    parameter N   = 2,
    parameter TCK = 2 * N
) (
    input wire [TCU_MAX_INPUTS-1:0] vld_mask,
    input wire [4:0]                fmt_s,
    output logic [TCK-1:0]          lane_mask
);
    `UNUSED_VAR (vld_mask)
    wire [TCK-1:0] mask_32;
    wire [TCK-1:0] mask_16;
    wire [TCK-1:0] mask_8;
    wire [TCK-1:0] mask_4;

    // 32-bit mask
    for (genvar i = 0; i < TCK; ++i) begin : g_mask_32
        if ((i % 2) == 0) begin : g_even_lane
            assign mask_32[i] = vld_mask[i * 4];
        end else begin : g_odd_lane
            assign mask_32[i] = 1'b0;
        end
    end
    `UNUSED_VAR (mask_32)

    // 16-bit mask
    for (genvar i = 0; i < TCK; ++i) begin : g_mask_16
        assign mask_16[i] = vld_mask[i * 4];
    end
    `UNUSED_VAR (mask_16)

    // 8-bit mask
    for (genvar i = 0; i < TCK; ++i) begin : g_mask_8
        assign mask_8[i] = vld_mask[i * 4 + 0] | vld_mask[i * 4 + 2];
    end
    `UNUSED_VAR (mask_8)

    // 4-bit mask
    for (genvar i = 0; i < TCK; ++i) begin : g_mask_4
        assign mask_4[i] = vld_mask[i * 4 + 0]
                         | vld_mask[i * 4 + 1]
                         | vld_mask[i * 4 + 2]
                         | vld_mask[i * 4 + 3];
    end
    `UNUSED_VAR (mask_4)

    // Format selection
    always_comb begin
        case (fmt_s)
        `ifdef VX_CFG_TCU_FP16_ENABLE
            TCU_FP16_ID,
            TCU_BF16_ID: lane_mask = mask_16;
        `endif
        `ifdef VX_CFG_TCU_TF32_ENABLE
            TCU_TF32_ID: lane_mask = mask_32;
        `endif
        `ifdef VX_CFG_TCU_FP8_ENABLE
            TCU_FP8_ID,
            TCU_BF8_ID:  lane_mask = mask_8;
        `ifdef VX_CFG_TCU_MX_ENABLE
            TCU_MXFP8_ID,
            TCU_MXBF8_ID:lane_mask = mask_8;
        `ifdef VX_CFG_TCU_FP4_ENABLE
        `ifdef VX_CFG_TCU_MXFP4_ENABLE
            TCU_MXFP4_ID:lane_mask = mask_4;
        `endif
        `ifdef VX_CFG_TCU_NVFP4_ENABLE
            TCU_NVFP4_ID:lane_mask = mask_4;
        `endif
        `endif
        `endif
        `endif
        `ifdef VX_CFG_TCU_INT8_ENABLE
            TCU_I8_ID,
            TCU_U8_ID:   lane_mask = mask_8;
        `ifdef VX_CFG_TCU_MX_ENABLE
            TCU_MXI8_ID: lane_mask = mask_8;
        `endif
        `endif
        `ifdef VX_CFG_TCU_INT4_ENABLE
            TCU_I4_ID,
            TCU_U4_ID:   lane_mask = mask_4;
        `endif
            default:     lane_mask = '0;
        endcase
    end

endmodule
