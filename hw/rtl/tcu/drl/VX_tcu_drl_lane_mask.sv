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

module VX_tcu_drl_lane_mask import VX_tcu_pkg::*; #(
    parameter N   = 2,
    parameter TCK = 2 * N
) (
    input wire [TCU_MAX_INPUTS-1:0] vld_mask,
    input wire [3:0]                fmt_s,
    output logic [TCK-1:0]          lane_mask
);
    `UNUSED_VAR (vld_mask)
    wire [TCK-1:0] mask_32;
    wire [TCK-1:0] mask_16;
    wire [TCK-1:0] mask_8;
    wire [TCK-1:0] mask_4;

    // ----------------------------------------------------------------------
    // 1. 32-bit Mask Generation
    // ----------------------------------------------------------------------
    for (genvar i = 0; i < TCK; ++i) begin : g_mask_32
        if ((i % 2) == 0) begin : g_even_lane
            assign mask_32[i] = vld_mask[i * 4];
        end else begin : g_odd_lane
            assign mask_32[i] = 1'b0;
        end
    end
    `UNUSED_VAR (mask_32)

    // ----------------------------------------------------------------------
    // 2. 16-bit Mask Generation
    // ----------------------------------------------------------------------
    for (genvar i = 0; i < TCK; ++i) begin : g_mask_16
        assign mask_16[i] = vld_mask[i * 4];
    end

    // ----------------------------------------------------------------------
    // 3. 8-bit Mask Generation
    // ----------------------------------------------------------------------
    for (genvar i = 0; i < TCK; ++i) begin : g_mask_8
        assign mask_8[i] = vld_mask[i * 2];
    end
    `UNUSED_VAR (mask_8)

    // ----------------------------------------------------------------------
    // 3. 4-bit Mask Generation
    // ----------------------------------------------------------------------
    for (genvar i = 0; i < TCK; ++i) begin : g_mask_4
        assign mask_4[i] = vld_mask[i];
    end
    `UNUSED_VAR (mask_4)

    // ----------------------------------------------------------------------
    // 4. Final Format Selection
    // ----------------------------------------------------------------------
    always_comb begin
        case (fmt_s)
            TCU_FP16_ID: lane_mask = mask_16;
        `ifdef TCU_BF16_ENABLE
            TCU_BF16_ID: lane_mask = mask_16;
        `endif
        `ifdef TCU_TF32_ENABLE
            TCU_TF32_ID: lane_mask = mask_32;
        `endif
        `ifdef TCU_FP8_ENABLE
            TCU_FP8_ID:  lane_mask = mask_8;
            TCU_BF8_ID:  lane_mask = mask_8;
        `endif
        `ifdef TCU_INT_ENABLE
            TCU_I8_ID:   lane_mask = mask_8;
            TCU_U8_ID:   lane_mask = mask_8;
            TCU_I4_ID:   lane_mask = mask_4;
            TCU_U4_ID:   lane_mask = mask_4;
        `endif
            default:     lane_mask = 'x;
        endcase
    end

endmodule
