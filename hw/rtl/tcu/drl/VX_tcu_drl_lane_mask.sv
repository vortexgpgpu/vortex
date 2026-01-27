`include "VX_define.vh"

module VX_tcu_drl_lane_mask import VX_tcu_pkg::*; #(
    parameter N   = 2,
    parameter TCK = 2 * N
) (
    input wire [TCU_MAX_INPUTS-1:0] vld_mask,
    input wire [2:0]                fmt,
    output logic [TCK-1:0]          lane_mask
);
    `UNUSED_VAR (vld_mask)
    wire [TCK-1:0] mask_tf32;
    wire [TCK-1:0] mask_fp16;
    wire [TCK-1:0] mask_bf16;

    // ----------------------------------------------------------------------
    // 1. TF32 Mask Generation
    // ----------------------------------------------------------------------
    // TF32 consumes a full 32-bit register (2 physical lanes).
    // We map it to even physical lanes (0, 2..). Odd lanes are forced to 0.
    for (genvar i = 0; i < TCK; ++i) begin : g_mask_tf32
        if ((i % 2) == 0) begin
            assign mask_tf32[i] = vld_mask[i * 4];
        end else begin
            assign mask_tf32[i] = 1'b0;
        end
    end

    // ----------------------------------------------------------------------
    // 2. FP16 Mask Generation
    // ----------------------------------------------------------------------
    for (genvar i = 0; i < TCK; ++i) begin : g_mask_fp16
        assign mask_fp16[i] = vld_mask[i * 4];
    end

    // ----------------------------------------------------------------------
    // 3. BF16 Mask Generation
    // ----------------------------------------------------------------------
    for (genvar i = 0; i < TCK; ++i) begin : g_mask_bf16
        assign mask_bf16[i] = vld_mask[i * 4];
    end

    // ----------------------------------------------------------------------
    // 4. Final Format Selection
    // ----------------------------------------------------------------------
    always_comb begin
        case (fmt)
            TCU_FP32_ID: lane_mask = mask_tf32;
            TCU_FP16_ID: lane_mask = mask_fp16;
            TCU_BF16_ID: lane_mask = mask_bf16;
            default:     lane_mask = '0;
        endcase
    end

endmodule
