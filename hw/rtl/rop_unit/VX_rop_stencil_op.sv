`include "VX_rop_define.vh"

module VX_rop_stencil_op #(
    parameter NUM_LANES = 4,
    parameter DATAW = 32
) (
    input wire clk,
    input wire reset,

    // Inputs
    input wire [NUM_LANES-1:0][`ROP_STENCIL_OP_BITS-1:0] stencil_op,
    input wire [NUM_LANES-1:0][DATAW-1:0] stencil_ref,
    input wire [NUM_LANES-1:0][DATAW-1:0] stencil_val,

    // Outputs
    output reg [NUM_LANES-1:0][DATAW-1:0] stencil_result
);

    always @(*) begin
        for (integer i = 0; i < NUM_LANES; i = i + 1) begin
            case (stencil_op[i])
            `ROP_STENCIL_OP_KEEP      : stencil_result[i] = stencil_val[i];
            `ROP_STENCIL_OP_ZERO      : stencil_result[i] = 0;
            `ROP_STENCIL_OP_REPLACE   : stencil_result[i] = stencil_ref[i];
            `ROP_STENCIL_OP_INCR      : stencil_result[i] = (stencil_val[i] < `ROP_STENCIL_BITS'hFF) ? stencil_val[i] + 1 : stencil_val[i];
            `ROP_STENCIL_OP_DECR      : stencil_result[i] = (stencil_val[i] > 0) ? stencil_val[i] - 1 : stencil_val[i];
            `ROP_STENCIL_OP_INVERT    : stencil_result[i] = ~stencil_val[i];
            `ROP_STENCIL_OP_INCR_WRAP : stencil_result[i] = (stencil_val[i] + 1) & `ROP_STENCIL_BITS'hFF;
            `ROP_STENCIL_OP_DECR_WRAP : stencil_result[i] = (stencil_val[i] - 1) & `ROP_STENCIL_BITS'hFF;
            default                   : stencil_result[i] = 'x;
            endcase
        end
    end

endmodule