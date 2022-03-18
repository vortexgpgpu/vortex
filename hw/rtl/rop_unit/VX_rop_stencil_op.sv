`include "VX_rop_define.vh"

module VX_rop_stencil_op #(
    parameter STENCIL_TEST = 1,
    parameter DATAW = 32
) (
    input wire clk,
    input wire reset,

    // Inputs
    input wire [`ROP_STENCIL_OP_BITS-1:0] stencil_op,
    input wire [DATAW-1:0] stencil_ref,
    input wire [DATAW-1:0] stencil_val,

    // Outputs
    output reg [DATAW-1:0] stencil_result
);

    always @(*) begin
        if (STENCIL_TEST) begin
            case (stencil_op)
                `ROP_STENCIL_OP_KEEP      : stencil_result = stencil_val;
                `ROP_STENCIL_OP_ZERO      : stencil_result = 0;
                `ROP_STENCIL_OP_REPLACE   : stencil_result = stencil_ref;
                `ROP_STENCIL_OP_INCR      : stencil_result = (stencil_val < 8'hFF) ? stencil_val + 1 : stencil_val;
                `ROP_STENCIL_OP_DECR      : stencil_result = (stencil_val > 0) ? stencil_val - 1 : stencil_val;
                `ROP_STENCIL_OP_INVERT    : stencil_result = ~stencil_val;
                `ROP_STENCIL_OP_INCR_WRAP : stencil_result = (stencil_val + 1) & 8'hFF;
                `ROP_STENCIL_OP_DECR_WRAP : stencil_result = (stencil_val - 1) & 8'hFF;
                default                   : stencil_result = 'x;
            endcase
        end else
            stencil_result = stencil_val;
    end

endmodule