`include "VX_rop_define.vh"

import VX_rop_types::*;

module VX_rop_stencil_op #(
    parameter DATAW = 8
) (
    // Inputs
    input wire [`ROP_STENCIL_OP_BITS-1:0] stencil_op,
    input wire [DATAW-1:0] stencil_ref,
    input wire [DATAW-1:0] stencil_val,

    // Outputs
    output reg [DATAW-1:0] stencil_result
);
    wire [DATAW-1:0] stencil_val_n = stencil_val + DATAW'(1);
    wire [DATAW-1:0] stencil_val_p = stencil_val - DATAW'(1);

    always @(*) begin
        case (stencil_op)
            `ROP_STENCIL_OP_KEEP      : stencil_result = stencil_val;
            `ROP_STENCIL_OP_ZERO      : stencil_result = '0;
            `ROP_STENCIL_OP_REPLACE   : stencil_result = stencil_ref;
            `ROP_STENCIL_OP_INCR      : stencil_result = (stencil_val < `ROP_STENCIL_MASK) ? stencil_val_n : stencil_val;
            `ROP_STENCIL_OP_DECR      : stencil_result = (stencil_val > 0) ? stencil_val_p : stencil_val;
            `ROP_STENCIL_OP_INVERT    : stencil_result = ~stencil_val;
            `ROP_STENCIL_OP_INCR_WRAP : stencil_result = stencil_val_n & `ROP_STENCIL_MASK;
            `ROP_STENCIL_OP_DECR_WRAP : stencil_result = stencil_val_p & `ROP_STENCIL_MASK;
            default                   : stencil_result = 'x;
        endcase
    end

endmodule
