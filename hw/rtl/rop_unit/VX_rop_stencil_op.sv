`include "VX_rop_define.vh"

module VX_rop_stencil_op #(
    parameter DATAW = 8
) (
    // Inputs
    input wire [`ROP_STENCIL_OP_BITS-1:0] op,
    input wire [DATAW-1:0] sref,
    input wire [DATAW-1:0] val,

    // Outputs
    output wire [DATAW-1:0] result
);
    wire [DATAW-1:0] stencil_val_n = val + DATAW'(1);
    wire [DATAW-1:0] stencil_val_p = val - DATAW'(1);

    reg [DATAW-1:0] result_r;

    always @(*) begin
        case (op)
            `ROP_STENCIL_OP_KEEP      : result_r = val;
            `ROP_STENCIL_OP_ZERO      : result_r = '0;
            `ROP_STENCIL_OP_REPLACE   : result_r = sref;
            `ROP_STENCIL_OP_INCR      : result_r = (val < `ROP_STENCIL_MASK) ? stencil_val_n : val;
            `ROP_STENCIL_OP_DECR      : result_r = (val > 0) ? stencil_val_p : val;
            `ROP_STENCIL_OP_INVERT    : result_r = ~val;
            `ROP_STENCIL_OP_INCR_WRAP : result_r = stencil_val_n & `ROP_STENCIL_MASK;
            `ROP_STENCIL_OP_DECR_WRAP : result_r = stencil_val_p & `ROP_STENCIL_MASK;
            default                   : result_r = 'x;
        endcase
    end

    assign result = result_r;

endmodule
