`include "VX_rop_define.vh"

module VX_rop_compare #(
    parameter DATAW = 32
) (
    input wire clk,
    input wire reset,

    // Inputs
    input wire [`ROP_DEPTH_FUNC_BITS-1:0] func,
    input wire [DATAW-1:0] a,
    input wire [DATAW-1:0] b,

    // Outputs
    output reg result
);

    always @(*) begin
        case (func)
            `ROP_DEPTH_FUNC_NEVER    : result = 0;
            `ROP_DEPTH_FUNC_LESS     : result = (a < b);
            `ROP_DEPTH_FUNC_EQUAL    : result = (a == b);
            `ROP_DEPTH_FUNC_LEQUAL   : result = (a <= b);
            `ROP_DEPTH_FUNC_GREATER  : result = (a > b);
            `ROP_DEPTH_FUNC_NOTEQUAL : result = (a != b);
            `ROP_DEPTH_FUNC_GEQUAL   : result = (a >= b);
            `ROP_DEPTH_FUNC_ALWAYS   : result = 1;
            default                  : result = 'x;
        endcase
    end

endmodule