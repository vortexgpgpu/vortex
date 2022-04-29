`include "VX_rop_define.vh"

module VX_rop_compare #(
    parameter DATAW = 24
) (
    // Inputs
    input wire [`ROP_DEPTH_FUNC_BITS-1:0] func,
    input wire [DATAW-1:0] a,
    input wire [DATAW-1:0] b,

    // Outputs
    output reg result
);
    wire [DATAW:0] sub = (a - b);
    wire equal = (0 == sub);
    wire less  = sub[DATAW];

    always @(*) begin
        case (func)
            `ROP_DEPTH_FUNC_NEVER    : result = 0;
            `ROP_DEPTH_FUNC_LESS     : result = less;
            `ROP_DEPTH_FUNC_EQUAL    : result = equal;
            `ROP_DEPTH_FUNC_LEQUAL   : result = less || equal;
            `ROP_DEPTH_FUNC_GREATER  : result = ~(less || equal);
            `ROP_DEPTH_FUNC_NOTEQUAL : result = ~equal;
            `ROP_DEPTH_FUNC_GEQUAL   : result = ~less;
            `ROP_DEPTH_FUNC_ALWAYS   : result = 1;
            default                  : result = 'x;
        endcase        
    end

endmodule
