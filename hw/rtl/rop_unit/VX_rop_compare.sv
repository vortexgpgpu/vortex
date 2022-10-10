`include "VX_rop_define.vh"

module VX_rop_compare #(
    parameter DATAW = 24
) (
    // Inputs
    input wire [`ROP_DEPTH_FUNC_BITS-1:0] func,
    input wire [DATAW-1:0] a,
    input wire [DATAW-1:0] b,

    // Outputs
    output wire result
);
    wire [DATAW:0] sub = (a - b);
    wire equal = (0 == sub);
    wire less  = sub[DATAW];

    reg result_r;

    always @(*) begin
        case (func)
            `ROP_DEPTH_FUNC_NEVER    : result_r = 0;
            `ROP_DEPTH_FUNC_LESS     : result_r = less;
            `ROP_DEPTH_FUNC_EQUAL    : result_r = equal;
            `ROP_DEPTH_FUNC_LEQUAL   : result_r = less || equal;
            `ROP_DEPTH_FUNC_GREATER  : result_r = ~(less || equal);
            `ROP_DEPTH_FUNC_NOTEQUAL : result_r = ~equal;
            `ROP_DEPTH_FUNC_GEQUAL   : result_r = ~less;
            `ROP_DEPTH_FUNC_ALWAYS   : result_r = 1;
            default                  : result_r = 'x;
        endcase        
    end

    assign result = result_r;

endmodule
