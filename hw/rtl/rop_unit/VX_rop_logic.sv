`include "VX_rop_define.vh"

module VX_rop_logic #(  
    parameter CORE_ID = 0
) (
    // Inputs
    input wire [`ROP_LOGIC_OP_BITS-1:0] logic_op,
    input wire [31:0] src_color,
    input wire [31:0] dst_color,

    // Outputs
    output wire [31:0] color_out
);

    always @(*) begin
        case (logic_op)
            `ROP_LOGIC_OP_CLEAR:         color_out = 32'b0;
            `ROP_LOGIC_OP_AND:           color_out = src_color & dst_color;
            `ROP_LOGIC_OP_AND_REVERSE:   color_out = src_color & (~dst_color);
            `ROP_LOGIC_OP_COPY:          color_out = src_color;
            `ROP_LOGIC_OP_AND_INVERTED:  color_out = (~src_color) & dst_color;
            `ROP_LOGIC_OP_NOOP:          color_out = dst_color;
            `ROP_LOGIC_OP_XOR:           color_out = src_color ^ dst_color;
            `ROP_LOGIC_OP_OR:            color_out = src_color | dst_color;
            `ROP_LOGIC_OP_NOR:           color_out = ~(src_color | dst_color);
            `ROP_LOGIC_OP_EQUIV:         color_out = ~(src_color ^ dst_color);
            `ROP_LOGIC_OP_INVERT:        color_out = ~dst_color;
            `ROP_LOGIC_OP_OR_REVERSE:    color_out = src_color | (~dst_color);
            `ROP_LOGIC_OP_COPY_INVERTED: color_out = ~src_color;
            `ROP_LOGIC_OP_OR_INVERTED:   color_out = (~src_color) | dst_color;
            `ROP_LOGIC_OP_NAND:          color_out = ~(src_color & dst_color);
            `ROP_LOGIC_OP_SET:           color_out = {32{1'b1}};
            default:                     color_out = src_color;
        endcase
    end

endmodule