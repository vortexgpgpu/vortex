`include "VX_rop_define.vh"

module VX_rop_logic_op #(
    parameter LATENCY = 1
) (
    input wire clk,
    input wire reset,

    input wire enable,

    input wire [`ROP_LOGIC_OP_BITS-1:0] op,
    input wire [31:0] src_color,
    input wire [31:0] dst_color,

    output wire [31:0] color_out
);

    `UNUSED_VAR (reset)
    
    reg [31:0] tmp_color;

    always @(*) begin
        case (op)
            `ROP_LOGIC_OP_CLEAR:         tmp_color = {32{1'b0}};
            `ROP_LOGIC_OP_AND:           tmp_color = src_color & dst_color;
            `ROP_LOGIC_OP_AND_REVERSE:   tmp_color = src_color & (~dst_color);
            `ROP_LOGIC_OP_COPY:          tmp_color = src_color;
            `ROP_LOGIC_OP_AND_INVERTED:  tmp_color = (~src_color) & dst_color;
            `ROP_LOGIC_OP_NOOP:          tmp_color = dst_color;
            `ROP_LOGIC_OP_XOR:           tmp_color = src_color ^ dst_color;
            `ROP_LOGIC_OP_OR:            tmp_color = src_color | dst_color;
            `ROP_LOGIC_OP_NOR:           tmp_color = ~(src_color | dst_color);
            `ROP_LOGIC_OP_EQUIV:         tmp_color = ~(src_color ^ dst_color);
            `ROP_LOGIC_OP_INVERT:        tmp_color = ~dst_color;
            `ROP_LOGIC_OP_OR_REVERSE:    tmp_color = src_color | (~dst_color);
            `ROP_LOGIC_OP_COPY_INVERTED: tmp_color = ~src_color;
            `ROP_LOGIC_OP_OR_INVERTED:   tmp_color = (~src_color) | dst_color;
            `ROP_LOGIC_OP_NAND:          tmp_color = ~(src_color & dst_color);
            `ROP_LOGIC_OP_SET:           tmp_color = {32{1'b1}};
            default:                     tmp_color = 'x;
        endcase
    end

    VX_shift_register #(
        .DATAW  (32),
        .DEPTH  (LATENCY)
    ) shift_reg (
        .clk      (clk),
        `UNUSED_PIN (reset),
        .enable   (enable),
        .data_in  (tmp_color),
        .data_out (color_out)
    );

endmodule
