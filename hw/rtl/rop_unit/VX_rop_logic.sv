`include "VX_rop_define.vh"

module VX_rop_logic #(  
    parameter CORE_ID = 0
) (
    input wire clk,
    input wire reset,

    // Inputs
    input rop_csrs_t rop_csrs,
    input wire [31:0] src_color,
    input wire [31:0] dst_color,
    input wire valid_in,

    // Outputs
    output wire ready_out,
    output wire [31:0] color_out
);

    reg [31:0] result_color;

    always @(*) begin
        case (rop_csrs.logic_op)
            `ROP_LOGIC_OP_CLEAR:         result_color = 32'b0;
            `ROP_LOGIC_OP_AND:           result_color = src_color & dst_color;
            `ROP_LOGIC_OP_AND_REVERSE:   result_color = src_color & (~dst_color);
            `ROP_LOGIC_OP_COPY:          result_color = src_color;
            `ROP_LOGIC_OP_AND_INVERTED:  result_color = (~src_color) & dst_color;
            `ROP_LOGIC_OP_NOOP:          result_color = dst_color;
            `ROP_LOGIC_OP_XOR:           result_color = src_color ^ dst_color;
            `ROP_LOGIC_OP_OR:            result_color = src_color | dst_color;
            `ROP_LOGIC_OP_NOR:           result_color = ~(src_color | dst_color);
            `ROP_LOGIC_OP_EQUIV:         result_color = ~(src_color ^ dst_color);
            `ROP_LOGIC_OP_INVERT:        result_color = ~dst_color;
            `ROP_LOGIC_OP_OR_REVERSE:    result_color = src_color | (~dst_color);
            `ROP_LOGIC_OP_COPY_INVERTED: result_color = ~src_color;
            `ROP_LOGIC_OP_OR_INVERTED:   result_color = (~src_color) | dst_color;
            `ROP_LOGIC_OP_NAND:          result_color = ~(src_color & dst_color);
            `ROP_LOGIC_OP_SET:           result_color = {32{1'b1}};
        endcase
    end

    wire        valid_out;
    wire        store_result;

    assign store_result = valid_in;

    VX_pipe_register #(
        .DATAW  (33),
        .RESETW (1)
    ) pipe_reg (
        .clk      (clk),
        .reset    (reset),
        .enable   (store_result),
        .data_in  ({valid_in,  result_color}),
        .data_out ({valid_out, color_out})
    );

    assign ready_out = valid_out;

endmodule