`include "VX_platform.vh"

module VX_ipdom_stack #(
    parameter WIDTH = 1,
    parameter DEPTH = 1
) (
    input  wire               clk,
    input  wire               reset,
    input  wire [WIDTH - 1:0] q1,
    input  wire [WIDTH - 1:0] q2,
    output wire [WIDTH - 1:0] d,
    input  wire               push,
    input  wire               pop,
    output wire               empty,
    output wire               full
);
    localparam STACK_SIZE = 2 ** DEPTH;

    reg is_part [STACK_SIZE-1:0];
    
    reg [DEPTH-1:0] rd_ptr, wr_ptr;

    wire [WIDTH - 1:0] d1, d2;

    always @(posedge clk) begin
        if (reset) begin   
            rd_ptr <= 0;
            wr_ptr <= 0;
        end else begin
            if (push) begin
                rd_ptr <= wr_ptr;
                wr_ptr <= wr_ptr + DEPTH'(1);
            end else if (pop) begin            
                wr_ptr <= wr_ptr - DEPTH'(is_part[rd_ptr]);
                rd_ptr <= rd_ptr - DEPTH'(is_part[rd_ptr]);
            end
        end
    end    

    VX_dp_ram #(
        .DATAW(WIDTH * 2),
        .SIZE(STACK_SIZE),
        .BUFFERED(0),
        .RWCHECK(0)
    ) store (
        .clk(clk),
        .waddr(wr_ptr),                                
        .raddr(rd_ptr),
        .wren(push),
        .byteen(1'b1),
        .rden(1'b1),
        .din({q2, q1}),
        .dout({d2, d1})
    );
    
    always @(posedge clk) begin
        if (push) begin
            is_part[wr_ptr] <= 0;   
        end else if (pop) begin            
            is_part[rd_ptr] <= 1;
        end
    end
    wire p = is_part[rd_ptr];

    assign d     = p ? d1 : d2;
    assign empty = ~(| wr_ptr);
    assign full  = ((STACK_SIZE-1) == wr_ptr);

endmodule