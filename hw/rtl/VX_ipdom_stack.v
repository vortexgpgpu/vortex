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

    `NO_RW_RAM_CHECK reg [WIDTH-1:0] stack_1 [0:STACK_SIZE-1];
    `NO_RW_RAM_CHECK reg [WIDTH-1:0] stack_2 [0:STACK_SIZE-1];
    reg is_part [0:STACK_SIZE-1];
    
    reg [DEPTH-1:0] rd_ptr, wr_ptr;

    reg [WIDTH - 1:0] d1, d2;
    reg p;

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

    always @(posedge clk) begin
        if (push) begin
            stack_1[wr_ptr] <= q1;  
        end
    end
    assign d1 = stack_1[rd_ptr];

    always @(posedge clk) begin
        if (push) begin
            stack_2[wr_ptr] <= q2;
        end
    end
    assign d2 = stack_2[rd_ptr];

    always @(posedge clk) begin
        if (push) begin
            is_part[wr_ptr] <= 0;   
        end else if (pop) begin            
            is_part[rd_ptr] <= 1;
        end
    end
    assign p = is_part[rd_ptr];

    assign d     = p ? d1 : d2;
    assign empty = ~(| wr_ptr);
    assign full  = ((STACK_SIZE-1) == wr_ptr);

endmodule