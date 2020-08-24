
`include "VX_platform.vh"

module VX_ipdom_stack #(
    parameter WIDTH = 1,
    parameter DEPTH = 1
) (
    input  wire              clk,
    input  wire              reset,
    input  reg [WIDTH - 1:0] q1,
    input  reg [WIDTH - 1:0] q2,
    output wire[WIDTH - 1:0] d,
    input  wire              push,
    input  wire              pop,
    output wire              empty,
    output wire              full
);
    localparam STACK_SIZE = 2 ** DEPTH;

    `USE_FAST_BRAM reg [WIDTH-1:0] stack_1 [0:STACK_SIZE-1];
    `USE_FAST_BRAM reg [WIDTH-1:0] stack_2 [0:STACK_SIZE-1];
    `USE_FAST_BRAM reg             is_part [0:STACK_SIZE-1];
    
    reg [DEPTH-1:0] rd_ptr, wr_ptr;

    always @(posedge clk) begin
        if (reset) begin   
            wr_ptr <= 0;
        end else begin
            if (push) begin
                stack_1[wr_ptr] <= q1;
                stack_2[wr_ptr] <= q2;
                is_part[wr_ptr] <= 0;            
                rd_ptr <= wr_ptr;
                wr_ptr <= wr_ptr + DEPTH'(1);
            end else if (pop) begin            
                wr_ptr <= wr_ptr - DEPTH'(is_part[rd_ptr]);
                rd_ptr <= rd_ptr - DEPTH'(is_part[rd_ptr]);
                is_part[rd_ptr] <= 1;
            end
        end
    end

    assign d = is_part[rd_ptr] ? stack_1[rd_ptr] : stack_2[rd_ptr];

    assign empty = (0 == wr_ptr);
    assign full  = ((STACK_SIZE-1) == wr_ptr);

endmodule