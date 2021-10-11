`include "VX_platform.vh"

module VX_ipdom_stack #(
    parameter WIDTH = 1,
    parameter DEPTH = 1
) (
    input  wire               clk,
    input  wire               reset,
    input  wire               pair,
    input  wire [WIDTH - 1:0] q1,
    input  wire [WIDTH - 1:0] q2,
    output wire [WIDTH - 1:0] d,
    input  wire               push,
    input  wire               pop,
    output wire               index,
    output wire               empty,
    output wire               full
);
    localparam ADDRW = $clog2(DEPTH);

    reg is_part [DEPTH-1:0];
    
    reg [ADDRW-1:0] rd_ptr, wr_ptr;

    wire [WIDTH-1:0] d1, d2;

    always @(posedge clk) begin
        if (reset) begin   
            rd_ptr <= 0;
            wr_ptr <= 0;
        end else begin
            if (push) begin
                rd_ptr <= wr_ptr;
                wr_ptr <= wr_ptr + ADDRW'(1);
            end else if (pop) begin            
                wr_ptr <= wr_ptr - ADDRW'(is_part[rd_ptr]);
                rd_ptr <= rd_ptr - ADDRW'(is_part[rd_ptr]);
            end
        end
    end    

    VX_dp_ram #(
        .DATAW  (WIDTH * 2),
        .SIZE   (DEPTH),
        .LUTRAM (1)
    ) store (
        .clk   (clk),
        .wren  (push),
        .waddr (wr_ptr),
        .wdata ({q2, q1}),
        .raddr (rd_ptr),
        .rdata ({d2, d1})
    );
    
    always @(posedge clk) begin
        if (push) begin
            is_part[wr_ptr] <= ~pair;   
        end else if (pop) begin            
            is_part[rd_ptr] <= 1;
        end
    end

    assign index = is_part[rd_ptr];
    assign d     = index ? d1 : d2;
    assign empty = (ADDRW'(0) == wr_ptr);
    assign full  = (ADDRW'(DEPTH-1) == wr_ptr);

endmodule