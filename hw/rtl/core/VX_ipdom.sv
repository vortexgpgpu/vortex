`include "VX_platform.vh"

module VX_ipdom #(
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
    `STATIC_ASSERT(`ISPOW2(DEPTH), ("depth must be a power of 2!"))

    localparam ADDRW = $clog2(DEPTH);

    reg is_part [DEPTH-1:0];
    
    reg [ADDRW-1:0] rd_ptr, wr_ptr;

    reg empty_r, full_r;

    wire [WIDTH-1:0] d1, d2;

    always @(posedge clk) begin
        if (reset) begin   
            rd_ptr  <= '0;
            wr_ptr  <= '0;
            empty_r <= 1;
            full_r  <= 0; 
        end else begin
            `ASSERT(~push || ~full, ("runtime error: writing to a full stack!"));
            `ASSERT(~pop || ~empty, ("runtime error: reading an empty stack!"));
            `ASSERT(~push || ~pop,  ("runtime error: push and pop in same cycle not supported!"));
            if (push) begin                
                rd_ptr  <= wr_ptr;
                wr_ptr  <= wr_ptr + ADDRW'(1);                
                empty_r <= 0;
                full_r  <= (ADDRW'(DEPTH-1) == wr_ptr);
            end else if (pop) begin                   
                wr_ptr  <= wr_ptr - ADDRW'(is_part[rd_ptr]);
                rd_ptr  <= rd_ptr - ADDRW'(is_part[rd_ptr]);
                empty_r <= is_part[rd_ptr] && (0 == rd_ptr);
                full_r  <= 0;
            end
        end
    end    

    VX_dp_ram #(
        .DATAW  (WIDTH * 2),
        .SIZE   (DEPTH),
        .LUTRAM (1)
    ) store (
        .clk   (clk),
        .write (push),        
        `UNUSED_PIN (wren),               
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
    assign empty = empty_r;
    assign full  = full_r;

endmodule
