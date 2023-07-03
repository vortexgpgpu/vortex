`include "VX_platform.vh"

module VX_ipdom_stack #(
    parameter WIDTH = 1,
    parameter DEPTH = 1,
    parameter ADDRW = `UP($clog2(DEPTH))
) (
    input  wire               clk,
    input  wire               reset,
    input  wire [WIDTH - 1:0] q0,
    input  wire [WIDTH - 1:0] q1,
    output wire [WIDTH - 1:0] d,
    output wire               d_idx,
    output wire [ADDRW-1:0]   q_ptr,
    output wire [ADDRW-1:0]   d_ptr,    
    input  wire               push,
    input  wire               pop,    
    output wire               empty,
    output wire               full
);
    `STATIC_ASSERT(`ISPOW2(DEPTH), ("depth must be a power of 2!"))   

    reg slot_idx [DEPTH-1:0];
    
    reg [ADDRW-1:0] rd_ptr, wr_ptr;

    reg empty_r, full_r;

    wire [WIDTH-1:0] d0, d1;

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
                wr_ptr  <= wr_ptr - ADDRW'(d_idx);
                rd_ptr  <= rd_ptr - ADDRW'(d_idx);
                empty_r <= (rd_ptr == 0) && (d_idx == 1);
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
        .wdata ({q1, q0}),
        .raddr (rd_ptr),
        .rdata ({d1, d0})
    );
    
    always @(posedge clk) begin
        if (push) begin
            slot_idx[wr_ptr] <= 0;   
        end else if (pop) begin            
            slot_idx[rd_ptr] <= 1;
        end
    end

    assign d     = d_idx ? d1 : d0;  
    assign d_idx = slot_idx[rd_ptr];
    assign d_ptr = rd_ptr;
    assign q_ptr = wr_ptr;
    assign empty = empty_r;
    assign full  = full_r;

endmodule
