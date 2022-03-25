`include "VX_platform.vh"

`TRACING_OFF
module VX_pending_size #(
    parameter SIZE  = 1,
    parameter SIZEW = $clog2(SIZE+1)
) (
    input wire  clk,
    input wire  reset,
    input wire  incr,
    input wire  decr,
    output wire empty,
    output wire full,
    output wire [SIZEW-1:0] size
);
    localparam ADDRW = $clog2(SIZE);

    reg [ADDRW-1:0] used_r;  
    reg empty_r;    
    reg full_r;    

    always @(posedge clk) begin
        if (reset) begin          
            used_r  <= 0;
            empty_r <= 1;
            full_r  <= 0;
        end else begin
            `ASSERT(!incr || !full, ("runtime error"));
            if (incr) begin
                if (!decr) begin
                    empty_r <= 0;
                    if (used_r == ADDRW'(SIZE-1))
                        full_r <= 1;
                end
            end else if (decr) begin
                full_r <= 0;
                if (used_r == ADDRW'(1))
                    empty_r <= 1;                
            end
            used_r <= used_r + ADDRW'($signed(2'(incr && !decr) - 2'(decr && !incr)));
        end
    end

    assign empty = empty_r;
    assign full  = full_r;
    assign size  = {full_r, used_r};
  
endmodule
`TRACING_ON
