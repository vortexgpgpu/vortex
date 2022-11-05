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
    localparam ADDRW = `LOG2UP(SIZE);

    reg [ADDRW-1:0] used_r;  
    reg empty_r;    
    reg full_r;    

    always @(posedge clk) begin
        if (reset) begin          
            used_r  <= '0;
            empty_r <= 1;
            full_r  <= 0;
        end else begin            
            `ASSERT(~(incr && ~decr) || ~full, ("runtime error: incrementing full counter"));
            `ASSERT(~(decr && ~incr) || ~empty, ("runtime error: decrementing empty counter"));
            if (incr) begin
                if (~decr) begin
                    empty_r <= 0;
                    if (used_r == ADDRW'(SIZE-1))
                        full_r <= 1;
                end
            end else if (decr) begin
                full_r <= 0;
                if (used_r == ADDRW'(1))
                    empty_r <= 1;                
            end
            used_r <= $signed(used_r) + ADDRW'($signed(2'(incr) - 2'(decr)));
        end
    end

    assign empty = empty_r;
    assign full  = full_r;
    
    if (SIZE > 1) begin
        if (SIZEW > ADDRW) begin
            assign size = {full_r, used_r};
        end else begin
            assign size = used_r;
        end
    end else begin
        assign size = full_r;
    end
  
endmodule
`TRACING_ON
