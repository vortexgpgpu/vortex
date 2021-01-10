`include "VX_platform.vh"

module VX_pending_size #(
    parameter SIZE = 1
) (
    input wire  clk,
    input wire  reset,
    input wire  push,
    input wire  pop,
    output wire full
);
    localparam ADDRW = $clog2(SIZE);

    reg [ADDRW-1:0] size_r;  
    reg full_r;    

    always @(posedge clk) begin
        if (reset) begin          
            size_r <= 0;
            full_r <= 0;
        end else begin
            assert(!push || !full);
            if (push) begin
                if (!pop && (used_r == ADDRW'(SIZE-1)))
                    full_r <= 1;
            end else if (pop) begin
                full_r <= 0;
            end
            size_r <= size_r + ADDRW'($signed(2'(push && !pop) - 2'(pop && !push)));
        end
    end

    assign full = full_r;
  
endmodule