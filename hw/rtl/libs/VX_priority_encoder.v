`include "VX_define.vh"

module VX_priority_encoder #( 
    parameter N
) (
    input  wire [N-1:0]             valids,
    output wire [`LOG2UP(N)-1:0]    index,
    output wire                     found
);
    reg [`LOG2UP(N)-1:0] index_r;
    reg found_r;

    integer i;
    always @(*) begin
        index_r = 0;
        found_r = 0;
        for (i = `NUM_WARPS-1; i >= 0; i = i - 1) begin
            if (valids[i]) begin
                index_r = `NW_BITS'(i);
                found_r = 1;
            end
        end
    end

    assign index = index_r;
    assign found = found_r;
    
endmodule