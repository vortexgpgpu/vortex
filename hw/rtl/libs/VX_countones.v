
`include "VX_platform.vh"

module VX_countones #(
    parameter N = 10,
    parameter N_BITS = $clog2(N+1)
) (
    input wire [N-1:0]       valids,
    output wire [N_BITS-1:0] count    
);
    /*reg [N_BITS-1:0] count_r;

    always @(*) begin
        count_r = 0;
        for (integer i = N-1; i >= 0; i = i - 1) begin
            if (valids[i]) begin
                count_r = count_r + N_BITS'(1);
            end
        end
    end

    assign count = count_r;*/

    assign count = $countones(valids);

endmodule