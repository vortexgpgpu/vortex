`include "VX_platform.vh"

module VX_priority_encoder #( 
    parameter N = 1
) (
    input  wire [N-1:0]         data_in,
    output reg [`LOG2UP(N)-1:0] data_out,
    output reg                  valid_out
);
    integer i;
    
    always @(*) begin
        data_out = 0;
        valid_out = 0;
        for (i = N-1; i >= 0; i = i - 1) begin
            if (data_in[i]) begin
                data_out = `LOG2UP(N)'(i);
                valid_out = 1;
            end
        end
    end
    
endmodule