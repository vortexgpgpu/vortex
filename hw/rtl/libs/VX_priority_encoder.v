`include "VX_platform.vh"

module VX_priority_encoder #( 
    parameter N    = 1,
    parameter LOGN = `LOG2UP(N)
) (
    input  wire [N-1:0]    data_in,
    output wire [LOGN-1:0] data_out,
    output wire            valid_out
);    
    reg [`LOG2UP(N)-1:0] data_out_r;

    always @(*) begin
        data_out_r = 'x;
        for (integer i = 0; i < N; i++) begin
            if (data_in[i]) begin
                data_out_r = LOGN'(i);
                break;
            end
        end
    end

    assign data_out  = data_out_r;
    assign valid_out = (| data_in);
    
endmodule