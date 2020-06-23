`include "VX_define.vh"

module VX_encoder_onehot #(
    parameter N = 6
) (
    input wire [N-1:0] onehot,
    output reg valid,
    output reg [`LOG2UP(N)-1:0] value
);
    integer i;

    always @(*) begin
        valid = 1'b0;    
        value = {`LOG2UP(N){1'bx}};                
        for (i = 0; i < N; i++) begin
            if (onehot[i]) begin
                valid = 1'b1;
                value = `LOG2UP(N)'(i);
                break;
            end
        end
    end
    
endmodule

