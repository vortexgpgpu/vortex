`include "VX_platform.vh"

module VX_onehot_encoder #(
    parameter N = 6
) (
    input wire [N-1:0] onehot,    
    output wire [`LOG2UP(N)-1:0] binary,
    output wire valid
);
    reg [`LOG2UP(N)-1:0] binary_r;
    reg valid_r;

    always @(*) begin        
        binary_r = 'x;                
        valid_r  = 1'b0;    
        for (integer i = 0; i < N; i++) begin
            if (onehot[i]) begin                
                binary_r = `LOG2UP(N)'(i);
                valid_r  = 1'b1;
            end
        end
    end

    assign binary = binary_r;
    assign valid  = valid_r;
    
endmodule

