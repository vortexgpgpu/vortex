`include "VX_platform.vh"

module VX_fixed_arbiter #(
    parameter N = 1
) (
    input  wire                  clk,
    input  wire                  reset,
    input  wire [N-1:0]          requests,           
    output wire [`LOG2UP(N)-1:0] grant_index,
    output wire [N-1:0]          grant_onehot,   
    output wire                  grant_valid
  );

    `UNUSED_VAR (clk)
    `UNUSED_VAR (reset)

    if (N == 1)  begin        
        
        assign grant_index  = 0;
        assign grant_onehot = requests;
        assign grant_valid  = requests[0];

    end else begin
    
        reg [N-1:0] grant_onehot_r; 

        VX_priority_encoder # (
            .N(N)
        ) priority_encoder (
            .data_in   (requests),
            .data_out  (grant_index),
            .valid_out (grant_valid)
        );

        always @(*) begin
            grant_onehot_r = N'(0);
            grant_onehot_r[grant_index] = 1;
        end
        assign grant_onehot = grant_onehot_r;    
        
    end
    
endmodule