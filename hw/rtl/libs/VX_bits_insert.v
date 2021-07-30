`include "VX_platform.vh"

module VX_bits_insert #(
    parameter N   = 1,
    parameter S   = 1,
    parameter POS = 0
) (
    input wire [N-1:0] data_in,    
    input wire [S-1:0] sel_in,    
    output wire [N+S-1:0] data_out
); 
    if (POS == 0) begin
        assign data_out = {data_in, sel_in};
    end else if (POS == N) begin
        assign data_out = {sel_in, data_in};
    end else begin
        assign data_out = {data_in[N-1:POS], sel_in, data_in[POS-1:0]};
    end    

endmodule