`include "VX_platform.vh"

`TRACING_OFF
module VX_mux #(
    parameter DATAW = 1,
    parameter N     = 1,
    parameter LN    = $clog2(N)
) (
    input wire [N-1:0][DATAW-1:0] data_in,    
    input wire [LN-1:0]           sel_in,    
    output wire [DATAW-1:0]       data_out
); 
    if (N > 1) begin
        assign data_out = data_in[sel_in];
    end else begin
        `UNUSED_VAR (sel_in)
        assign data_out = data_in;
    end

endmodule
`TRACING_ON
