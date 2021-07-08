`include "VX_platform.vh"

module VX_onehot_mux #(
    parameter DATAW = 1,
    parameter COUNT = 1
) (
    input wire [COUNT-1:0][DATAW-1:0] data_in,    
    input wire [COUNT-1:0]            sel_in,    
    output wire [DATAW-1:0]           data_out
); 
    if (COUNT > 1) begin
        for (genvar i = 0; i < COUNT; ++i) begin
            assign data_out = sel_in[i] ? data_in[i] : 'z;
        end
    end else begin
        `UNUSED_VAR (sel_in)
        assign data_out = data_in;
    end

endmodule