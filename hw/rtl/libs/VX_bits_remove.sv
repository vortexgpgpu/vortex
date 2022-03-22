`include "VX_platform.vh"

module VX_bits_remove #(
    parameter N   = 1,
    parameter S   = 1,
    parameter POS = 0
) (
    input wire [N-1:0] data_in, 
    output wire [N-S-1:0] data_out
);
    `UNUSED_VAR (data_in)

    if (POS == 0) begin
        assign data_out = data_in[N-1:S];
    end else if (POS == N) begin
        assign data_out = data_in[N-S-1:0];
    end else begin
        assign data_out = {data_in[N-1:(POS+S)], data_in[POS-1:0]};
    end    

endmodule
