`include "VX_platform.vh"

`TRACING_OFF
module VX_shift_register #( 
    parameter DATAW  = 1,
    parameter RESETW = 0,
    parameter DEPTH  = 1,
    parameter NTAPS  = 1    
) (
    input wire                         clk,
    input wire                         reset,
    input wire                         enable,
    input wire [DATAW-1:0]             data_in,
    output wire [NTAPS-1:0][DATAW-1:0] data_out
);
    localparam TOTAL_DEPTH = NTAPS * DEPTH;

    reg [TOTAL_DEPTH-1:0][DATAW-1:0] entries;

    always @(posedge clk) begin
        for (integer i = 0; i < DATAW; ++i) begin
            if ((i >= (DATAW-RESETW)) && reset) begin
                for (integer j = 0; j < TOTAL_DEPTH; ++j)
                    entries[j][i] <= 0;
            end else if (enable) begin          
                for (integer j = 1; j < TOTAL_DEPTH; ++j)
                    entries[j-1][i] <= entries[j][i];
                entries[TOTAL_DEPTH-1][i] <= data_in[i];
            end
        end
    end

    for (genvar i = 0; i < NTAPS; ++i) begin
        assign data_out[i] = entries[i*DEPTH];
    end

endmodule
`TRACING_ON
