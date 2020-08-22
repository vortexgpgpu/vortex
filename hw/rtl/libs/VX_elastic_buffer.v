`include "VX_platform.vh"

module VX_elastic_buffer #(
    parameter DATAW    = 1,
    parameter SIZE     = 2,
    parameter BUFFERED = 1
) ( 
    input  wire             clk,
    input  wire             reset,
    input  wire             valid_in,
    output wire             ready_in,        
    input  wire [DATAW-1:0] data_in,
    output wire [DATAW-1:0] data_out,
    input  wire             ready_out,
    output wire             valid_out
); 
    wire empty, full;

    VX_generic_queue #(
        .DATAW    (DATAW),
        .SIZE     (SIZE),
        .BUFFERED (BUFFERED)
    ) queue (
        .clk    (clk),
        .reset  (reset),
        .push   (valid_in),
        .pop    (ready_out),
        .data_in(data_in),
        .data_out(data_out),        
        .empty  (empty),
        .full   (full),
        `UNUSED_PIN (size)
    );

    assign ready_in  = ~full;
    assign valid_out = ~empty;

endmodule