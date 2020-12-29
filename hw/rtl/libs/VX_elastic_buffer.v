`include "VX_platform.vh"

module VX_elastic_buffer #(
    parameter DATAW    = 1,
    parameter SIZE     = 2,
    parameter BUFFERED = 0,
    parameter FASTRAM  = 0,
    parameter PASSTHRU = 0
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
    if (PASSTHRU) begin

        `UNUSED_VAR (clk)
        `UNUSED_VAR (reset)

        assign valid_out = valid_in;
        assign data_out  = data_in;
        assign ready_in  = ready_out;

    end else begin
        
        wire empty, full;

        wire push = valid_in && ready_in;
        wire pop  = valid_out && ready_out;

        VX_fifo_queue #(
            .DATAW    (DATAW),
            .SIZE     (SIZE),
            .BUFFERED (BUFFERED),
            .FASTRAM  (FASTRAM)
        ) queue (
            .clk    (clk),
            .reset  (reset),
            .push   (push),
            .pop    (pop),
            .data_in(data_in),
            .data_out(data_out),    
            .empty  (empty),
            .full   (full),
            `UNUSED_PIN (size)
        );

        assign ready_in  = ~full;
        assign valid_out = ~empty;

    end

endmodule