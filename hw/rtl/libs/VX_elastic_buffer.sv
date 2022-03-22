`include "VX_platform.vh"

`TRACING_OFF
module VX_elastic_buffer #(
    parameter DATAW   = 1,
    parameter SIZE    = 2,
    parameter OUT_REG = 0,
    parameter LUTRAM  = 0
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
    `STATIC_ASSERT (SIZE != 1, ("invalid value"))

    if (SIZE == 0) begin

        `UNUSED_VAR (clk)
        `UNUSED_VAR (reset)

        assign valid_out = valid_in;
        assign data_out  = data_in;
        assign ready_in  = ready_out;

    end else if (SIZE == 2) begin

        VX_skid_buffer #(
            .DATAW   (DATAW),
            .OUT_REG (OUT_REG)
        ) queue (
            .clk       (clk),
            .reset     (reset),
            .valid_in  (valid_in),        
            .data_in   (data_in),
            .ready_in  (ready_in),      
            .valid_out (valid_out),
            .data_out  (data_out),
            .ready_out (ready_out)
        );
    
    end else begin
        
        wire empty, full;

        wire push = valid_in && ready_in;
        wire pop  = valid_out && ready_out;

        VX_fifo_queue #(
            .DATAW   (DATAW),
            .SIZE    (SIZE),
            .OUT_REG (OUT_REG),
            .LUTRAM  (LUTRAM)
        ) queue (
            .clk    (clk),
            .reset  (reset),
            .push   (push),
            .pop    (pop),
            .data_in(data_in),
            .data_out(data_out),    
            .empty  (empty),
            .full   (full),
            `UNUSED_PIN (alm_empty),
            `UNUSED_PIN (alm_full),
            `UNUSED_PIN (size)
        );

        assign ready_in  = ~full;
        assign valid_out = ~empty;

    end

endmodule
`TRACING_ON
