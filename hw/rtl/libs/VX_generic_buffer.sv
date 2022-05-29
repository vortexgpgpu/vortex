`include "VX_platform.vh"

`TRACING_OFF
module VX_generic_buffer #(
    parameter DATAW   = 1,
    parameter SKID    = 0,
    parameter OUT_REG = 0
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
    if (0 == SKID) begin

        if (0 == OUT_REG) begin

            `UNUSED_VAR (clk)
            `UNUSED_VAR (reset)

            assign valid_out = valid_in;
            assign data_out  = data_in;
            assign ready_in  = ready_out;

        end else begin

            wire stall = valid_out && ~ready_out;

            VX_pipe_register #(
                .DATAW	(1 + DATAW),
                .RESETW (1),
                .DEPTH  (1)
            ) pipe_reg (
                .clk      (clk),
                .reset    (reset),
                .enable	  (~stall),
                .data_in  ({valid_in,  data_in}),
                .data_out ({valid_out, data_out})
            );

            assign ready_in = ~stall;

        end

    end else begin

        VX_skid_buffer #(
            .DATAW   (DATAW),
            .OUT_REG (OUT_REG)
        ) skid_buf (
            .clk       (clk),
            .reset     (reset),
            .valid_in  (valid_in),        
            .data_in   (data_in),
            .ready_in  (ready_in),      
            .valid_out (valid_out),
            .data_out  (data_out),
            .ready_out (ready_out)
        );
    
    end

endmodule
`TRACING_ON
