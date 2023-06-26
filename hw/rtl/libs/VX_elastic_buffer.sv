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
    `STATIC_ASSERT (SIZE != 1, ("invalid parameter"))

    if (SIZE == 0) begin

        `UNUSED_VAR (clk)
        `UNUSED_VAR (reset)

        assign valid_out = valid_in;
        assign data_out  = data_in;
        assign ready_in  = ready_out;

    end else if (SIZE == 1) begin

        reg [DATAW-1:0] data_out_r;
        reg             valid_out_r;
        
        wire push = valid_in && ready_in;
        wire stall_out = valid_out_r && ~ready_out;
        
        always @(posedge clk) begin
            if (reset) begin
                valid_out_r <= 0;
            end else begin
                valid_out_r <= valid_in || stall_out;
            end
            if (push) begin
                data_out_r <= data_in;
            end
        end

        assign ready_in  = ~valid_out_r || ready_out;
        assign valid_out = valid_out_r;
        assign data_out  = data_out_r;

    end else if (SIZE == 2) begin

        VX_skid_buffer #(
            .DATAW   (DATAW),
            .OUT_REG (OUT_REG)
        ) queue (
            .clk       (clk),
            .reset     (reset),
            .valid_in  (valid_in),                    
            .ready_in  (ready_in),
            .data_in   (data_in),          
            .data_out  (data_out),
            .valid_out (valid_out),
            .ready_out (ready_out)
        );
    
    end else begin
        
        wire empty, full;

        wire push = valid_in && ready_in;
        wire pop  = valid_out && ready_out;

        VX_fifo_queue #(
            .DATAW   (DATAW),
            .DEPTH   (SIZE),
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
