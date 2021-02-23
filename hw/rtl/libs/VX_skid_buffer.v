`include "VX_platform.vh"

module VX_skid_buffer #(
    parameter DATAW          = 1,
    parameter PASSTHRU       = 0,
    parameter NOBACKPRESSURE = 0,
    parameter BUFFERED       = 0,
    parameter FASTRAM        = 1
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

    end else if (NOBACKPRESSURE) begin

        always @(posedge clk) begin
            assert(ready_out) else $error("ready_out should always be asserted");
        end

        wire stall = valid_out && ~ready_out;

        VX_pipe_register #(
            .DATAW  (1 + DATAW),
            .RESETW (1)
        ) pipe_reg (
            .clk      (clk),
            .reset    (reset),
            .enable   (!stall),
            .data_in  ({valid_in, data_in}),
            .data_out ({valid_out, data_out})
        );

        assign ready_in = ~stall;
    
    end else begin

        if (BUFFERED) begin

            reg [DATAW-1:0] data_out_r;
            reg [DATAW-1:0] buffer;
            reg             valid_out_r;
            reg             use_buffer;
            
            wire push = valid_in && ready_in;
            wire pop = !valid_out_r || ready_out;
            
            always @(posedge clk) begin
                if (reset) begin
                    valid_out_r <= 0; 
                    use_buffer  <= 0;
                end else begin             
                    if (ready_out) begin
                        use_buffer <= 0;
                    end
                    if (push && !pop) begin
                        assert(!use_buffer);
                        use_buffer <= 1;
                    end
                    if (pop) begin
                        valid_out_r <= valid_in || use_buffer;
                    end
                end
            end

            always @(posedge clk) begin
                if (push) begin
                    buffer <= data_in;
                end
                if (pop) begin
                    data_out_r <= use_buffer ? buffer : data_in;
                end
            end

            assign ready_in  = !use_buffer;
            assign valid_out = valid_out_r;
            assign data_out  = data_out_r;

        end else begin

            wire q_push = valid_in && ready_in;
            wire q_pop = valid_out && ready_out;

            wire q_empty, q_full;

            VX_fifo_queue #(
                .DATAW    (DATAW), 
                .SIZE     (2),
                .BUFFERED (BUFFERED),
                .FASTRAM  (FASTRAM)
            ) fifo (
                .clk        (clk),
                .reset      (reset),
                .push       (q_push),
                .pop        (q_pop),
                .data_in    (data_in),        
                .data_out   (data_out),
                .empty      (q_empty),
                .alm_full   (q_full),
                `UNUSED_PIN (full),        
                `UNUSED_PIN (alm_empty),
                `UNUSED_PIN (size)
            );

            assign ready_in  = !q_full;
            assign valid_out = !q_empty;            

        end
    end

endmodule