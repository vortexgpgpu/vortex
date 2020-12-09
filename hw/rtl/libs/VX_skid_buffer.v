`include "VX_platform.vh"

module VX_skid_buffer #(
    parameter DATAW          = 1,
    parameter PASSTHRU       = 0,
    parameter NOBACKPRESSURE = 0
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

        VX_generic_register #(
            .N (1 + DATAW),
            .R (1)
        ) pipe_reg (
            .clk      (clk),
            .reset    (reset),
            .stall    (stall),
            .flush    (1'b0),
            .data_in  ({valid_in, data_in}),
            .data_out ({valid_out, data_out})
        );

        assign ready_in = ~stall;
    
    end else begin

        reg [DATAW-1:0] data_out_r;
        reg [DATAW-1:0] buffer;
        reg             valid_out_r;
        reg             use_buffer;
        
        wire push = valid_in && ready_in;
        
        always @(posedge clk) begin
            if (reset) begin
                valid_out_r <= 0; 
                use_buffer  <= 0;
            end else begin             
                if (ready_out) begin
                    use_buffer <= 0;
                end
                if (push && valid_out_r && !ready_out) begin
                    assert(!use_buffer);
                    use_buffer <= 1;
                end
                if (!valid_out_r || ready_out) begin
                    valid_out_r <= valid_in || use_buffer;
                end
            end

            if (push) begin
                buffer <= data_in;
            end
            
            if (!valid_out_r || ready_out) begin
                data_out_r <= use_buffer ? buffer : data_in;
            end
        end

        assign ready_in  = !use_buffer;
        assign valid_out = valid_out_r;
        assign data_out  = data_out_r;

    end

endmodule