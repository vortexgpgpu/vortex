`include "VX_platform.vh"

module VX_bypass_buffer #(
    parameter DATAW    = 1,
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
        assign ready_in  = ready_out;
        assign valid_out = valid_in;        
        assign data_out  = data_in;
    end else begin
        reg [DATAW-1:0] buffer;
        reg buffer_valid;

        always @(posedge clk) begin
            if (reset) begin
                buffer_valid <= 0;
            end else begin            
                if (ready_out) begin
                    buffer_valid <= 0;
                end
                if (valid_in && ~ready_out) begin
                    `ASSERT(!buffer_valid, ("runtime error"));
                    buffer_valid <= 1;
                end
            end

            if (valid_in && ~ready_out) begin
                buffer <= data_in;
            end
        end

        assign ready_in  = ready_out || !buffer_valid;
        assign data_out  = buffer_valid ? buffer : data_in;
        assign valid_out = valid_in || buffer_valid;
    end

endmodule
