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
    if (0 == SIZE) begin

        reg [DATAW-1:0] skid_buffer;
        reg skid_valid;

        always @(posedge clk) begin
            if (reset) begin
                skid_valid <= 0;
            end else begin
                if (valid_in && ~ready_out) begin
                    assert(~skid_valid);
                    skid_buffer <= data_in;
                    skid_valid <= 1;
                end
                if (ready_out) begin
                    skid_valid <= 0;
                end
            end
        end

        assign ready_in  = ready_out || ~skid_valid;
        assign data_out  = skid_valid ? skid_buffer : data_in;
        assign valid_out = valid_in || skid_valid;

    end else begin

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

    end

endmodule