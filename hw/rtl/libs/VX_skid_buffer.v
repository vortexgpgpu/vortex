`include "VX_platform.vh"

module VX_skid_buffer #(
    parameter DATAW = 1
) ( 
    input  wire             clk,
    input  wire             reset,
    input  wire             valid_in,
    output reg              ready_in,        
    input  wire [DATAW-1:0] data_in,
    output reg  [DATAW-1:0] data_out,
    input  wire             ready_out,
    output reg              valid_out
); 
    reg	[DATAW-1:0]	buffer;
    reg use_buffer;

    wire push = valid_in && ready_in;
    
    always @(posedge clk) begin
        if (reset) begin            
            use_buffer <= 0;
            valid_out  <= 0;    
        end else begin
            if (push && (valid_out && !ready_out)) begin
                assert(!use_buffer);
                use_buffer <= 1;
            end 
            if (ready_out) begin
                use_buffer <= 0;
            end
            if (push) begin
                buffer <= data_in;
            end
            if (!valid_out || ready_out) begin
                valid_out <= valid_in || use_buffer;
                data_out  <= use_buffer ? buffer : data_in;
            end
        end
    end

    assign ready_in = !use_buffer;
    
    /*wire empty, full;

    VX_generic_queue #(
        .DATAW    (DATAW),
        .SIZE     (2),
        .BUFFERED (0)
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
    assign valid_out = ~empty;*/

endmodule