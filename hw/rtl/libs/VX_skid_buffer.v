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
    reg             use_buffer;
    
    always @(posedge clk) begin
        if (reset) begin            
            use_buffer <= 0;
            valid_out  <= 0;  
            data_out   <= 0;  
            buffer     <= 0;
        end else begin
            if (valid_in && ready_in && valid_out && !ready_out) begin
                assert(!use_buffer);
                use_buffer <= 1;
            end 
            if (ready_out) begin
                use_buffer <= 0;
            end
            if (valid_in && ready_in) begin
                buffer <= data_in;
            end
            if (!valid_out || ready_out) begin
                valid_out <= valid_in || use_buffer;
                data_out  <= use_buffer ? buffer : data_in;
            end
        end
    end

    assign ready_in = !use_buffer;

endmodule