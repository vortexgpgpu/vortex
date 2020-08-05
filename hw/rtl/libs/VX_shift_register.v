`include "VX_platform.vh"

module VX_shift_register #( 
    parameter DATAW = 1, 
    parameter DEPTH = 1
) (
    input wire              clk,
    input wire              reset,
    input wire              enable,
    input wire [DATAW-1:0]  in,
    output wire [DATAW-1:0] out
);
    reg [DEPTH-1:0][DATAW-1:0] entries;

    if (1 == DEPTH) begin

        always @(posedge clk) begin
            if (reset) begin
                entries <= '0;
            end else begin
                if (enable) begin                    
                    entries <= in;
                end
            end
        end

    end else begin                
        
        always @(posedge clk) begin
            if (reset) begin
                entries <= '0;
            end else begin
                if (enable) begin                    
                    entries <= {entries[DEPTH-2:0], in};
                end
            end
        end
    end    

    assign out = entries [DEPTH-1];

endmodule