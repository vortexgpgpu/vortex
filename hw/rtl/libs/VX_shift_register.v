`include "VX_platform.vh"

module VX_shift_register #( 
    parameter DATAW = 1, 
    parameter DEPTH = 0
) (
    input wire              clk,
    input wire              reset,
    input wire              enable,
    input wire [DATAW-1:0]  in,
    output wire [DATAW-1:0] out
);
    if (0 == DEPTH) begin

        assign out = in;

    end if (1 == DEPTH) begin

        reg [DATAW-1:0] ram;

        always @(posedge clk) begin
            if (reset) begin
                ram <= '0;
            end else begin
                if (enable) begin                    
                    ram <= in;
                end
            end
        end

        assign out = ram;

    end else begin        
        
        reg [DEPTH-1:0][DATAW-1:0] ram;
        
        always @(posedge clk) begin
            if (reset) begin
                ram <= '0;
            end else begin
                if (enable) begin                    
                    ram <= {ram[DEPTH-2:0], in};
                end
            end
        end

        assign out = ram [DEPTH-1];
    end    

endmodule