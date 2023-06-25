`include "VX_cache_define.vh"

module VX_cache_init #(
    // Size of cache in bytes
    parameter CACHE_SIZE    = 16384, 
    // Size of line inside a bank in bytes
    parameter LINE_SIZE     = 1, 
    // Number of banks
    parameter NUM_BANKS     = 1,
    // Number of associative ways
    parameter NUM_WAYS      = 1
) (
    input  wire clk,
    input  wire reset,    
    output wire [`CS_LINE_SEL_BITS-1:0] addr_out,
    output wire valid_out
);
    reg enabled;
    reg [`CS_LINE_SEL_BITS-1:0] line_ctr;

    always @(posedge clk) begin
        if (reset) begin
            enabled  <= 1;
            line_ctr <= '0;
        end else begin
            if (enabled) begin
                if (line_ctr == ((2 ** `CS_LINE_SEL_BITS)-1)) begin
                    enabled <= 0;
                end
                line_ctr <= line_ctr + `CS_LINE_SEL_BITS'(1);           
            end
        end
    end

    assign addr_out  = line_ctr;
    assign valid_out = enabled;

endmodule
