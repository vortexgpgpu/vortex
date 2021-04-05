`include "VX_cache_define.vh"

module VX_flush_ctrl #(
    // Size of cache in bytes
    parameter CACHE_SIZE        = 16384, 
    // Size of line inside a bank in bytes
    parameter CACHE_LINE_SIZE   = 1, 
    // Number of banks
    parameter NUM_BANKS         = 1
) (
    input wire  clk,
    input wire  reset,    
    output wire [`LINE_SELECT_BITS-1:0] addr_out,
    output wire valid_out
);
    reg flush_enable;
    reg [`LINE_SELECT_BITS-1:0] flush_ctr;

    always @(posedge clk) begin
        if (reset) begin
            flush_enable <= 1;
            flush_ctr    <= 0;
        end else begin
            if (flush_enable) begin
                if (flush_ctr == ((2 ** `LINE_SELECT_BITS)-1)) begin
                    flush_enable <= 0;
                end
                flush_ctr <= flush_ctr + 1;            
            end
        end
    end

    assign addr_out  = flush_ctr;
    assign valid_out = flush_enable;

endmodule