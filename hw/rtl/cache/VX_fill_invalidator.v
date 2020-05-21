`include "VX_cache_config.vh"

module VX_fill_invalidator #(
    // Size of line inside a bank in bytes
    parameter BANK_LINE_SIZE                = 0, 
    // Number of banks {1, 2, 4, 8,...}
    parameter NUM_BANKS                     = 0, 
     // Fill Invalidator Size {Fill invalidator must be active}
     parameter FILL_INVALIDAOR_SIZE         = 0
) (
    input  wire     clk,
    input  wire     reset,
    input  wire     possible_fill,
    input  wire     success_fill,
    input  wire[`LINE_ADDR_WIDTH-1:0] fill_addr,
    output reg      invalidate_fill    
);

    if (FILL_INVALIDAOR_SIZE == 0) begin
    
        assign invalidate_fill = 0;

        `UNUSED_VAR (clk)
        `UNUSED_VAR (reset)
        `UNUSED_VAR (possible_fill)
        `UNUSED_VAR (success_fill)
        `UNUSED_VAR (fill_addr)

    end else begin 

        reg [FILL_INVALIDAOR_SIZE-1:0]                       fills_active;
        reg [FILL_INVALIDAOR_SIZE-1:0][`LINE_ADDR_WIDTH-1:0] fills_address;

        reg [FILL_INVALIDAOR_SIZE-1:0] matched_fill;
        wire matched;

        integer i;
        always @(*) begin
            for (i = 0; i < FILL_INVALIDAOR_SIZE; i+=1) begin
                matched_fill[i] = fills_active[i] 
                               && ((fills_address[i] == fill_addr) === 1); // use "case equality" to handle uninitialized entry
            end
        end

        assign matched = (|(matched_fill));

        wire [(`LOG2UP(FILL_INVALIDAOR_SIZE))-1:0]  enqueue_index;
        wire                                        enqueue_found;

        VX_generic_priority_encoder #(
            .N(FILL_INVALIDAOR_SIZE)
        ) sel_bank (
            .valids(~fills_active),
            .index (enqueue_index),
            .found (enqueue_found)
        );

        assign invalidate_fill = possible_fill && matched;

        always @(posedge clk) begin
            if (reset) begin
                fills_active <= 0;
            end else begin
                if (possible_fill && !matched && enqueue_found) begin
                    fills_active [enqueue_index] <= 1;
                    fills_address[enqueue_index] <= fill_addr;
                end else if (success_fill && matched) begin
                    fills_active <= fills_active & (~matched_fill);
                end

            end
        end
    end

endmodule