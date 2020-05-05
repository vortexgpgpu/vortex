`include "VX_cache_config.vh"

module VX_fill_invalidator #(
    // Size of cache in bytes
    parameter CACHE_SIZE                    = 1024, 
    // Size of line inside a bank in bytes
    parameter BANK_LINE_SIZE                = 16, 
    // Number of banks {1, 2, 4, 8,...}
    parameter NUM_BANKS                     = 8, 
    // Size of a word in bytes
    parameter WORD_SIZE                     = 4, 
    // Number of Word requests per cycle {1, 2, 4, 8, ...}
    parameter NUM_REQUESTS                  = 2, 
    // Number of cycles to complete stage 1 (read from memory)
    parameter STAGE_1_CYCLES                = 2, 

    // Queues feeding into banks Knobs {1, 2, 4, 8, ...}
    // Core Request Queue Size
    parameter REQQ_SIZE                     = 8, 
    // Miss Reserv Queue Knob
    parameter MRVQ_SIZE                     = 8, 
    // Dram Fill Rsp Queue Size
    parameter DFPQ_SIZE                     = 2, 
    // Snoop Req Queue
    parameter SNRQ_SIZE                     = 8, 

    // Queues for writebacks Knobs {1, 2, 4, 8, ...}
    // Core Writeback Queue Size
    parameter CWBQ_SIZE                     = 8, 
    // Dram Writeback Queue Size
    parameter DWBQ_SIZE                     = 4, 
    // Dram Fill Req Queue Size
    parameter DFQQ_SIZE                     = 8, 
    // Lower Level Cache Hit Queue Size
    parameter LLVQ_SIZE                     = 16, 

     // Fill Invalidator Size {Fill invalidator must be active}
     parameter FILL_INVALIDAOR_SIZE          = 16
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

    end else begin 

        reg [FILL_INVALIDAOR_SIZE-1:0]                       fills_active;
        reg [FILL_INVALIDAOR_SIZE-1:0][`LINE_ADDR_WIDTH-1:0] fills_address;

        reg [FILL_INVALIDAOR_SIZE-1:0] matched_fill;
        wire matched;

        integer i;
        always @(*) begin
            for (i = 0; i < FILL_INVALIDAOR_SIZE; i+=1) begin
                matched_fill[i] = fills_active[i] && (fills_address[i] == fill_addr);
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
                fills_active  <= 0;
                fills_address <= 0;
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