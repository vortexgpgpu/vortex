// To Do: Change way_id_out to an internal register which holds when in between access and finished. 
//        Also add a bit about wheter the "Way ID" is valid / being held or if it is just default
//        Also make sure all possible output states are transmitted back to the bank correctly

`define NUM_WORDS_PER_BLOCK 4

`include "VX_define.v"
`include "VX_cache_data.v"

module VX_Cache_Bank
          #(
            // parameter NUMBER_INDEXES = 256
            parameter NUMBER_INDEXES = 256
          )
          (
            clk,
            state,
            read_or_write, // Read = 0 | Write = 1
            valid_in,
            //write_from_mem,
            actual_index,
            o_tag,
            block_offset,
            writedata,
            fetched_writedata,


            readdata,
            hit,
            //miss,

            eviction_wb, // Need to evict
            eviction_addr, // What's the eviction tag

            data_evicted
           );

    parameter cache_entry = 14;
    parameter ways_per_set = 4;
    parameter Number_Blocks = 32;

    localparam CACHE_IDLE = 0; // Idle
    localparam SORT_BY_BANK = 1; // Determines the bank each thread will access
    localparam INITIAL_ACCESS = 2; // Accesses the bank and checks if it is a hit or miss
    localparam INITIAL_PROCESSING = 3; // Check to see if there were misses 
    localparam CONTINUED_PROCESSING = 4; // Keep checking status of banks that need to be written back or fetched
    localparam DIRTY_EVICT_GRAB_BLOCK = 5; // Grab the full block of dirty data
    localparam DIRTY_EVICT_WB = 6; // Write back this block into memory
    localparam FETCH_FROM_MEM = 7; // Send a request to mem looking for read data
    localparam FETCH2 = 8; // Stall until memory gets back with the data
    localparam UPDATE_CACHE  = 9; // Update the cache with the data read from mem
    localparam RE_ACCESS = 10; // Access the cache after the block has been fetched from memory
    localparam RE_ACCESS_PROCESSING = 11; // Access the cache after the block has been fetched from memory

    // Inputs
    input wire clk;
    input wire [3:0] state;
    //input wire write_from_mem;

      // Reading Data
    input wire[$clog2(NUMBER_INDEXES)-1:0] actual_index;
    input wire[16:0] o_tag; // When write_from_mem = 1, o_tag is the new tag
    input wire[1:0]  block_offset;
    input wire[31:0] writedata;
    input wire       valid_in;
    input wire read_or_write; // Specifies if it is a read or write operation

    input wire[`NUM_WORDS_PER_BLOCK-1:0][31:0] fetched_writedata;



    // Outputs
      // Normal shit
    output wire[31:0] readdata;
    output wire       hit;
    //output wire       miss;

      // Eviction Data (Notice)
    output wire       eviction_wb; // Need to evict
    output wire[31:0] eviction_addr; // What's the eviction tag

      // Eviction Data (Extraction)
    output wire[`NUM_WORDS_PER_BLOCK-1:0][31:0] data_evicted;



    wire[`NUM_WORDS_PER_BLOCK-1:0][31:0] data_use;
    wire[16:0] tag_use;
    wire[16:0] eviction_tag;
    wire       valid_use;
    wire       dirty_use;
    wire       access;
    wire       write_from_mem;
    wire miss; // -10/21


    assign miss = (tag_use != o_tag) && valid_use && valid_in;


    assign data_evicted = data_use;

    assign eviction_wb  = miss && (dirty_use != 1'b0);
    assign eviction_tag = tag_use;
    assign access = (state == INITIAL_ACCESS || state == RE_ACCESS) && valid_in;
    assign write_from_mem = (state == UPDATE_CACHE) && valid_in;
    assign readdata     = (access) ? data_use[block_offset] : 32'b0; // Fix with actual data
    assign hit          = (access && (tag_use == o_tag) && valid_use);
    //assign eviction_addr = {eviction_tag, actual_index, block_offset, 5'b0}; // Fix with actual data
    assign eviction_addr = {eviction_tag, actual_index, 7'b0}; // Fix with actual data


    wire[`NUM_WORDS_PER_BLOCK-1:0]       we;
    wire[`NUM_WORDS_PER_BLOCK-1:0][31:0] data_write;
    genvar g; 
    for (g = 0; g < `NUM_WORDS_PER_BLOCK; g = g + 1) begin
        wire correct_block   = (block_offset == g);
        assign we[g]         = (read_or_write  && ((access && correct_block) || (write_from_mem && !correct_block)) ) ? 1'b1 : 1'b0;
        //assign we[g]         = (!(write_from_mem && correct_block) && ((write_from_mem || correct_block) && read_or_write == 1'b1)) ? 1 : 0; // added the "not"
        assign data_write[g] = write_from_mem ? fetched_writedata[g] : writedata;
    end

    VX_cache_data data_structures(
        .clk       (clk),
        // Inputs
        .addr      (actual_index),
        .we        (we),
        .evict     (write_from_mem),
        .data_write(data_write),
        .tag_write (o_tag),

        // Outputs
        .tag_use   (tag_use),
        .data_use  (data_use),
        .valid_use (valid_use),
        .dirty_use (dirty_use)
      );


endmodule




