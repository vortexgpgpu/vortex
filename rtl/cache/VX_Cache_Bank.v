// To Do: Change way_id_out to an internal register which holds when in between access and finished. 
//        Also add a bit about wheter the "Way ID" is valid / being held or if it is just default
//        Also make sure all possible output states are transmitted back to the bank correctly

`include "../VX_define.v"
`include "VX_cache_data.v"


module VX_Cache_Bank
            #(
              parameter CACHE_SIZE = 4096, // Bytes
              parameter CACHE_WAYS = 1,
              parameter CACHE_BLOCK = 128, // Bytes
              parameter CACHE_BANKS = 8
            )
          (
            clk,
            state,
            read_or_write, // Read = 0 | Write = 1
           i_p_mem_read,
           i_p_mem_write,
            valid_in,
            //write_from_mem,
            actual_index,
            o_tag,
            block_offset,
            writedata,
            fetched_writedata,

            byte_select,

            readdata,
            hit,
            //miss,

            eviction_wb, // Need to evict
            eviction_addr, // What's the eviction tag

            data_evicted
           );

    localparam NUMBER_BANKS         = CACHE_BANKS;
    localparam CACHE_BLOCK_PER_BANK = (CACHE_BLOCK / CACHE_BANKS);
    localparam NUM_WORDS_PER_BLOCK  = CACHE_BLOCK / (CACHE_BANKS*4);
    localparam NUMBER_INDEXES       = `NUM_IND;

    localparam CACHE_IDLE    = 0; // Idle
    localparam SEND_MEM_REQ  = 1; // Write back this block into memory
    localparam RECIV_MEM_RSP = 2;

    // Inputs
    input wire clk;
    input wire [3:0] state;
//input wire write_from_mem;

      // Reading Data
    input wire[`CACHE_IND_SIZE_RNG] actual_index;


    input wire[`CACHE_TAG_SIZE_RNG] o_tag; // When write_from_mem = 1, o_tag is the new tag
    input wire[`CACHE_OFFSET_SIZE_RNG]  block_offset;


    input wire[31:0] writedata;
    input wire       valid_in;
    input wire read_or_write; // Specifies if it is a read or write operation

    input wire[NUM_WORDS_PER_BLOCK-1:0][31:0] fetched_writedata;
    input wire[2:0] i_p_mem_read;
    input wire[2:0] i_p_mem_write;
    input wire[1:0] byte_select;

    // Outputs
      // Normal shit
    output wire[31:0] readdata;
    output wire       hit;
    //output wire       miss;

      // Eviction Data (Notice)
    output wire       eviction_wb; // Need to evict
    output wire[31:0] eviction_addr; // What's the eviction tag

      // Eviction Data (Extraction)
    output wire[NUM_WORDS_PER_BLOCK-1:0][31:0] data_evicted;



    wire[NUM_WORDS_PER_BLOCK-1:0][31:0] data_use;
    wire[16:0] tag_use;
    wire[16:0] eviction_tag;
    wire       valid_use;
    wire       dirty_use;
    wire       access;
    wire       write_from_mem;
    wire miss; // -10/21


    assign miss = (tag_use != o_tag) && valid_use && valid_in;


    assign data_evicted = data_use;

    assign eviction_wb  = (dirty_use != 1'b0) && valid_use;
    assign eviction_tag = tag_use;
    assign access = (state == CACHE_IDLE) && valid_in;
    assign write_from_mem = (state == RECIV_MEM_RSP) && valid_in; // TODO
    assign hit          = (access && (tag_use == o_tag) && valid_use);
    //assign eviction_addr = {eviction_tag, actual_index, block_offset, 5'b0}; // Fix with actual data
    assign eviction_addr = {eviction_tag, actual_index, 7'b0}; // Fix with actual data




    wire lb  = (i_p_mem_read == `LB_MEM_READ);
    wire lh  = (i_p_mem_read == `LH_MEM_READ);
    wire lhu = (i_p_mem_read == `LHU_MEM_READ);
    wire lbu = (i_p_mem_read == `LBU_MEM_READ);

    wire sw  = (i_p_mem_write == `SW_MEM_WRITE);
    wire sb  = (i_p_mem_write == `SB_MEM_WRITE);
    wire sh = (i_p_mem_write == `SH_MEM_WRITE);

    wire b0 = (byte_select == 0);
    wire b1 = (byte_select == 1);
    wire b2 = (byte_select == 2);
    wire b3 = (byte_select == 3);

    wire[31:0] data_unQual = b0 ? (data_use[block_offset]     )  :
                             b1 ? (data_use[block_offset] >> 8)  :
                             b2 ? (data_use[block_offset] >> 16) :
                             (data_use[block_offset] >> 24);


    wire[31:0] lb_data     = (data_unQual[7] ) ? (data_unQual | 32'hFFFFFF00) : (data_unQual & 32'hFF);
    wire[31:0] lh_data     = (data_unQual[15]) ? (data_unQual | 32'hFFFF0000) : (data_unQual & 32'hFFFF);
    wire[31:0] lbu_data    = (data_unQual & 32'hFF);
    wire[31:0] lhu_data    = (data_unQual & 32'hFFFF);
    wire[31:0] lw_data     = (data_unQual);

    wire[31:0] data_Qual   = lb  ? lb_data  :
                             lh  ? lh_data  :
                             lhu ? lhu_data :
                             lbu ? lbu_data :
                             lw_data;


    assign readdata     = (access) ? data_Qual : 32'b0; // Fix with actual data


    wire[3:0] sb_mask = (b0 ? 4'b0001 : (b1 ? 4'b0010 : (b2 ? 4'b0100 : 4'b1000)));
    wire[3:0] sh_mask = (b0 ? 4'b0011 : 4'b1100);


    wire[NUM_WORDS_PER_BLOCK-1:0][3:0]  we;
    wire[NUM_WORDS_PER_BLOCK-1:0][31:0] data_write;
    genvar g; 
    for (g = 0; g < NUM_WORDS_PER_BLOCK; g = g + 1) begin
        wire normal_write = (read_or_write  && ((access && (block_offset == g))) && !miss);

        assign we[g]      = (write_from_mem)     ? 4'b1111  : 
                            (normal_write && sw) ? 4'b1111  :
                            (normal_write && sb) ? sb_mask  :
                            (normal_write && sh) ? sh_mask  :
                            4'b0000;


        // assign we[g]      = (normal_write || (write_from_mem)) ? 1'b1 : 1'b0;
        assign data_write[g] = write_from_mem ? fetched_writedata[g] : writedata;
    end

    VX_cache_data #(
          .CACHE_SIZE(CACHE_SIZE),
          .CACHE_WAYS(CACHE_WAYS),
          .CACHE_BLOCK(CACHE_BLOCK),
          .CACHE_BANKS(CACHE_BANKS)) data_structures(
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




