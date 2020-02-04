// To Do: Change way_id_out to an internal register which holds when in between access and finished. 
//        Also add a bit about wheter the "Way ID" is valid / being held or if it is just default
//        Also make sure all possible output states are transmitted back to the bank correctly

`include "VX_define.v"
// `include "VX_cache_data.v"


module VX_Cache_Bank
    #(
      parameter CACHE_SIZE          = 4096, // Bytes
      parameter CACHE_WAYS          = 1,
      parameter CACHE_BLOCK         = 128, // Bytes
      parameter CACHE_BANKS         = 8,
      parameter LOG_NUM_BANKS       = 3,
      parameter NUM_REQ             = 8,
      parameter LOG_NUM_REQ         = 3,
      parameter NUM_IND             = 8,
      parameter CACHE_WAY_INDEX     = 1,
      parameter NUM_WORDS_PER_BLOCK = 4, 
      parameter OFFSET_SIZE_START   = 0,
      parameter OFFSET_SIZE_END     = 1,
      parameter TAG_SIZE_START      = 0,
      parameter TAG_SIZE_END        = 16,
      parameter IND_SIZE_START      = 0,
      parameter IND_SIZE_END        = 7,
      parameter ADDR_TAG_START      = 15,
      parameter ADDR_TAG_END        = 31,
      parameter ADDR_OFFSET_START   = 5,
      parameter ADDR_OFFSET_END     = 6,
      parameter ADDR_IND_START      = 7,
      parameter ADDR_IND_END        = 14
    )
          (
            clk,
            rst,
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

            data_evicted,
            evicted_way
           );

   // localparam NUMBER_BANKS         = `CACHE_BANKS;
   // localparam CACHE_BLOCK_PER_BANK = (`CACHE_BLOCK / `CACHE_BANKS);
   // localparam NUM_WORDS_PER_BLOCK  = `CACHE_BLOCK / (`CACHE_BANKS*4);
   // localparam NUMBER_INDEXES       = `NUM_IND;

    localparam CACHE_IDLE    = 0; // Idle
    localparam SEND_MEM_REQ  = 1; // Write back this block into memory
    localparam RECIV_MEM_RSP = 2;


    localparam BLOCK_NUM_BITS = `CLOG2(CACHE_BLOCK);
    // Inputs
    input wire rst;
    input wire clk;
    input wire [3:0] state;
//input wire write_from_mem;

      // Reading Data
    input wire[IND_SIZE_END:IND_SIZE_START] actual_index;


    input wire[TAG_SIZE_END:TAG_SIZE_START] o_tag; // When write_from_mem = 1, o_tag is the new tag
    input wire[OFFSET_SIZE_END:OFFSET_SIZE_START]  block_offset;


    input wire[31:0] writedata;
    input wire       valid_in;
    input wire read_or_write; // Specifies if it is a read or write operation

    input wire[NUM_WORDS_PER_BLOCK-1:0][31:0] fetched_writedata;
    input wire[2:0] i_p_mem_read;
    input wire[2:0] i_p_mem_write;
    input wire[1:0] byte_select;


    input  wire[CACHE_WAY_INDEX-1:0] evicted_way;

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
    wire[TAG_SIZE_END:TAG_SIZE_START] tag_use;
    wire[TAG_SIZE_END:TAG_SIZE_START] eviction_tag;
    wire       valid_use;
    wire       dirty_use;
    wire       access;
    wire       write_from_mem;
    wire miss; // -10/21



    wire[CACHE_WAY_INDEX-1:0] way_to_update;

    assign miss = (tag_use != o_tag) && valid_use && valid_in;


    assign data_evicted = data_use;

    // assign eviction_wb    = miss && (dirty_use != 1'b0) && valid_use;
    assign eviction_wb    = (dirty_use != 1'b0);
    assign eviction_tag   = tag_use;
    assign access         = (state == CACHE_IDLE) && valid_in;
    assign write_from_mem = (state == RECIV_MEM_RSP) && valid_in; // TODO
    assign hit            = (access && (tag_use == o_tag) && valid_use);
    //assign eviction_addr = {eviction_tag, actual_index, block_offset, 5'b0}; // Fix with actual data
    assign eviction_addr  = {eviction_tag, actual_index, {(BLOCK_NUM_BITS){1'b0}}}; // Fix with actual data



    wire lw  = (i_p_mem_read == `LW_MEM_READ);
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

    wire[31:0] data_unQual = (b0 || lw) ? (data_use[block_offset]     )  :
                             b1 ? (data_use[block_offset] >> 8)  :
                             b2 ? (data_use[block_offset] >> 16) :
                             (data_use[block_offset] >> 24);


    wire[31:0] lb_data     = (data_unQual[7] ) ? (data_unQual | 32'hFFFFFF00) : (data_unQual & 32'hFF);
    wire[31:0] lh_data     = (data_unQual[15]) ? (data_unQual | 32'hFFFF0000) : (data_unQual & 32'hFFFF);
    wire[31:0] lbu_data    = (data_unQual & 32'hFF);
    wire[31:0] lhu_data    = (data_unQual & 32'hFFFF);
    wire[31:0] lw_data     = (data_unQual);


    wire[31:0] sw_data     = writedata;

    wire[31:0] sb_data     = b1 ? {{16{1'b0}}, writedata[7:0], { 8{1'b0}}} :
                             b2 ? {{ 8{1'b0}}, writedata[7:0], {16{1'b0}}} :
                             b3 ? {{ 0{1'b0}}, writedata[7:0], {24{1'b0}}} :
                             writedata;

    wire[31:0] sh_data     = b2 ? {writedata[15:0], {16{1'b0}}} : writedata;



    wire[31:0] use_write_data  = sb ? sb_data :
                                 sh ? sh_data :
                                 sw_data;


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
	 generate
    for (g = 0; g < NUM_WORDS_PER_BLOCK; g = g + 1) begin : write_enables
        wire normal_write = (read_or_write  && ((access && (block_offset == g))) && !miss);

        assign we[g]      = (write_from_mem)     ? 4'b1111  : 
                            (normal_write && sw) ? 4'b1111  :
                            (normal_write && sb) ? sb_mask  :
                            (normal_write && sh) ? sh_mask  :
                            4'b0000;


        // assign we[g]      = (normal_write || (write_from_mem)) ? 1'b1 : 1'b0;
        assign data_write[g] = write_from_mem ? fetched_writedata[g] : use_write_data;
        assign way_to_update = evicted_way;
    end
	 endgenerate


    VX_cache_data_per_index #(
          .CACHE_WAYS         (CACHE_WAYS),
          .NUM_IND            (NUM_IND),
          .CACHE_WAY_INDEX    (CACHE_WAY_INDEX),
          .NUM_WORDS_PER_BLOCK(NUM_WORDS_PER_BLOCK),
          .TAG_SIZE_START     (TAG_SIZE_START),
          .TAG_SIZE_END       (TAG_SIZE_END),
          .IND_SIZE_START     (IND_SIZE_START),
          .IND_SIZE_END       (IND_SIZE_END)) data_structures(
        .clk          (clk),
        .rst          (rst),
        .valid_in     (valid_in),
        .state        (state),
        // Inputs
        .addr         (actual_index),
        .we           (we),
        .evict        (write_from_mem),
        .data_write   (data_write),
        .tag_write    (o_tag),
        .way_to_update(way_to_update),
        // Outputs
        .tag_use      (tag_use),
        .data_use     (data_use),
        .valid_use    (valid_use),
        .dirty_use    (dirty_use)
      );



endmodule




