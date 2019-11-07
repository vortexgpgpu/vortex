// Cache Memory (8way 4word)               //
// i_  means input port                    //
// o_  means output port                   //
// _p_  means data exchange with processor //
// _m_  means data exchange with memory    //


// TO DO:
//   - Send in a response from memory of what the data is from the test bench

`include "../VX_define.v"
//`include "VX_priority_encoder.v"
// `include "VX_Cache_Bank.v"
//`include "cache_set.v"

module VX_d_cache
    /*#(
      parameter CACHE_SIZE  = 4096, // Bytes
      parameter CACHE_WAYS  = 1,
      parameter CACHE_BLOCK = 128, // Bytes
      parameter CACHE_BANKS = 8,
      parameter NUM_REQ     = 8
    )*/
    (
               clk,
               rst,
               i_p_addr,
               //i_p_byte_en,
               i_p_writedata,
               i_p_read_or_write, // 0 = Read | 1 = Write
               i_p_mem_read,
               i_p_mem_write,
               i_p_valid,
               //i_p_write,
               o_p_readdata,
               o_p_delay, // 0 = all threads done | 1 = Still threads that need to 

               o_m_evict_addr,
               o_m_read_addr,

               o_m_writedata,
               
               o_m_read_or_write, // 0 = Read | 1 = Write
               o_m_valid,
               i_m_readdata,

               i_m_ready
     );

    //parameter NUMBER_BANKS         = `CACHE_BANKS;
    //localparam NUM_WORDS_PER_BLOCK = `CACHE_BLOCK / (`CACHE_BANKS*4);

    //localparam CACHE_BLOCK_PER_BANK = (`CACHE_BLOCK / `CACHE_BANKS);

    localparam CACHE_IDLE    = 0; // Idle
    localparam SEND_MEM_REQ  = 1; // Write back this block into memory
    localparam RECIV_MEM_RSP = 2;
   
    
    //parameter cache_entry = 9;
    input wire         clk, rst;
    input wire [`DCACHE_NUM_REQ-1:0] i_p_valid;
    input wire  [`DCACHE_NUM_REQ-1:0][31:0] i_p_addr; // FIXME
    input wire [`DCACHE_NUM_REQ-1:0][31:0]  i_p_writedata;
    input wire         i_p_read_or_write; //, i_p_write;
    output reg [`DCACHE_NUM_REQ-1:0][31:0]  o_p_readdata;
    output wire        o_p_delay;
    output reg [31:0]  o_m_evict_addr; // Address is xxxxxxxxxxoooobbbyy
    output reg [31:0]  o_m_read_addr;
    output reg         o_m_valid;
    output reg[`DCACHE_BANKS - 1:0][`DCACHE_NUM_WORDS_PER_BLOCK-1:0][31:0] o_m_writedata;
    output reg                                                     o_m_read_or_write; //, o_m_write;
    input wire[`DCACHE_BANKS - 1:0][`DCACHE_NUM_WORDS_PER_BLOCK-1:0][31:0] i_m_readdata;
    input wire i_m_ready;

    input wire[2:0] i_p_mem_read;
    input wire[2:0] i_p_mem_write;


    // Buffer for final data
    reg [`DCACHE_NUM_REQ-1:0][31:0] final_data_read;
    reg [`DCACHE_NUM_REQ-1:0][31:0] new_final_data_read;
    wire[`DCACHE_NUM_REQ-1:0][31:0] new_final_data_read_Qual;

    assign o_p_readdata = new_final_data_read_Qual;



    wire[`DCACHE_BANKS - 1 : 0][`DCACHE_NUM_REQ-1:0]         thread_track_banks;        // Valid thread mask per bank
    wire[`DCACHE_BANKS - 1 : 0][$clog2(`DCACHE_NUM_REQ)-1:0]  index_per_bank;            // Index of thread each bank will try to service
    wire[`DCACHE_BANKS - 1 : 0][`DCACHE_NUM_REQ-1:0]         use_mask_per_bank;         // A mask of index_per_bank
    wire[`DCACHE_BANKS - 1 : 0]                   valid_per_bank;            // Valid request going to each bank
    wire[`DCACHE_BANKS - 1 : 0][`DCACHE_NUM_REQ-1:0]         threads_serviced_per_bank; // Bank successfully serviced per bank

    wire[`DCACHE_BANKS-1:0][31:0]         readdata_per_bank; // Data read from each bank
    wire[`DCACHE_BANKS-1:0]               hit_per_bank;      // Whether each bank got a hit or a miss
    wire[`DCACHE_BANKS-1:0]               eviction_wb;
    reg[`DCACHE_BANKS-1:0]               eviction_wb_old;


    wire[`DCACHE_BANKS -1 : 0][`DCACHE_WAY_INDEX-1:0] evicted_way_new;
    reg [`DCACHE_BANKS -1 : 0][`DCACHE_WAY_INDEX-1:0] evicted_way_old;
    wire[`DCACHE_BANKS -1 : 0][`DCACHE_WAY_INDEX-1:0] way_used;

    // Internal State
    reg [3:0] state;
    wire[3:0] new_state;

    wire[`DCACHE_NUM_REQ-1:0] use_valid;        // Valid used throught the code
    reg[`DCACHE_NUM_REQ-1:0]  stored_valid;     // Saving the threads still left (bank conflict or bank miss)
    wire[`DCACHE_NUM_REQ-1:0] new_stored_valid; // New stored valid



    reg[`DCACHE_BANKS - 1 : 0][31:0] eviction_addr_per_bank;

    reg[31:0] miss_addr;
    reg[31:0] evict_addr;

    wire curr_processor_request_valid = (|i_p_valid);


    assign use_valid = (stored_valid == 0) ?  i_p_valid : stored_valid;






    VX_cache_bank_valid #(.NUMBER_BANKS(`DCACHE_BANKS)) multip_banks(
      .i_p_valid         (use_valid),
      .i_p_addr          (i_p_addr),
      .thread_track_banks(thread_track_banks)
      );


    reg[`DCACHE_NUM_REQ-1:0] threads_serviced_Qual;

    reg[`DCACHE_NUM_REQ-1:0] debug_hit_per_bank_mask[`DCACHE_BANKS-1:0];

    genvar bid;
    for (bid = 0; bid < `DCACHE_BANKS; bid=bid+1)
    begin
      wire[`DCACHE_NUM_REQ-1:0]        use_threads_track_banks = thread_track_banks[bid];
      wire[$clog2(`DCACHE_NUM_REQ)-1:0] use_thread_index        = index_per_bank[bid];
      wire                  use_write_final_data    = hit_per_bank[bid];
      wire[31:0]            use_data_final_data     = readdata_per_bank[bid];
        VX_priority_encoder_w_mask #(.N(`DCACHE_NUM_REQ)) choose_thread(
          .valids(use_threads_track_banks),
          .mask  (use_mask_per_bank[bid]),
          .index (index_per_bank[bid]),
          .found (valid_per_bank[bid])
          );

        assign debug_hit_per_bank_mask[bid]   = {`DCACHE_NUM_REQ{hit_per_bank[bid]}};
        assign threads_serviced_per_bank[bid] = use_mask_per_bank[bid] & debug_hit_per_bank_mask[bid];
    end

    integer test_bid;
    always @(*) begin
      new_final_data_read = 0;
      for (test_bid=0; test_bid < `DCACHE_BANKS; test_bid=test_bid+1)
      begin
          if (hit_per_bank[test_bid]) begin
            new_final_data_read[index_per_bank[test_bid]] = readdata_per_bank[test_bid];
          end
      end
    end


    wire[`DCACHE_BANKS - 1 : 0] detect_bank_miss;
    //assign threads_serviced_Qual = threads_serviced_per_bank[0] | threads_serviced_per_bank[1] |
    //                               threads_serviced_per_bank[2] | threads_serviced_per_bank[3] |
    //                               threads_serviced_per_bank[4] | threads_serviced_per_bank[5] |
    //                               threads_serviced_per_bank[6] | threads_serviced_per_bank[7];
     integer bbid;
     always @(*) begin
       threads_serviced_Qual = 0;
       for (bbid = 0; bbid < `DCACHE_BANKS; bbid=bbid+1)
       begin
         threads_serviced_Qual = threads_serviced_Qual | threads_serviced_per_bank[bbid];
       end
     end



    genvar tid;
    for (tid = 0; tid < `DCACHE_NUM_REQ; tid =tid+1)
    begin
      assign new_final_data_read_Qual[tid] = threads_serviced_Qual[tid] ? new_final_data_read[tid] : final_data_read[tid];
    end


    assign detect_bank_miss = (valid_per_bank & ~hit_per_bank);

    wire   delay;
    assign delay = (new_stored_valid != 0) || (state != CACHE_IDLE); // add other states

    assign o_p_delay = delay;

    wire[`DCACHE_BANKS - 1 : 0][$clog2(`DCACHE_NUM_REQ)-1:0] send_index_to_bank = index_per_bank;


    wire[$clog2(`DCACHE_BANKS)-1:0] miss_bank_index;
    wire                           miss_found;
    VX_generic_priority_encoder #(.N(`DCACHE_BANKS)) get_miss_index
    (
      .valids(detect_bank_miss),
      .index (miss_bank_index),
      .found (miss_found)
      );

  

  assign new_state        = ((state == CACHE_IDLE)    && (|detect_bank_miss)) ? SEND_MEM_REQ  :
                            (state == SEND_MEM_REQ)                           ? RECIV_MEM_RSP :
                            ((state == RECIV_MEM_RSP) && !i_m_ready)          ? RECIV_MEM_RSP :
                            CACHE_IDLE;

  // Handle if there is more than one miss
  assign new_stored_valid = use_valid & (~threads_serviced_Qual);

///////////////////////////////////////////////////////////////////////
  genvar cur_t;
  integer init_b;
  always @(posedge clk, posedge rst) begin
    if (rst) begin
      final_data_read         <= 0;
      // new_final_data_read      = 0;
      state                   <= 0;
      stored_valid            <= 0;
      // eviction_addr_per_bank  <= 0;
      miss_addr               <= 0;
      evict_addr              <= 0;
      // threads_serviced_Qual    = 0;
      // for (init_b = 0; init_b < NUMBER_BANKS; init_b=init_b+1)
      // begin
      //   debug_hit_per_bank_mask[init_b] <= 0;
      // end
      evicted_way_old <= 0;
      eviction_wb_old <= 0;
      
    end else begin
      state           <= new_state;

      stored_valid    <= new_stored_valid;

      if (miss_found) begin
        miss_addr   <= i_p_addr[send_index_to_bank[miss_bank_index]];
        evict_addr  <= eviction_addr_per_bank[miss_bank_index];
      end

      final_data_read <= new_final_data_read_Qual;
      evicted_way_old <= evicted_way_new;
      eviction_wb_old <= eviction_wb;
    end
  end


  genvar bank_id;
  generate
    for (bank_id = 0; bank_id < `DCACHE_BANKS; bank_id = bank_id + 1)
      begin
        wire[31:0] bank_addr    = (state == SEND_MEM_REQ)  ? evict_addr  :
                                  (state == RECIV_MEM_RSP) ? miss_addr   :
                                  i_p_addr[send_index_to_bank[bank_id]];

        assign evicted_way_new[bank_id] = (state == SEND_MEM_REQ)  ? way_used[bank_id]        :
                                          (state == RECIV_MEM_RSP) ? evicted_way_old[bank_id] :
                                          0;

        wire[1:0]  byte_select                    = bank_addr[1:0];
        wire[`DCACHE_OFFSET_SIZE_RNG] cache_offset = bank_addr[`DCACHE_ADDR_OFFSET_RNG];
        wire[`DCACHE_IND_SIZE_RNG]    cache_index  = bank_addr[`DCACHE_ADDR_IND_RNG];
        wire[`DCACHE_TAG_SIZE_RNG]    cache_tag    = bank_addr[`DCACHE_ADDR_TAG_RNG];


        wire       normal_valid_in = valid_per_bank[bank_id];
        wire       use_valid_in    = ((state == RECIV_MEM_RSP) && i_m_ready)  ? 1'b1 :
                                     ((state == RECIV_MEM_RSP) && !i_m_ready) ? 1'b0 :
                                     ((state == SEND_MEM_REQ))                ? 1'b0 :
                                     normal_valid_in;

        /*VX_Cache_Bank #(
          .CACHE_SIZE(CACHE_SIZE),
          .CACHE_WAYS(CACHE_WAYS),
          .CACHE_BLOCK(CACHE_BLOCK),
          .CACHE_BANKS(CACHE_BANKS)) bank_structure*/
        VX_Cache_Bank bank_structure(
          .clk              (clk),
          .rst              (rst),
          .state            (state),
          .valid_in         (use_valid_in),
          .actual_index     (cache_index),
          .o_tag            (cache_tag),
          .block_offset     (cache_offset),
          .writedata        (i_p_writedata[send_index_to_bank[bank_id]]),
          .read_or_write    (i_p_read_or_write),
          .i_p_mem_read     (i_p_mem_read),
          .i_p_mem_write    (i_p_mem_write),
          .byte_select      (byte_select),
          .hit              (hit_per_bank[bank_id]),
          .readdata         (readdata_per_bank[bank_id]),          // Data read
          .eviction_addr    (eviction_addr_per_bank[bank_id]), 
          .data_evicted     (o_m_writedata[bank_id]),
          .eviction_wb      (eviction_wb[bank_id]),       // Something needs to be written back
          .fetched_writedata(i_m_readdata[bank_id]), // Data From memory
          .evicted_way      (evicted_way_new[bank_id]),
          .way_use          (way_used[bank_id])
        );

      end
  endgenerate

    // Mem Rsp

    // Req to mem:
    assign o_m_evict_addr     = evict_addr & 32'hffffffc0;
    assign o_m_read_addr      = miss_addr  & 32'hffffffc0;
    assign o_m_valid          = (state == SEND_MEM_REQ);
    assign o_m_read_or_write  = (state == SEND_MEM_REQ) && (|eviction_wb_old);
    //end

endmodule





