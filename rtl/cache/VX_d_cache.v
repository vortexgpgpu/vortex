// Cache Memory (8way 4word)               //
// i_  means input port                    //
// o_  means output port                   //
// _p_  means data exchange with processor //
// _m_  means data exchange with memory    //


// TO DO:
//   - Send in a response from memory of what the data is from the test bench

`include "VX_define.v"
//`include "VX_priority_encoder.v"
// `include "VX_Cache_Bank.v"
//`include "cache_set.v"

module VX_d_cache
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
      parameter ADDR_IND_END        = 14,
      parameter MEM_ADDR_REQ_MASK   = 32'hffffffc0
    )
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
    input wire [NUM_REQ-1:0] i_p_valid;
    input wire  [NUM_REQ-1:0][31:0] i_p_addr; // FIXME
    input wire [NUM_REQ-1:0][31:0]  i_p_writedata;
    input wire         i_p_read_or_write; //, i_p_write;
    output reg [NUM_REQ-1:0][31:0]  o_p_readdata;
    output wire        o_p_delay;
    output reg [31:0]  o_m_evict_addr; // Address is xxxxxxxxxxoooobbbyy
    output reg [31:0]  o_m_read_addr;
    output reg         o_m_valid;
    output reg[CACHE_BANKS - 1:0][NUM_WORDS_PER_BLOCK-1:0][31:0] o_m_writedata;
    output reg                                                     o_m_read_or_write; //, o_m_write;
    input wire[CACHE_BANKS - 1:0][NUM_WORDS_PER_BLOCK-1:0][31:0] i_m_readdata;
    input wire i_m_ready;

    input wire[2:0] i_p_mem_read;
    input wire[2:0] i_p_mem_write;


    // Buffer for final data
    reg [NUM_REQ-1:0][31:0] final_data_read;
    reg [NUM_REQ-1:0][31:0] new_final_data_read;
    wire[NUM_REQ-1:0][31:0] new_final_data_read_Qual;

    assign o_p_readdata = new_final_data_read_Qual;


    reg[CACHE_WAY_INDEX-1:0] global_way_to_evict;


    wire[CACHE_BANKS - 1 : 0][NUM_REQ-1:0]         thread_track_banks;        // Valid thread mask per bank
    wire[CACHE_BANKS - 1 : 0][LOG_NUM_REQ-1:0]  index_per_bank;            // Index of thread each bank will try to service
    wire[CACHE_BANKS - 1 : 0][NUM_REQ-1:0]         use_mask_per_bank;         // A mask of index_per_bank
    wire[CACHE_BANKS - 1 : 0]                   valid_per_bank;            // Valid request going to each bank
    wire[CACHE_BANKS - 1 : 0][NUM_REQ-1:0]         threads_serviced_per_bank; // Bank successfully serviced per bank

    wire[CACHE_BANKS-1:0][31:0]         readdata_per_bank; // Data read from each bank
    wire[CACHE_BANKS-1:0]               hit_per_bank;      // Whether each bank got a hit or a miss
    wire[CACHE_BANKS-1:0]               eviction_wb;
    reg[CACHE_BANKS-1:0]               eviction_wb_old;


    // wire[CACHE_BANKS -1 : 0][CACHE_WAY_INDEX-1:0] evicted_way_new;
    // reg [CACHE_BANKS -1 : 0][CACHE_WAY_INDEX-1:0] evicted_way_old;
    // wire[CACHE_BANKS -1 : 0][CACHE_WAY_INDEX-1:0] way_used;

    // Internal State
    reg [3:0] state;
    wire[3:0] new_state;

    wire[NUM_REQ-1:0] use_valid;        // Valid used throught the code
    reg[NUM_REQ-1:0]  stored_valid;     // Saving the threads still left (bank conflict or bank miss)
    wire[NUM_REQ-1:0] new_stored_valid; // New stored valid



    reg[CACHE_BANKS - 1 : 0][31:0] eviction_addr_per_bank;

    reg[31:0] miss_addr;
    // reg[31:0] evict_addr;

    wire curr_processor_request_valid = (|i_p_valid);


    assign use_valid = (stored_valid == 0) ?  i_p_valid : stored_valid;






    VX_cache_bank_valid #(.NUMBER_BANKS  (CACHE_BANKS), 
                          .LOG_NUM_BANKS (LOG_NUM_BANKS), 
                          .NUM_REQ       (NUM_REQ)) multip_banks(
      .i_p_valid         (use_valid),
      .i_p_addr          (i_p_addr),
      .thread_track_banks(thread_track_banks)
      );


    reg[NUM_REQ-1:0] threads_serviced_Qual;

    reg[NUM_REQ-1:0] debug_hit_per_bank_mask[CACHE_BANKS-1:0];

    genvar bid;
	 generate
    for (bid = 0; bid < CACHE_BANKS; bid=bid+1) begin : chooose_threads
      wire[NUM_REQ-1:0]        use_threads_track_banks = thread_track_banks[bid];
      wire[LOG_NUM_REQ-1:0] use_thread_index        = index_per_bank[bid];
      wire                  use_write_final_data    = hit_per_bank[bid];
      wire[31:0]            use_data_final_data     = readdata_per_bank[bid];
        VX_priority_encoder_w_mask #(.N(NUM_REQ)) choose_thread(
          .valids(use_threads_track_banks),
          .mask  (use_mask_per_bank[bid]),
          .index (index_per_bank[bid]),
          .found (valid_per_bank[bid])
          );

        assign debug_hit_per_bank_mask[bid]   = {NUM_REQ{hit_per_bank[bid]}};
        assign threads_serviced_per_bank[bid] = use_mask_per_bank[bid] & debug_hit_per_bank_mask[bid];
    end
	 endgenerate

    integer test_bid;
    always @(*) begin
      new_final_data_read = 0;
      for (test_bid=0; test_bid < CACHE_BANKS; test_bid=test_bid+1)
      begin
          if (hit_per_bank[test_bid]) begin
            new_final_data_read[index_per_bank[test_bid]] = readdata_per_bank[test_bid];
          end
      end
    end


    wire[CACHE_BANKS - 1 : 0] detect_bank_miss;
    //assign threads_serviced_Qual = threads_serviced_per_bank[0] | threads_serviced_per_bank[1] |
    //                               threads_serviced_per_bank[2] | threads_serviced_per_bank[3] |
    //                               threads_serviced_per_bank[4] | threads_serviced_per_bank[5] |
    //                               threads_serviced_per_bank[6] | threads_serviced_per_bank[7];
     integer bbid;
     always @(*) begin
       threads_serviced_Qual = 0;
       for (bbid = 0; bbid < CACHE_BANKS; bbid=bbid+1)
       begin
         threads_serviced_Qual = threads_serviced_Qual | threads_serviced_per_bank[bbid];
       end
     end



    genvar tid;
	 generate
    for (tid = 0; tid < NUM_REQ; tid =tid+1) begin : new_final_data_read_Qual_setup
      assign new_final_data_read_Qual[tid] = threads_serviced_Qual[tid] ? new_final_data_read[tid] : final_data_read[tid];
    end
	 endgenerate


    assign detect_bank_miss = (valid_per_bank & ~hit_per_bank);

    wire   delay;
    assign delay = (new_stored_valid != 0) || (state != CACHE_IDLE); // add other states

    assign o_p_delay = delay;

    wire[CACHE_BANKS - 1 : 0][LOG_NUM_REQ-1:0] send_index_to_bank = index_per_bank;


    wire[LOG_NUM_BANKS-1:0] miss_bank_index;
    wire                           miss_found;
    VX_generic_priority_encoder #(.N(CACHE_BANKS)) get_miss_index
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


  wire update_global_way_to_evict = ((state == RECIV_MEM_RSP) && (new_state == CACHE_IDLE)) && (CACHE_WAYS > 1);

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
      // evict_addr              <= 0;
      // threads_serviced_Qual    = 0;
      // for (init_b = 0; init_b < NUMBER_BANKS; init_b=init_b+1)
      // begin
      //   debug_hit_per_bank_mask[init_b] <= 0;
      // end
      // evicted_way_old <= 0;
      // eviction_wb_old <= 0;
      global_way_to_evict <= 0;
      
    end else begin

      global_way_to_evict <= (update_global_way_to_evict) ? (global_way_to_evict+1) : global_way_to_evict;

      state           <= new_state;

      stored_valid    <= new_stored_valid;

      if (state == CACHE_IDLE) begin
        if (miss_found) begin
          miss_addr   <= i_p_addr[send_index_to_bank[miss_bank_index]];
          // evict_addr  <= eviction_addr_per_bank[miss_bank_index];
        end else begin
          miss_addr   <= 0;
          // evict_addr  <= 0;
        end
      end

      final_data_read <= new_final_data_read_Qual;
      // evicted_way_old <= evicted_way_new;
      // eviction_wb_old <= eviction_wb;
    end
  end


  genvar bank_id;
  generate
    for (bank_id = 0; bank_id < CACHE_BANKS; bank_id = bank_id + 1) begin : cache_banks
        wire[31:0] bank_addr    = (state == SEND_MEM_REQ)  ? miss_addr  :
                                  (state == RECIV_MEM_RSP) ? miss_addr   :
                                  i_p_addr[send_index_to_bank[bank_id]];

        // assign evicted_way_new[bank_id] = (state == SEND_MEM_REQ)  ? way_used[bank_id]        :
        //                                   (state == RECIV_MEM_RSP) ? evicted_way_old[bank_id] :
        //                                   0;

        wire[1:0]  byte_select                    = bank_addr[1:0];
        wire[TAG_SIZE_END:TAG_SIZE_START]    cache_tag    = bank_addr[ADDR_TAG_END:ADDR_TAG_START];

        `ifdef SYN_FUNC
        wire[OFFSET_SIZE_END:OFFSET_SIZE_START] cache_offset = 0;
        wire[IND_SIZE_END:IND_SIZE_START]    cache_index  = 0;
        `else
        wire[OFFSET_SIZE_END:OFFSET_SIZE_START] cache_offset = bank_addr[ADDR_OFFSET_END:ADDR_OFFSET_START];
        wire[IND_SIZE_END:IND_SIZE_START]    cache_index  = bank_addr[ADDR_IND_END:ADDR_IND_START];
        `endif


        wire       normal_valid_in = valid_per_bank[bank_id];
        wire       use_valid_in    = ((state == RECIV_MEM_RSP) && i_m_ready)  ? 1'b1 :
                                     ((state == RECIV_MEM_RSP) && !i_m_ready) ? 1'b0 :
                                     ((state == SEND_MEM_REQ))                ? 1'b0 :
                                     normal_valid_in;


        VX_Cache_Bank #(
          .CACHE_SIZE          (CACHE_SIZE),
          .CACHE_WAYS          (CACHE_WAYS),
          .CACHE_BLOCK         (CACHE_BLOCK),
          .CACHE_BANKS         (CACHE_BANKS),
          .LOG_NUM_BANKS       (LOG_NUM_BANKS),
          .NUM_REQ             (NUM_REQ),
          .LOG_NUM_REQ         (LOG_NUM_REQ),
          .NUM_IND             (NUM_IND),
          .CACHE_WAY_INDEX     (CACHE_WAY_INDEX),
          .NUM_WORDS_PER_BLOCK (NUM_WORDS_PER_BLOCK),
          .OFFSET_SIZE_START   (OFFSET_SIZE_START),
          .OFFSET_SIZE_END     (OFFSET_SIZE_END),
          .TAG_SIZE_START      (TAG_SIZE_START),
          .TAG_SIZE_END        (TAG_SIZE_END),
          .IND_SIZE_START      (IND_SIZE_START),
          .IND_SIZE_END        (IND_SIZE_END),
          .ADDR_TAG_START      (ADDR_TAG_START),
          .ADDR_TAG_END        (ADDR_TAG_END),
          .ADDR_OFFSET_START   (ADDR_OFFSET_START),
          .ADDR_OFFSET_END     (ADDR_OFFSET_END),
          .ADDR_IND_START      (ADDR_IND_START),
          .ADDR_IND_END        (ADDR_IND_END)
          ) bank_structure (
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
          .evicted_way      (global_way_to_evict)
        );

      end
  endgenerate

    // Mem Rsp

    // Req to mem:
    assign o_m_evict_addr     = (eviction_addr_per_bank[0]) & MEM_ADDR_REQ_MASK; // Could be anything because tag+index are same
    assign o_m_read_addr      = miss_addr  & MEM_ADDR_REQ_MASK;
    assign o_m_valid          = (state == SEND_MEM_REQ);
    assign o_m_read_or_write  = (state == SEND_MEM_REQ) && (|eviction_wb);
    //end

endmodule





