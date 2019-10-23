// Cache Memory (8way 4word)               //
// i_  means input port                    //
// o_  means output port                   //
// _p_  means data exchange with processor //
// _m_  means data exchange with memory    //


// TO DO:
//   - Send in a response from memory of what the data is from the test bench

`include "../VX_define.v"
//`include "VX_priority_encoder.v"
`include "VX_Cache_Bank.v"
//`include "cache_set.v"


module VX_d_cache(clk,
               rst,
               i_p_addr,
               //i_p_byte_en,
               i_p_writedata,
               i_p_read_or_write, // 0 = Read | 1 = Write
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

    parameter NUMBER_BANKS = 8;

    localparam CACHE_IDLE    = 0; // Idle
    localparam SEND_MEM_REQ  = 1; // Write back this block into memory
    localparam RECIV_MEM_RSP = 2;
   
    
    //parameter cache_entry = 9;
    input wire         clk, rst;
    input wire [`NT_M1:0] i_p_valid;
    input wire  [`NT_M1:0][31:0] i_p_addr; // FIXME
    input wire [`NT_M1:0][31:0]  i_p_writedata;
    input wire         i_p_read_or_write; //, i_p_write;
    output reg [`NT_M1:0][31:0]  o_p_readdata;
    output wire        o_p_delay;
    output reg [31:0]  o_m_evict_addr; // Address is xxxxxxxxxxoooobbbyy
    output reg [31:0]  o_m_read_addr;
    output reg         o_m_valid;
    output reg[NUMBER_BANKS - 1:0][`NUM_WORDS_PER_BLOCK-1:0][31:0] o_m_writedata;
    output reg                                                     o_m_read_or_write; //, o_m_write;
    input wire[NUMBER_BANKS - 1:0][`NUM_WORDS_PER_BLOCK-1:0][31:0] i_m_readdata;
    input wire i_m_ready;



    // Buffer for final data
    reg [`NT_M1:0][31:0] final_data_read;
    wire[`NT_M1:0][31:0] new_final_data_read;

    assign o_p_readdata = final_data_read;



    wire[NUMBER_BANKS - 1 : 0][`NT_M1:0]         thread_track_banks;        // Valid thread mask per bank
    wire[NUMBER_BANKS - 1 : 0][$clog2(`NT)-1:0]  index_per_bank;            // Index of thread each bank will try to service
    wire[NUMBER_BANKS - 1 : 0][`NT_M1:0]         use_mask_per_bank;         // A mask of index_per_bank
    wire[NUMBER_BANKS - 1 : 0]                   valid_per_bank;            // Valid request going to each bank
    wire[NUMBER_BANKS - 1 : 0][`NT_M1:0]         threads_serviced_per_bank; // Bank successfully serviced per bank

    wire[NUMBER_BANKS-1:0][31:0]         readdata_per_bank; // Data read from each bank
    wire[NUMBER_BANKS-1:0]               hit_per_bank;      // Whether each bank got a hit or a miss
    wire[NUMBER_BANKS-1:0]               eviction_wb;

    // Internal State
    reg [3:0] state;
    wire[3:0] new_state;

    wire[`NT_M1:0] use_valid;        // Valid used throught the code
    reg[`NT_M1:0]  stored_valid;     // Saving the threads still left (bank conflict or bank miss)
    wire[`NT_M1:0] new_stored_valid; // New stored valid



    reg[NUMBER_BANKS - 1 : 0][31:0] eviction_addr_per_bank;

    reg[31:0] miss_addr;
    reg[31:0] evict_addr;


    assign use_valid = (stored_valid == 0) ?  i_p_valid : stored_valid;





    reg[`NT_M1:0] threads_serviced_Qual;

    VX_cache_bank_valid #(.NUMBER_BANKS(NUMBER_BANKS)) multip_banks(
      .i_p_valid         (use_valid),
      .i_p_addr          (i_p_addr),
      .thread_track_banks(thread_track_banks)
      );


    reg detect_bank_conflict;
    genvar bank_ind;
    for (bank_ind = 0; bank_ind < NUMBER_BANKS; bank_ind=bank_ind+1)
    begin
      assign detect_bank_conflict = detect_bank_conflict | ($countones(thread_track_banks[bank_ind]) > 1);
    
      VX_priority_encoder_w_mask #(.N(`NT)) choose_thread(
        .valids(thread_track_banks[bank_ind]),
        .mask  (use_mask_per_bank[bank_ind]),
        .index (index_per_bank[bank_ind]),
        .found (valid_per_bank[bank_ind])
        );

      ////////////////

      assign new_final_data_read[index_per_bank[bank_ind]] = hit_per_bank[bank_ind] ? readdata_per_bank[bank_ind] : 0;

      assign threads_serviced_per_bank[bank_ind] = use_mask_per_bank[bank_ind] & {`NT{hit_per_bank[bank_ind]}};

    end

    genvar bid;
    for (bid = 0; bid < NUMBER_BANKS; bid=bid+1)
    begin
      assign threads_serviced_Qual = threads_serviced_Qual | threads_serviced_per_bank[bid];
    end


    wire[NUMBER_BANKS - 1 : 0] detect_bank_miss = (valid_per_bank & ~hit_per_bank);

    wire   delay;
    assign delay = (new_stored_valid != 0); // add other states

    assign o_p_delay = delay;

    wire[NUMBER_BANKS - 1 : 0][$clog2(`NT)-1:0] send_index_to_bank = index_per_bank;


    wire[$clog2(NUMBER_BANKS)-1:0] miss_bank_index;
    wire                           miss_found;
    VX_generic_priority_encoder #(.N(NUMBER_BANKS)) get_miss_index
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


  genvar cur_t;
  always @(posedge clk) begin
    state           <= new_state;

    if (state == CACHE_IDLE) stored_valid    <= new_stored_valid;

    if (miss_found) begin
      miss_addr   <= i_p_addr[send_index_to_bank[miss_bank_index]];
      evict_addr  <= eviction_addr_per_bank[miss_bank_index];
    end

    for (cur_t = 0; cur_t < `NT; cur_t=cur_t+1)
    begin
      if (threads_serviced_Qual[cur_t]) final_data_read[cur_t] <= new_final_data_read[cur_t];
    end
  end


  genvar bank_id;
  generate
    for (bank_id = 0; bank_id < NUMBER_BANKS; bank_id = bank_id + 1)
      begin
        wire[31:0] bank_addr    = (state == SEND_MEM_REQ)  ? evict_addr  :
                                  (state == RECIV_MEM_RSP) ? miss_addr   :
                                  i_p_addr[send_index_to_bank[bank_id]];



        wire[7:0]  cache_index  = bank_addr[14:7];
        wire[16:0] cache_tag    = bank_addr[31:15];
        wire[1:0]  cache_offset = bank_addr[6:5];

        wire       normal_valid_in = valid_per_bank[bank_id];
        wire       use_valid_in    = ((state == RECIV_MEM_RSP) && i_m_ready)  ? 1'b1 :
                                     ((state == RECIV_MEM_RSP) && !i_m_ready) ? 1'b0 :
                                     ((state == SEND_MEM_REQ))                ? 1'b0 :
                                     normal_valid_in;

        VX_Cache_Bank bank_structure (
          .clk              (clk),
          .state            (state),
          .valid_in         (use_valid_in),
          .actual_index     (cache_index),
          .o_tag            (cache_tag),
          .block_offset     (cache_offset),
          .writedata        (i_p_writedata[send_index_to_bank[bank_id]]),
          .read_or_write    (i_p_read_or_write),
          .hit              (hit_per_bank[bank_id]),
          .readdata         (readdata_per_bank[bank_id]),          // Data read
          .eviction_addr    (eviction_addr_per_bank[bank_id]), 
          .data_evicted     (o_m_writedata[bank_id]),
          .eviction_wb      (eviction_wb[bank_ind]),       // Something needs to be written back


          .fetched_writedata(i_m_readdata[bank_ind]) // Data From memory
        );

      end
  endgenerate

    // Mem Rsp

    // Req to mem:
    assign o_m_evict_addr     = evict_addr;
    assign o_m_read_addr      = miss_addr;
    assign o_m_valid          = (state == SEND_MEM_REQ);
    assign o_m_read_or_write  = (state == SEND_MEM_REQ) && (|eviction_wb);
    //end

endmodule





