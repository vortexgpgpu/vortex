// Cache Memory (8way 4word)               //
// i_  means input port                    //
// o_  means output port                   //
// _p_  means data exchange with processor //
// _m_  means data exchange with memory    //


// TO DO:
//   - Send in a response from memory of what the data is from the test bench

`include "VX_define.v"
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
               o_p_waitrequest, // 0 = all threads done | 1 = Still threads that need to 

               o_m_addr,
               //o_m_byte_en,
               o_m_writedata,
               
               o_m_read_or_write, // 0 = Read | 1 = Write
               o_m_valid,
               //o_m_write,
               i_m_readdata,

               //i_m_readdata_ready,
               //i_m_waitrequest,
               i_m_ready

               //cnt_r,
               //cnt_w,
               //cnt_hit_r,
               //cnt_hit_w
               //cnt_wb_r,
               //cnt_wb_w
     );

    parameter NUMBER_BANKS = 8;

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
   
    
    //parameter cache_entry = 9;
    input wire         clk, rst;
    input wire [`NT_M1:0] i_p_valid;
    //input wire [`NT_M1:0][24:0]  i_p_addr; // FIXME
    input wire  [`NT_M1:0][31:0] i_p_addr; // FIXME
    input wire         i_p_initial_request;
    //input wire [3:0]   i_p_byte_en;
    input wire [`NT_M1:0][31:0]  i_p_writedata;
    input wire         i_p_read_or_write; //, i_p_write;
    output reg [`NT_M1:0][31:0]  o_p_readdata;
    output reg [`NT_M1:0]        o_p_readdata_valid;
    output wire        o_p_waitrequest;
    //output reg [24:0]  o_m_addr; // Only one address is sent out at a time to memory -- FIXME
    output reg [31:0]  o_m_addr; // Address is xxxxxxxxxxoooobbbyy
    output reg         o_m_valid;
    //output wire [255:0][31:0]         evicted_data;
    //output wire [3:0]  o_m_byte_en;
    //output reg [(NUMBER_BANKS * 32) - 1:0] o_m_writedata;
    output reg[NUMBER_BANKS - 1:0][`NUM_WORDS_PER_BLOCK-1:0][31:0] o_m_writedata;
    output reg         o_m_read_or_write; //, o_m_write;
    //input wire [(NUMBER_BANKS * 32) - 1:0] i_m_readdata;  // Read Data that is passed from the memory module back to the controller
    input wire[NUMBER_BANKS - 1:0][`NUM_WORDS_PER_BLOCK-1:0][31:0] i_m_readdata;
    //input wire         i_m_readdata_ready;
    //input wire         i_m_waitrequest;
    input wire i_m_ready;


// Actual logic
    reg [3:0] state;
    wire[3:0] new_state;

    reg [`NT_M1:0][31:0] final_data_read;
    wire[`NT_M1:0][31:0] new_final_data_read;

    wire[NUMBER_BANKS-1:0] readdata_per_bank;

    wire[NUMBER_BANKS-1:0] hit_per_bank;

    wire[`NT_M1:0] use_valid;
    reg[`NT_M1:0]  stored_valid;
    wire[`NT_M1:0] new_stored_valid;


    wire[NUMBER_BANKS - 1 : 0][$clog2(`NT)-1:0] index_per_bank;
    wire[NUMBER_BANKS - 1 : 0]                  valid_per_bank;


    assign use_valid = (stored_valid == 0) ?  i_p_valid : stored_valid;


    wire[NUMBER_BANKS - 1 : 0][`NT_M1:0] thread_track_banks;

    VX_cache_bank_valid #(.NUMBER_BANKS(NUMBER_BANKS)) multip_banks(
      .i_p_valid         (use_valid),
      .i_p_addr          (i_p_addr),
      .thread_track_banks(thread_track_banks)
      );


    reg detect_bank_conflict;
    genvar bank_ind;
    for (bank_ind = 0; bank_ind < NUMBER_BANKS; bank_ind=bank_ind+1)
    begin
      detect_bank_conflict = detect_bank_conflict | ($countones(thread_track_banks[bank_ind]) > 1);
    
      VX_generic_priority_encoder #(.N(1)) choose_thread(
        .valids(thread_track_banks[bank_ind]),
        .index (index_per_bank[bank_ind]),
        .found (valid_per_bank[bank_ind])
        );

      ////////////////

      assign new_final_data_read[index_per_bank[bank_ind]] = hit_per_bank ? readdata_per_bank[bank_ind] : 0;

    end


    wire[NUMBER_BANKS - 1 : 0] detect_bank_miss = (valid_per_bank & ~hit_per_bank);

    wire   delay;
    assign delay = (new_stored_valid != 0); // add other states
    // assign delay = detect_bank_conflict || (|detect_bank_miss) || (state != CACHE_IDLE); // add other states


    wire[NUMBER_BANKS - 1 : 0][$clog2(`NT)-1:0] send_index_to_bank = index_per_bank;

// End actual logic


  

  assign new_state        = detect_bank_miss ? DIRTY_EVICT_WB : CACHE_IDLE;

  // Handle if there is more than one miss
  assign new_stored_valid = (state == CACHE_IDLE) ? ( & ~hit_per_bank);


  genvar bank_id;
  generate
    for (bank_id = 0; bank_id < NUMBER_BANKS; bank_id = bank_id + 1)
      begin
        wire[31:0] bank_addr    = i_p_addr[send_index_to_bank[bank_ind]];
        wire[7:0]  cache_index  = bank_addr[14:7];
        wire[16:0] cache_tag    = bank_addr[31:15];
        wire[1:0]  cache_offset = bank_addr[6:5];
        VX_Cache_Bank bank_structure (
          .clk              (clk),
          .state            (state),
          .valid_in         (valid_per_bank[bank_ind])
          .actual_index     (cache_index),
          .o_tag            (cache_tag),
          .block_offset     (cache_offset),
          .writedata        (i_p_writedata[send_index_to_bank[bank_ind]]),
          .read_or_write    (rd_or_wr),
          .hit              (hit_per_bank[bank_ind]),
          .readdata         (readdata_per_bank[bank_ind]),          // Data read

          .fetched_writedata(fetched_writedata), // From memory
          .eviction_wb      (eviction_wb),
          .eviction_addr    (eviction_addr),
          .data_evicted     (data_evicted)
        );

      end
  endgenerate

    //end

endmodule