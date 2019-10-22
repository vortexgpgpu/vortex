`include "VX_define.v"
//`include "cache_set.v"
`include "VX_Cache_Block_DM.v"

module bank(clk,
            rst,
            state,
            read_or_write,
            //index,
            //tag,
            addr,
            writedata,
            fetched_write_data,
            valid,
            readdata,
            miss_cache,
            w2m_needed,
            w2m_addr,
            e_data,
            //w2m_data,
            ready
      );

  //parameter NUMBER_INDEXES = 16;
  parameter NUMBER_INDEXES = 64;

    localparam CACHE_IDLE = 0; // Idle
    localparam SORT_BY_BANK = 1; // Determines the bank each thread will access
    localparam CACHE_ACCESS = 2; // Accesses the bank and checks if it is a hit or miss
    localparam FETCH_FROM_MEM = 3; // Send a request to mem looking for read data
    localparam FETCH2 = 4; // Stall until memory gets back with the data
    localparam UPDATE_CACHE  = 5; // Update the cache with the data read from mem
    localparam DIRTY_EVICT_GRAB_BLOCK = 6; // Grab the full block of dirty data
    localparam DIRTY_EVICT_WB = 7; // Write back this block into memory
    localparam WB_FROM_MEM = 8; // Currently unused

  input wire          clk, rst;
  input wire          read_or_write;
  input wire [31:0]   writedata;
  input wire [31:0][31:0]        fetched_write_data;
  input wire [3:0]    state;
  //input wire [1:0]    tag;
  //input wire [3:0]    index;
  input wire [31:0]   addr;
  input wire          valid;
  output wire[NUMBER_INDEXES-1:0] [31:0]  readdata;
  output wire         ready;
  //output wire         miss_cache;
  output reg miss_cache;
  output wire [31:0][31:0]        e_data;
  output wire         w2m_needed;
  //output reg [31:0]        w2m_data;
  output reg [31:0]        w2m_addr;

  wire [NUMBER_INDEXES-1:0]               miss;
  //wire [15:0][31:0]               e_data;
  wire [NUMBER_INDEXES-1:0]               e_wb;
  wire [NUMBER_INDEXES-1:0][21:0]          e_tag;
  //wire [3:0]           index;
  //wire                valid_in_set;
  //wire                read_miss;
  //wire                modify;
  wire                hit;
  reg       [NUMBER_INDEXES-1:0] set_to_access;
  reg       [NUMBER_INDEXES-1:0] set_find_evict;
  reg       [NUMBER_INDEXES-1:0] set_idle;
  reg       [NUMBER_INDEXES-1:0] set_wfm;
  //reg       [1:0][15:0] way_id_recieved;
  //reg       [1:0][15:0] way_id_sending;
  //reg                 wb_addr; // Concatination of tag and index for which we will write the data after a memory fetch
 
  // Do logic about processing before going into the cache set here

  assign miss_cache = (miss != 0);
  assign ready = hit && (miss == 0);
  //assign set_wfm = 
  //assign e_tag = miss ? 

  //always @(state) begin
    //miss_cache = (miss != 0);
  //end
  

  //always @(state) begin
    //for (indeces = 0; indeces < NUMBER_INDEXES; indeces = indeces + 1) begin
      //if (set_to_access == indeces) begin
      //if ({28'b0,addr[11:8]} == indeces && state == UPDATE_CACHE && valid) begin
      // reset
        //set_wfm[indeces] = 1'b1;
        //set_find_evict[indeces] = 1'b0;
        //set_idle[indeces] = 1'b0;
        //set_to_access[indeces] = 1'b0;
      //end else if ({28'b0,addr[11:8]} == indeces && state == CACHE_ACCESS && valid) begin 
        //set_to_access[indeces] = 1'b1;
        //set_wfm[indeces] = 1'b0;
        //set_idle[indeces] = 1'b0;
        //set_find_evict[indeces] = 1'b0;
      //end else if ({28'b0,addr[11:8]} == indeces && state == DIRTY_EVICT_GRAB_BLOCK && valid) begin 
        //set_to_access[indeces] = 1'b0;
        //set_wfm[indeces] = 1'b0;
        //set_idle[indeces] = 1'b0;
        //set_find_evict[indeces] = 1'b1;
      //end else begin
        //set_find_evict[indeces] = 1'b0;
        //set_to_access[indeces] = 1'b0;
        //set_idle[indeces] = 1'b1;
        //set_wfm[indeces] = 1'b0;
      //end
    //end
  //end

  for (indeces = 0; indeces < NUMBER_INDEXES; indeces = indeces + 1) begin
    assign set_to_access[indeces] = ({28'b0,addr[11:8]} == indeces && state == CACHE_ACCESS && valid) ? 1'b1 : 1'b0;
    assign set_find_evict[indeces] = ({28'b0,addr[11:8]} == indeces && state == DIRTY_EVICT_GRAB_BLOCK && valid) ? 1'b1 : 1'b0;
    assign set_wfm[indeces] = ({28'b0,addr[11:8]} == indeces && state == UPDATE_CACHE && valid) ? 1'b1 : 1'b0;
    assign set_idle[indeces] = (!set_to_access[indeces] && !set_wfm[indeces] && !set_find_evict[indeces]) ? 1'b1 : 1'b0;
  end


  // reg[31:0][31:0] data[NUMBER_INDEXES-1:0];

  wire[$clog2(NUMBER_INDEXES)-1:0] actual_index;

  assign actual_index = addr[11:8];

  genvar indeces;
  generate
    for (indeces = 0; indeces < NUMBER_INDEXES; indeces = indeces + 1)
      begin
        VX_Cache_Block_DM set(
          .clk          (clk),
          .rst          (rst),
          .actual_index (actual_index)
          .access       (set_to_access[indeces]),
          .find_evict        (set_find_evict[indeces]), 
          .write_from_mem    (set_wfm[indeces]),
          .idle         (set_idle[indeces]),
          //.entry,
          //.o_tag        (tag),
          .o_tag        (addr[31:10]),
          .block_offset (addr[9:5]),
          .writedata    (writedata),
           //byte_en,
          .write        (read_or_write),
          .fetched_writedata (fetched_write_data),
          //.way_id_in    (way_id_sending[indeces]),
          //.way_id_out   (way_id_recieved[indeces]),
           //word_en,

          .readdata     (readdata[indeces]),
          //.wb_addr,
          .hit          (hit),
          //.modify       (modify),
          .eviction_wb   (e_wb[indeces]),
          .eviction_tag  (e_tag[indeces]),
          //.evicted_data (e_data[indeces]),
          .evicted_data (e_data),
          .miss         (miss[indeces])
          //.valid_data        (valid_in_set)
          //.read_miss    (read_miss)
        );
      end
  endgenerate

  //always @(e_wb) begin
  //  for (indeces = 0; indeces < NUMBER_INDEXES; indeces = indeces + 1) begin
  //    //if (set_to_access == indeces) begin
  //    if (e_wb[indeces] == 1'b1) begin
  //    // reset
  //      w2m_needed = 1'b1;
  //      w2m_addr =  {e_tag[indeces], addr[11:0]}; // FIXME !!! Need to figure out how to do this (reassemble the address)
  //     //w2m_data = e_data[indeces];
  //    end
  //  end
  //end

  wire[$clog2(NUMBER_INDEXES)-1:0] index_w2m_addr;
  wire   found_w2m_addr;
  VX_generic_pe #(.N(NUMBER_INDEXES)) find_evicted(
    .valids(e_wb),
    .index(index_w2m_addr),
    .found (found_w2m_addr)
    );

  assign w2m_addr = {e_tag[index_w2m_addr], addr[9:0]};




  assign w2m_needed = (e_wb != 0) ? 1'b1 : 1'b0; 
  for (indeces = 0; indeces < NUMBER_INDEXES; indeces = indeces + 1) begin
    assign set_to_access[indeces] = ({28'b0,addr[11:8]} == indeces && state == CACHE_ACCESS && valid) ? 1'b1 : 1'b0;
  end
  // Do logic about processing done after going into the cache set here

endmodule





