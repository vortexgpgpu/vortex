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
               i_p_initial_request,
               i_p_addr,
               //i_p_byte_en,
               i_p_writedata,
               i_p_read_or_write, // 0 = Read | 1 = Write
               i_p_valid,
               //i_p_write,
               o_p_readdata,
               o_p_readdata_valid,
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

    //output reg [31:0]  cnt_r;
    //output reg [31:0]  cnt_w;
    //output reg [31:0]  cnt_hit_r;
    //output reg [31:0]  cnt_hit_w;
    //output reg [31:0]  cnt_wb_r;
    //output reg [31:0]  cnt_wb_w;        
  
    //wire [1:0] tag [`NT_M1:0];
    //wire [3:0] index [`NT_M1:0];
    //wire [2:0] bank [`NT_M1:0];
    //wire all_done;

    //integer i;
    reg [`NT_M1:0] thread_done; // Maybe should have "thread_serviced" and "thread_done", serviced==checked cache
    //reg [`NT_M1:0] thread_serviced; // Maybe should have "thread_serviced" and "thread_done", serviced==checked cache
    reg [NUMBER_BANKS - 1:0] banks_ready;
    //reg [NUMBER_BANKS - 1:0] banks_missed;
    reg [NUMBER_BANKS - 1:0] banks_to_service;
    reg [NUMBER_BANKS - 1:0] banks_wb_needed;
    reg [NUMBER_BANKS - 1:0][31:0] banks_wb_addr;
    //reg [NUMBER_BANKS - 1:0] bank_states;
    //reg [NUMBER_BANKS - 1:0][31:0] banks_wb_data;
    //reg [NUMBER_BANKS - 1:0][13:0] banks_in_addr;
    

    reg [3:0] state;
    reg [NUMBER_BANKS - 1:0][31:0] data_from_bank;
    //reg got_valid_data;
    //reg [31:0] data_to_write;


    //reg [`NT_M1:0] thread_track_bank_0;
    //reg [`NT_M1:0] thread_track_bank_1;
    //reg [`NT_M1:0] thread_track_bank_2;
    //reg [`NT_M1:0] thread_track_bank_3;
    //reg [`NT_M1:0] thread_track_bank_4;
    //reg [`NT_M1:0] thread_track_bank_5;
    //reg [`NT_M1:0] thread_track_bank_6;
    //reg [`NT_M1:0] thread_track_bank_7;
    reg [NUMBER_BANKS - 1 : 0][`NT_M1:0] thread_track_banks;
    reg [NUMBER_BANKS - 1 : 0] bank_has_access;  // Will track if a bank has been accessed in this cycle
    reg [NUMBER_BANKS - 1 : 0][31:0] bank_access_addr;
    reg [NUMBER_BANKS - 1 : 0][31:0] bank_access_data;
    reg [NUMBER_BANKS - 1 : 0][1:0] threads_in_banks;


    //reg [1:0] thread_in_memory; // keeps track of threadID which is in memory
    reg rd_or_wr;
    //reg did_miss, needs_service; Commented out Oct 21

    integer bnk;
    integer found;
    integer t_id;
    //integer num_misses;
    //integer num_evictions_to_wb;
    integer i;    //reg [1:0] correct_tag;
    integer index;
    //reg [3:0] correct_index;

    //assign tag = i_p_addr[13:12];

    assign o_p_waitrequest =  (thread_done == 4'hF) ? 1'b0 : 1'b1; // change thread_done to be generic
    //assign did_miss = (banks_missed != 8'h0) ?  1'b1 : 1'b0;
    //assign needs_service = ((banks_to_service != 8'b0 || banks_to_service_temp != 8'b0)) ? 1'b1 : 1'b0; // added banks_to_service temp
    //assign w_Test1 = r_Check  ? 1'b1  : 1'b0;
    //for ( i = 0;i < `NT_M1;i = i + 1) begin
     // assign tag[i] = i_p_addr[i][13:12];

     // Fares
//     wire no_bank_misses;
//     assign no_bank_misses = banks_to_service != 8'b0;

     reg[NUMBER_BANKS - 1:0] banks_to_service_temp;
     reg[NUMBER_BANKS - 1:0] banks_to_wb;
     reg[NUMBER_BANKS - 1:0] banks_to_wb_temp;
     reg[NUMBER_BANKS - 1:0] banks_all_help;
    

    always @(posedge clk) begin 
      if (rst) begin 
        state <= 0;
        //banks_ready <= 8'b0;
        //cnt_r <= 0;
        //cnt_w <= 0;
        //cnt_hit_r <= 0;
        //cnt_hit_w <= 0;
        //cnt_wb_r <= 0;
        //cnt_wb_w <= 0;

      end else begin
        // Change Logic of which state the cache is in
        case (state)
          CACHE_IDLE:begin
            if (i_p_initial_request == 1'b1) begin 
              state <= SORT_BY_BANK;
            end else begin
              state <= CACHE_IDLE;
            end
          end
          SORT_BY_BANK:begin
            state <= INITIAL_ACCESS;
          end
          INITIAL_ACCESS:begin 
            if (thread_done == 4'hF) begin
              state <= CACHE_IDLE;
            end else begin
              state <= INITIAL_PROCESSING;
            end
          end
          INITIAL_PROCESSING:begin 
            if (bank_has_access == banks_ready ) begin // if all hits
              state <= INITIAL_ACCESS;
            end else begin
              state <= CONTINUED_PROCESSING;
            end

          end          
          CONTINUED_PROCESSING:begin
            if (banks_to_wb == 8'b0 && banks_to_service == 8'b0) begin // If all threads are done, then the cache can go back into idle state (not currently fetching any requests)
              state <= INITIAL_ACCESS;
            //end else if (num_misses > 0) begin
            end else if ((banks_to_wb != 8'b0)) begin // change 1pm
              state <= DIRTY_EVICT_GRAB_BLOCK;
            //end else if (did_miss == 1'b1 || needs_service == 1'b1) begin
            end else if(banks_to_service != 8'b0) begin
              state <= FETCH_FROM_MEM;
           // end else if (did_miss == 1'b0 && num_evictions_to_wb > 0) begin
            //end else if (needs_service == 1'b0 && did_miss == 1'b0 && (banks_to_wb != 8'b0)) begin
            //end else if (did_miss == 1'b0 && needs_service == 1'b0) begin
              //state <= INITIAL_ACCESS;
            end            
          end
          FETCH_FROM_MEM:begin
            state <= FETCH2;
          end
          FETCH2:begin
            if (i_m_ready == 1'b1) begin
              state <= UPDATE_CACHE; // Not sure about this one !!!!!! Check
            end else begin
              state <= FETCH2;
            end
          end
          UPDATE_CACHE:begin
            state <= RE_ACCESS;
          end
          RE_ACCESS:begin
            state <= RE_ACCESS_PROCESSING;
          end
          RE_ACCESS_PROCESSING: begin
            state <= CONTINUED_PROCESSING;
          end
          DIRTY_EVICT_GRAB_BLOCK:begin
            state <= DIRTY_EVICT_WB;
          end
          DIRTY_EVICT_WB:begin
            state <= CONTINUED_PROCESSING;
          end
        endcase
      end

      //tag[`NT_M1:0] <= i_p_addr[`NT_M1:0][13:12]; 
    end

    // Change values which will be fed into the cache
    always @(*) begin
        case (state)
          CACHE_IDLE:begin
            thread_done = 0;
            o_m_read_or_write = 0;
            o_m_valid = 0;
            o_m_writedata = 0;
            o_p_readdata = 0;
            o_p_readdata_valid = 0;
            bank_has_access = 8'b0;
            //bank_states = CACHE_IDLE;
            //thread_track_bank_0 = 4'b0;
            //thread_track_bank_1 = 4'b0;
            //thread_track_bank_2 = 4'b0;
            //thread_track_bank_3 = 4'b0;
            //thread_track_bank_4 = 4'b0;
            //thread_track_bank_5 = 4'b0;
            //thread_track_bank_6 = 4'b0;
            //thread_track_bank_7 = 4'b0;
            for (bnk = 0; bnk < NUMBER_BANKS; bnk = bnk + 1) begin
              thread_track_banks[bnk] = 4'b0;
            end
          end
          SORT_BY_BANK:begin
            //bank_states = SORT_BY_BANK;
            rd_or_wr = i_p_read_or_write;
            for (t_id = 0; t_id <= `NT_M1; t_id = t_id + 1) begin
              //t_id = {1'b0,t_id};
              if (i_p_valid[t_id] == 1'b0) begin
                thread_done[t_id] = 1'b1;
              end
               //if (i_p_valid[t_id] == 1'b1 && thread_done[t_id] == 1'b0) begin // Need logic for thread done
              else  if (i_p_addr[t_id][4:2] == 3'b000) begin
                  //banks_in_addr[0] = i_p_addr[t_id]; // WIll need to do this later
                  //thread_track_bank_0[t_id] = 1'b1;
                  thread_track_banks[0][t_id] = 1'b1;
              end
              else if (i_p_addr[t_id][4:2] == 3'b001) begin // !!!!!!!
                  //banks_in_addr[0] = i_p_addr[t_id]; // WIll need to do this later
                  //thread_track_bank_1[t_id] = 1'b1;
                  thread_track_banks[1][t_id] = 1'b1;
              end
              else if (i_p_addr[t_id][4:2] == 3'b010) begin
                  //banks_in_addr[0] = i_p_addr[t_id]; // WIll need to do this later
                  //thread_track_bank_2[t_id] = 1'b1;
                  thread_track_banks[2][t_id] = 1'b1;
              end
              else if (i_p_addr[t_id][4:2] == 3'b011) begin
                  //banks_in_addr[0] = i_p_addr[t_id]; // WIll need to do this later
                  //thread_track_bank_3[t_id] = 1'b1;
                  thread_track_banks[3][t_id] = 1'b1;
              end
              else if (i_p_addr[t_id][4:2] == 3'b100) begin
                  //banks_in_addr[0] = i_p_addr[t_id]; // WIll need to do this later
                  //thread_track_bank_4[t_id] = 1'b1;
                  thread_track_banks[4][t_id] = 1'b1;
              end
              else if (i_p_addr[t_id][4:2] == 3'b101) begin
                  //banks_in_addr[0] = i_p_addr[t_id]; // WIll need to do this later
                  //thread_track_bank_5[t_id] = 1'b1;
                  thread_track_banks[5][t_id] = 1'b1;
              end
              else if (i_p_addr[t_id][4:2] == 3'b110) begin
                  //banks_in_addr[0] = i_p_addr[t_id]; // WIll need to do this later
                  //thread_track_bank_6[t_id] = 1'b1;
                  thread_track_banks[6][t_id] = 1'b1;
            end
              else if (i_p_addr[t_id][4:2] == 3'b111) begin
                  //banks_in_addr[0] = i_p_addr[t_id]; // WIll need to do this later
                  //thread_track_bank_7[t_id] = 1'b1;
                  thread_track_banks[7][t_id] = 1'b1;
              end
            end
          end
          INITIAL_ACCESS:begin
            //bank_states = INITIAL_ACCESS;
            o_m_valid = 1'b0;

            // Before Access
//            if (no_bank_misses) begin
              // Dont do anything, next clock cycle it will switch back to (Fetch from mem)
//            end else begin // Do logic to send requests to each bank (look through thread_track_bank regs)
            bank_has_access = 8'b0;
            for (t_id = 0; t_id <= `NT_M1; t_id = t_id + 1) begin
              for (bnk = 0; bnk < NUMBER_BANKS; bnk = bnk + 1) begin
                if(thread_track_banks[bnk][t_id] == 1'b1 && bank_has_access[bnk] == 1'b0) begin
                  bank_has_access[bnk] = 1'b1;
                  bank_access_data[bnk] = i_p_writedata[t_id];
                  bank_access_addr[bnk] = i_p_addr[t_id];
                  threads_in_banks[bnk] = t_id[1:0];
                end
              end 
                //if (banks_wb_needed[bnk]) begin // need to fix this for multiple misses
                  //o_m_read_or_write = 1'b0;
                  //o_m_addr = banks_wb_addr[bnk];
                  //o_m_valid = 1'b1;
                  //o_m_writedata = {banks_wb_data[bnk], 96'b0};
                //end
                //if(thread_track_bank_0[t_id] == 1'b1 && bank_has_access[0] == 1'b0) begin
                  //bank_has_access[0] = 1'b1;
                  //bank_access_data[0] = i_p_writedata[t_id];
                  //bank_access_addr[0] = i_p_addr[t_id];
                  //threads_in_banks[0] = t_id;
                //end
                // NEED TO UPDATE HITS (STORE IN THREADS_DONE)
            end 
              //num_misses = {28'b0, $countones(banks_missed)};
              //did_miss = (banks_missed == 4'hF);
              
//            end


          end
          INITIAL_PROCESSING:begin
            //bank_has_access = 8'b0;
            for (bnk = 0; bnk < NUMBER_BANKS; bnk = bnk + 1) begin
                if(banks_ready[bnk]) begin // FIX to handle hits
                  thread_done[threads_in_banks[bnk]] = 1'b1;
                  o_p_readdata[threads_in_banks[bnk]] = data_from_bank[bnk];
                  if(i_p_read_or_write == 1'b0) begin
                    o_p_readdata_valid[threads_in_banks[bnk]] = 1'b1;
                  end
                  thread_track_banks[bnk][threads_in_banks[bnk]] = 1'b0; // Update that this thread does not need to be serviced again
                end
            end             
            //banks_to_service_temp = !banks_ready; // These are clean misses
            for (bnk = 0; bnk < NUMBER_BANKS; bnk = bnk + 1) begin
              assign banks_to_service_temp[bnk] = (banks_ready[bnk] || (bank_has_access[bnk] == 0)) ? 1'b0 : 1'b1;
              assign banks_to_wb_temp[bnk] = (banks_wb_needed[bnk]);
              assign banks_all_help[bnk] = banks_to_service_temp[bnk] || banks_to_wb_temp[bnk];
            end 
          end
          CONTINUED_PROCESSING:begin
            //for (i = `NW-1; i >= 0; i = i - 1) begin
            //  if (thread_done[threads_in_banks[bnk]] == 1'b1) begin // Not sure about this logic
            //    //index = i[`NW_M1:0];
            //    banks_to_service_temp[i] = 1'b0;
            //    banks_to_wb_temp[i] = 1'b0; 
            // end
            //end
          end
          FETCH_FROM_MEM:begin 
            // NEED TO ADD LOGIC TO SEE IF MISSES GO TO SAME BLOCK
            index = 0;
            found = 0;
            for (i = `NW-1; i >= 0; i = i - 1) begin
              if (banks_to_service[i]) begin // Not sure about this logic
                //index = i[`NW_M1:0];
                index = i;
                found = 1;  
              end
            end
            if (found == 1) begin
              //banks_missed[index] = 0;
              //thread_done
              
              //thread_in_memory = threads_in_banks[index];
              //o_m_writedata = bank_access_data[index];
              banks_to_service_temp[index] = 0;
              o_m_addr = bank_access_addr[index];
              o_m_valid = 1'b1;
              o_m_read_or_write = 1'b0;
            end
            //bank_states = FETCH_FROM_MEM;
          end
          FETCH2:begin
            o_m_valid = 1'b0;
          end
          UPDATE_CACHE:begin
            for (bnk = 0; bnk < NUMBER_BANKS; bnk = bnk + 1) begin
              //if(thread_track_banks[bnk][t_id] == 1'b1 && bank_has_access[bnk] == 1'b0) begin
                bank_has_access[bnk] = 1'b1;
                //bank_access_data[bnk] = i_m_readdata[(bnk+1)*32 - 1:bnk*32];
                bank_access_addr[bnk] = o_m_addr;
                threads_in_banks[bnk] = t_id[1:0];
              //end
            end 
            //bank_access_data = i_m_readdata;
            rd_or_wr = 1'b1;
            //thread_done[thread_in_memory] = 1'b1;   // Removed, new cache style - Oct 21
            //o_p_readdata[thread_in_memory] = i_m_readdata[i_p_addr[thread_in_memory][9:5]]; // Removed, new cache style
          end
          DIRTY_EVICT_WB:begin       // this begininng logic should be added to dirty evict grab block     

            //thread_done[thread_in_memory] = 1'b1;
            o_m_valid = 1'b1;
          end
          DIRTY_EVICT_GRAB_BLOCK:begin
            index = 0;
            found = 0;
            for (i = `NW-1; i >= 0; i = i - 1) begin
              if (banks_to_wb_temp[i]) begin
                //index = i[`NW_M1:0];
                index = i;
                found = 1;
              end
            end
            if (found == 1) begin
              banks_to_wb_temp[index] = 0;
              for (i = `NW-1; i >= 0; i = i - 1) begin
                if (banks_to_wb_temp[i] && banks_wb_addr[index][31:7] == banks_wb_addr[i][31:7]) begin
                  //index = i[`NW_M1:0];
                  banks_to_wb_temp[i] = 0;
                end
              end
              //thread_done
              //thread_in_memory = threads_in_banks[index];
              //o_m_writedata[(bnk+1)*32 - 1:bnk*32] = banks_wb_data[index];
              o_m_addr = banks_wb_addr[index];
              o_m_read_or_write = 1'b1;
            end
            //for (bnk = 0; bnk < NUMBER_BANKS; bnk = bnk + 1) begin
              //o_m_writedata[(bnk+1)*32 - 1:bnk*32] = banks_wb_data[index];
            //end 
            // NEXT LINE CONTAINS DATA TO WB !!!! Think need to just change this to be read data and can remove banks_wb_data
            //o_m_writedata = {banks_wb_data[7],banks_wb_data[6],banks_wb_data[5],banks_wb_data[4],banks_wb_data[3],banks_wb_data[2],banks_wb_data[1],banks_wb_data[0]};
            //num_evictions_to_wb = {28'b0, $countones(banks_wb_needed)};
            rd_or_wr = 1'b0;
            for (bnk = 0; bnk < NUMBER_BANKS; bnk = bnk + 1) begin
              //if(thread_track_banks[bnk][t_id] == 1'b1 && bank_has_access[bnk] == 1'b0) begin
                bank_has_access[bnk] = 1'b1;
                bank_access_addr[bnk] = o_m_addr;
              //end
            end             
          end
          RE_ACCESS:begin
            //bank_states = INITIAL_ACCESS;
            o_m_valid = 1'b0;

            // Before Access
//            if (no_bank_misses) begin
              // Dont do anything, next clock cycle it will switch back to (Fetch from mem)
//            end else begin // Do logic to send requests to each bank (look through thread_track_bank regs)
            //bank_has_access = banks_all_help & !(banks_to_wb) & !(banks_to_service);
            for (t_id = 0; t_id <= `NT_M1; t_id = t_id + 1) begin
              for (bnk = 0; bnk < NUMBER_BANKS; bnk = bnk + 1) begin
                //bank_has_access[bnk] = banks_all_help[bnk] && !thread_done[threads_in_banks[bnk]]; // Not sure
                bank_has_access[bnk] = banks_all_help[bnk] && !thread_done[t_id]; // Not sure
                if(thread_track_banks[bnk][t_id] == 1'b1 && bank_has_access[bnk] == 1'b1) begin
                  //bank_has_access[bnk] = 1'b1;
                  bank_access_data[bnk] = i_p_writedata[t_id];
                  bank_access_addr[bnk] = i_p_addr[t_id];
                  threads_in_banks[bnk] = t_id[1:0];
                end
              end 
            end 



          end
          RE_ACCESS_PROCESSING:begin
            // After Access
            for (bnk = 0; bnk < NUMBER_BANKS; bnk = bnk + 1) begin
                if(banks_ready[bnk]) begin // FIX to handle hits
                  thread_done[threads_in_banks[bnk]] = 1'b1;
                  o_p_readdata[threads_in_banks[bnk]] = data_from_bank[bnk];
                  if(i_p_read_or_write == 1'b0) begin
                    o_p_readdata_valid[threads_in_banks[bnk]] = 1'b1;
                  end
                  thread_track_banks[bnk][threads_in_banks[bnk]] = 1'b0; // Update that this thread does not need to be serviced again
                  // Added Oct 21
                  banks_to_service_temp[bnk] = 1'b0;
                  banks_to_wb_temp[bnk] = 1'b0;
                end
            end 
          end

        endcase      
    end

    always @(posedge clk) begin
      banks_to_service <= banks_to_service_temp;
      banks_to_wb <= banks_to_wb_temp;
    end


  genvar bank_id;
  generate
    for (bank_id = 0; bank_id < NUMBER_BANKS; bank_id = bank_id + 1)
      begin
        //VX_alu vx_alu(
          // .in_reg_data   (in_reg_data[1:0]),
        //  .in_1          (in_a_reg_data[index_out_reg]),
        //  .in_2          (in_b_reg_data[index_out_reg]),
        //  .in_rs2_src    (in_rs2_src),
        //  .in_itype_immed(in_itype_immed),
        //  .in_upper_immed(in_upper_immed),
        //  .in_alu_op     (in_alu_op),
        //  .in_csr_data   (in_csr_data),
        //  .in_curr_PC    (in_curr_PC),
        //  .out_alu_result(VX_exe_mem_req.alu_result[index_out_reg])
        //);
//        bank VX_banks(
//          .clk              (clk),
//          .rst              (rst),
//          //.state            (bank_states[bank_id]), 
//          .state            (state),
//          .read_or_write    (rd_or_wr),
//          //.index            (correct_index),
//          //.tag              (correct_tag),
//          .addr             (bank_access_addr[bank_id]),
//          .writedata        (bank_access_data[bank_id]),
//          .fetched_write_data(i_m_readdata[(bank_id+1)*32-1 -: 32]),
//          .valid            (bank_has_access[bank_id]),
//          .readdata         (data_from_bank[bank_id]),
//          .miss_cache       (banks_missed[bank_id]),
//          .w2m_needed       (banks_wb_needed[bank_id]),
//          .w2m_addr         (banks_wb_addr[bank_id]),
//          .e_data           (o_m_writedata[(bank_id+1)*32-1 -: 32]),
//          //.w2m_data         (banks_wb_data[bank_id]),
//          .ready            (banks_ready[bank_id])
//          //.valid_data        (valid_in_set)
//          //.read_miss    (read_miss)
//        );

        VX_Cache_Bank bank_structure (
          .clk                        (clk),
          .state                      (state),
          .read_or_write              (rd_or_wr),
          .valid_in                   (bank_has_access[bank_id]),
          .actual_index               (bank_access_addr[bank_id][14:7]), // fix when size changes
          .o_tag                      (bank_access_addr[bank_id][31:15]), // fix when size changes
          .block_offset               (bank_access_addr[bank_id][6:5]),
          .writedata                  (bank_access_data[bank_id]),
          //.fetched_writedata         (i_m_readdata[(bank_id+1)*32-1 -: 32]),
          .fetched_writedata         (i_m_readdata[bank_id[3:0]]),
          .readdata                   (data_from_bank[bank_id]),
          .hit                        (banks_ready[bank_id]),
          //.miss                       (banks_missed[bank_id]),
          .eviction_wb                (banks_wb_needed[bank_id]),
          .eviction_addr              (banks_wb_addr[bank_id]),
          //.data_evicted               (o_m_writedata[(bank_id+1)*32-1 -: 32])
          .data_evicted               (o_m_writedata[bank_id[3:0]])
        );

      end
  endgenerate

    //end

endmodule





