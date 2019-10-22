// To Do: Change way_id_out to an internal register which holds when in between access and finished. 
//        Also add a bit about wheter the "Way ID" is valid / being held or if it is just default
//        Also make sure all possible output states are transmitted back to the bank correctly

`include "VX_define.v"
module cache_set(clk,
           rst,
           // These next 4 are possible modes that the Set could be in, I am making them 4 different variables for indexing purposes
           access, // First 
           find_evict, 
           write_from_mem,
           idle, 
          // entry,
           o_tag,
           writedata,
           //byte_en,
           write,
           //word_en,
           //way_id_in,
           //way_id_out,
           readdata,
           //wb_addr,
           hit,
           eviction_wb,
           eviction_tag,
           //eviction_data,
           //modify,
           miss
           //valid_data
           //read_miss
           );

    parameter cache_entry = 14;
    parameter ways_per_set = 4;

    input wire                    clk, rst;
    input wire                    access;
    input wire                    find_evict;
    input wire                    write_from_mem;
    input wire                    idle;
    //input wire [cache_entry-1:0]  entry;
    input wire [1:0] o_tag;
    input wire [31:0]          writedata;
    //input wire [3:0]          byte_en;
    input wire                  write; // 0 == False
    //input wire [3:0]              word_en;
    //input wire                read_miss;
    //input wire [1:0]        way_id_in;
    //output reg [1:0]        way_id_out;
    output reg [31:0]       readdata;
    //output reg [3:0]        hit;
    output reg hit;
    output reg              miss;
    output wire              eviction_wb;
    output wire [1:0]             eviction_tag;
    reg [31:0]            eviction_data;
    //output wire [22:0]          wb_addr;
    //output wire             modify, valid_data;



    //wire [2:0]    i_tag;
    //wire                   dirty;
    //wire [24-cache_entry:0]    write_tag_data;

    // Table for one set
    reg [2:0] counter; // Determines which to evict
    reg valid [ways_per_set-1:0];
    reg [1:0] tag [ways_per_set-1:0];
    reg clean [ways_per_set-1:0];
    reg [31:0] data [ways_per_set-1:0];


    assign eviction_wb = miss && clean[counter[1:0]] != 1'b1 && valid[counter[1:0]] == 1'b1;
    assign eviction_tag = tag[counter[1:0]];
    //assign eviction_data = data[counter[1:0]];
    //assign hit = valid_data && (o_tag == i_tag);
    //assign modify = valid_data && (o_tag != i_tag) && dirty;
    //assign miss = !valid_data || ((o_tag != i_tag) && !dirty);

    //assign wb_addr = {i_tag, entry};
    always @(posedge clk) begin
      if (rst) begin 

      end
      if (find_evict) begin
        if (tag[0] == o_tag && valid[0]) begin
          readdata <= data[0];
        end else if (tag[1] == o_tag && valid[1]) begin
          readdata <= data[1];
         end else if (tag[2] == o_tag && valid[2]) begin
          readdata <= data[2];
        end else if (tag[3] == o_tag && valid[3]) begin
          readdata <= data[3];
        end
      end else if (access) begin 
      //tag[`NT_M1:0] <= i_p_addr[`NT_M1:0][13:12]; 
        counter <= ((counter + 1) ^ 3'b100); // Counter determining which to evict in the event of miss only increment when miss !!! NEED TO FIX LOGIC
        // Hit in First Column
        if (tag[0] == o_tag && valid[0]) begin 
          if (write == 1'b0) begin  // if it is a read
            if (clean[0] == 1'b1 ) begin
              //hit <= 4'b0001;
              hit <= 1'b1;
              readdata <= data[0];
              miss <= 1'b0;
            end else begin
              //hit <= 4'b0000;  // SHOULD PROBABLY TRACK WHERE THIS MISS IS IN A DIFFERENT VARIABLE
              hit <= 1'b0;
              readdata <= 32'b0;
              miss <= 1'b1;
            end
          end else if (write == 1'b1) begin
            data[0] <= writedata;
            clean[0] <= 1'b0;
            //hit <= 4'b0001;
            hit <= 1'b1;
          end
        end 
        // Hit in Second Column
        else if (tag[1] == o_tag && valid[1]) begin 
          if (write == 1'b0) begin  // if it is a read
            if (clean[1] == 1'b1 ) begin
              //hit <= 4'b0010;
              hit <= 1'b1;
              readdata <= data[1];
              miss <= 1'b0;
            end else begin
              //hit <= 4'b0000;
              hit <= 1'b0;
              readdata <= 32'b0;
              miss <= 1'b1;
            end
          end else if (write == 1'b1) begin
            data[1] <= writedata;
            clean[1] <= 1'b0;
            //hit <= 4'b0010;
            hit <= 1'b1;
          end
        end 
        // Hit in Third Column
        else if (tag[2] == o_tag && valid[2]) begin 
          if (write == 1'b0) begin  // if it is a read
            if (clean[2] == 1'b1 ) begin
              //hit <= 4'b0100;
              hit <= 1'b1;
              readdata <= data[2];
              miss <= 1'b0;
            end else begin
              //hit <= 4'b0000;
              hit <= 1'b0;
              readdata <= 32'b0;
              miss <= 1'b1;
            end
          end else if (write == 1'b1) begin
            data[2] <= writedata;
            clean[2] <= 1'b0;
            //hit <= 4'b0100;
            hit <= 1'b1;
          end
        end 
        // Hit in Fourth Column
        else if (tag[3] == o_tag && valid[3]) begin 
          if (write == 1'b0) begin  // if it is a read
            if (clean[3] == 1'b1 ) begin
              //hit <= 4'b1000;
              hit <= 1'b1;
              readdata <= data[3];
              miss <= 1'b0;
            end else begin
              //hit <= 4'b0000;
              hit <= 1'b0;
              readdata <= 32'b0;
              miss <= 1'b1;
            end
          end else if (write == 1'b1) begin
            data[3] <= writedata;
            clean[3] <= 1'b0;
            //hit <= 4'b1000;
            hit <= 1'b1;
          end
        end 
        // Miss
        else begin 
          //way_id_out <= counter;
          miss <= 1'b1;
          if (write == 1'b0) begin  // Read Miss
            clean[counter[1:0]] <= 1'b1;
            data[counter[1:0]] <= 32'h7FF; // FIX WITH ACTUAL MEMORY ACCESS
          end else if (write == 1'b1) begin // Write Miss
            clean[counter[1:0]] <= 1'b1;
            data[counter[1:0]] <= writedata;
          end
        end
      
      end
      if (write_from_mem) begin
        tag[counter[1:0]] <= o_tag;
        valid[counter[1:0]] <= 1'b1;
        hit <= 1'b1;
          if (write == 1'b0) begin  // Read Miss
            clean[counter[1:0]] <= 1'b1;
            data[counter[1:0]] <= 32'h7FF; // FIX WITH ACTUAL MEMORY ACCESS
          end else if (write == 1'b1) begin // Write Miss
            clean[counter[1:0]] <= 1'b0;
            data[counter[1:0]] <= writedata;
          end        
      end
      if (idle) begin  // Set "way" register equal to invalid value
        hit <= 1'b1; // set to know it is ready
        miss <= 1'b0;
        readdata <= 32'hFFFFFFFF;
      end
      if (find_evict) begin  // Keep "way" value the same !!!! Fix. Need to send back data with matching tag. Also need to ensure evicted data doesnt get lost
        if (tag[3] == o_tag && valid[3]) begin 
          readdata <= data[3];
        end else if (tag[1] == o_tag && valid[1]) begin
          readdata <= data[1];
        end else if (tag[2] == o_tag && valid[2]) begin
          readdata <= data[2];
        end else if (tag[0] == o_tag && valid[0]) begin
          readdata <= data[0];
        end else begin
          readdata <= eviction_data;
        end
        hit <= 1'b1;
        miss <= 1'b0;
      end
      counter <= ((counter + 1) ^ 3'b100); // Counter determining which to evict in the event of miss only increment when miss !!! NEED TO FIX LOGIC
      eviction_data <= data[counter[1:0]];
    end

endmodule