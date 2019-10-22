// To Do: Change way_id_out to an internal register which holds when in between access and finished. 
//        Also add a bit about wheter the "Way ID" is valid / being held or if it is just default
//        Also make sure all possible output states are transmitted back to the bank correctly

`include "VX_define.v"
module VX_Cache_Block_DM(clk,
           rst,
           // These next 4 are possible modes that the Set could be in, I am making them 4 different variables for indexing purposes
           access, // First 
           find_evict, 
           write_from_mem,
           idle, 
          // entry,
           o_tag,
           block_offset,
           writedata,
           //byte_en,
           write,
           fetched_writedata,
           //word_en,
           //way_id_in,
           //way_id_out,
           readdata,
           //wb_addr,
           hit,
           eviction_wb,
           eviction_tag,
           evicted_data,
           //modify,
           miss
           //valid_data
           //read_miss
           );

    parameter cache_entry = 14;
    parameter ways_per_set = 4;
    parameter Number_Blocks = 32;

    input wire                    clk, rst;
    input wire                    access;
    input wire                    find_evict;
    input wire                    write_from_mem;
    input wire                    idle;
    //input wire [cache_entry-1:0]  entry;
    input wire [21:0] o_tag;
    input wire [4:0] block_offset;
    input wire [31:0]          writedata;
    //input wire [3:0]          byte_en;
    input wire                  write; // 0 == False
    input wire [31:0][31:0]                 fetched_writedata;
    //input wire [3:0]              word_en;
    //input wire                read_miss;
    //input wire [1:0]        way_id_in;
    //output reg [1:0]        way_id_out;
    //output reg [31:0]       readdata;
    output wire [31:0]        readdata;
    //output reg hit;
    output wire hit;
    output reg              miss;
    output wire              eviction_wb;
    output wire [21:0]             eviction_tag;
    output wire [31:0][31:0]      evicted_data;
    //reg [31:0]            eviction_data;
    //output wire [22:0]          wb_addr;
    //output wire             modify, valid_data;



    //wire [2:0]    i_tag;
    //wire                   dirty;
    //wire [24-cache_entry:0]    write_tag_data;

    // Table for one set
    //reg [2:0] counter; // Determines which to evict
    reg valid;
    reg [21:0] tag;

    reg clean;


    //reg [31:0] data[31:0]; 
    reg [31:0] data[31:0]; 

    integer j;

    //  WS   AW             BS
    //reg[3:0][31:0] some_data[5:0]; // before variable name is width, after name is height

    //wire blockNun;
    //wire WordNumWIthinABlock;

    //ddata[31:0] =some_data[blockNun][WordNumWIthinABlock]


    assign eviction_wb = miss && clean != 1'b1 && valid == 1'b1;
    assign eviction_tag = tag;
    assign readdata = (access && !write && tag == o_tag && valid) ? data[0] : 32'b0; // Fix with actual data
    assign hit = (access && !write && tag == o_tag && valid) ? 1'b1 : 1'b0;
    //assign evicted_data = (eviction_wb ) ? data : 0; 
    genvar k;
    for (k = 0; k < Number_Blocks; k = k + 1) begin
      assign evicted_data[k] = (eviction_wb) ? data[k] : 32'b0;
      //data[j] <= fetched_writedata[(j+1) * 32 - 1 -: 32];
    end
    //assign eviction_data = data[counter[1:0]];
    //assign hit = valid_data && (o_tag == i_tag);
    //assign modify = valid_data && (o_tag != i_tag) && dirty;
    //assign miss = !valid_data || ((o_tag != i_tag) && !dirty);

    //assign wb_addr = {i_tag, entry};
    always @(posedge clk) begin
      if (rst) begin 

      end
      if (find_evict) begin
        if (tag == o_tag && valid) begin
          //readdata <= data;
          // evicted_data <= data;
        end
      end else if (access) begin 
      // Hit in First Column
        if (tag == o_tag && valid) begin 
          if (write == 1'b0) begin  // if it is a read
            if (clean == 1'b1 ) begin
              //hit <= 1'b1;
              //readdata <= data;
              miss <= 1'b0;
            end else begin
              //hit <= 1'b0;
              //readdata <= 32'b0;
              miss <= 1'b1;
            end
          end else if (write == 1'b1) begin
            //for (j = 0; j < Number_Blocks; j = j + 1) begin
              //data[j] <= fetched_writedata[(j+1) * 32 - 1 -: 32];
            //end
            data[block_offset] <= writedata;
            clean <= 1'b0;
            //hit <= 1'b1;
          end
        end 
        // Miss
        else begin 
          //way_id_out <= counter;
          miss <= 1'b1;
          if (write == 1'b0) begin  // Read Miss
            clean <= 1'b1;
            //data <= 0; // FIX WITH ACTUAL MEMORY ACCESS
            for (j = 0; j < Number_Blocks; j = j + 1) begin
              data[j] <= 32'b0;
            end
          end else if (write == 1'b1) begin // Write Miss
            clean <= 1'b1;
            data[block_offset] <= writedata;
            //for (j = 0; j < Number_Blocks; j = j + 1) begin
              //data[j] <= fetched_writedata[(j+1) * 32 - 1 -: 32];
            //end
          end
        end
      
      end
      if (write_from_mem) begin
        tag <= o_tag;
        valid <= 1'b1;
        //hit <= 1'b1;
          if (write == 1'b0) begin  // Read Miss
            clean <= 1'b1;
            //data <= 0; // FIX WITH ACTUAL MEMORY ACCESS
            for (j = 0; j < Number_Blocks; j = j + 1) begin
              data[j] <= 32'b0;
            end
          end else if (write == 1'b1) begin // Write Miss
            clean <= 1'b0;
            //data <= fetched_writedata;
            for (j = 0; j < Number_Blocks; j = j + 1) begin
              //data[j] <= fetched_writedata[(j+1) * 32 - 1 -: 32];
              data[j] <= fetched_writedata[j];
            end
          end        
      end
      if (idle) begin  // Set "way" register equal to invalid value
        //hit <= 1'b1; // set to know it is ready
        miss <= 1'b0;
        //readdata <= 32'hFFFFFFFF;
      end
      if (find_evict) begin  // Keep "way" value the same !!!! Fix. Need to send back data with matching tag. Also need to ensure evicted data doesnt get lost
        if (tag == o_tag && valid) begin 
          //readdata <= data;
        end
        //hit <= 1'b1;
        miss <= 1'b0;
      end
      //eviction_data <= data;
    end

endmodule