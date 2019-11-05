

`include "../VX_define.v"

module VX_cache_data_per_index
    /*#(
      parameter CACHE_SIZE     = 4096, // Bytes
      parameter CACHE_WAYS     = 1,
      parameter CACHE_BLOCK    = 128, // Bytes
      parameter CACHE_BANKS    = 8,
      parameter NUM_WORDS_PER_BLOCK = CACHE_BLOCK / (CACHE_BANKS*4)
    )*/
    (
	input wire clk,    // Clock
  input wire rst,
  input wire valid_in,
	// Addr
	input wire[`DCACHE_IND_SIZE_RNG] 			addr,
	// WE
	input wire[`DCACHE_NUM_WORDS_PER_BLOCK-1:0][3:0]   	we,
	input wire                             		evict,
	input wire[`DCACHE_WAY_INDEX-1:0]	   		way_to_update,
	// Data
	input wire[`DCACHE_NUM_WORDS_PER_BLOCK-1:0][31:0] 	data_write, // Update Data
	input wire[`DCACHE_TAG_SIZE_RNG]             tag_write,


	output wire[`DCACHE_TAG_SIZE_RNG]           	tag_use,
	output wire[`DCACHE_NUM_WORDS_PER_BLOCK-1:0][31:0] 	data_use,
	output wire                                 valid_use,
	output wire                                 dirty_use, 
	output wire[`DCACHE_WAY_INDEX-1:0]			way
	
);
    //localparam NUMBER_BANKS         = CACHE_BANKS;
    //localparam CACHE_BLOCK_PER_BANK = (CACHE_BLOCK / CACHE_BANKS);
    // localparam NUM_WORDS_PER_BLOCK  = CACHE_BLOCK / (CACHE_BANKS*4);
    //localparam NUMBER_INDEXES       = `DCACHE_NUM_IND;

    wire [`DCACHE_WAYS-1:0][`DCACHE_TAG_SIZE_RNG]          	tag_use_per_way;
    wire [`DCACHE_WAYS-1:0][`DCACHE_NUM_WORDS_PER_BLOCK-1:0][31:0] 	data_use_per_way;
    wire [`DCACHE_WAYS-1:0] 									valid_use_per_way;
    wire [`DCACHE_WAYS-1:0] 									dirty_use_per_way;
    wire [`DCACHE_WAYS-1:0] 									hit_per_way;
    reg  [`DCACHE_NUM_IND-1:0][`DCACHE_WAY_INDEX-1:0] 		eviction_way_index;
    wire [`DCACHE_WAYS-1:0][`DCACHE_NUM_WORDS_PER_BLOCK-1:0][3:0] 	we_per_way;
    wire [`DCACHE_WAYS-1:0][`DCACHE_NUM_WORDS_PER_BLOCK-1:0][31:0] 	data_write_per_way;
    wire [`DCACHE_WAYS-1:0] 									write_from_mem_per_way;
    wire invalid_found;

    wire [`DCACHE_WAY_INDEX-1:0]  way_index;
    wire [`DCACHE_WAY_INDEX-1:0] invalid_index;


    if(`DCACHE_WAYS != 1) begin
        VX_generic_priority_encoder #(.N(`DCACHE_WAYS)) valid_index
        (
          .valids(~valid_use_per_way),
          .index (invalid_index),
          .found (invalid_found)
        );

        VX_generic_priority_encoder #(.N(`DCACHE_WAYS)) way_indexing
        (
          .valids(hit_per_way),
          .index (way_index),
          .found ()
        );    
    end 
    else begin
      assign  way_index = 0;
      assign invalid_found = (valid_use_per_way == 1'b0) ? 1 : 0;
      assign invalid_index = 0;
    end




    wire hit       = |hit_per_way;
    wire miss      = ~hit;
    wire update    = |we && !miss;
    wire valid     = &valid_use_per_way;

	  assign way 		   = hit ? way_index : (valid ? eviction_way_index[addr] : (invalid_found ? invalid_index : 0));
    assign tag_use   = hit ? tag_use_per_way[way_index]   : (valid ? tag_use_per_way[eviction_way_index[addr]] : (invalid_found ? tag_use_per_way[invalid_index] : 0));
    assign data_use  = hit ? data_use_per_way[way_index]  : (valid ? data_use_per_way[eviction_way_index[addr]] : (invalid_found ? data_use_per_way[invalid_index] : 0));
    assign valid_use = hit ? valid_use_per_way[way_index] : (valid ? valid_use_per_way[eviction_way_index[addr]] : (invalid_found ? valid_use_per_way[invalid_index] : 0));
    assign dirty_use = hit ? dirty_use_per_way[way_index] : (valid ? dirty_use_per_way[eviction_way_index[addr]] : (invalid_found ? dirty_use_per_way[invalid_index] : 0));



    genvar ways;
	  for(ways=0; ways < `DCACHE_WAYS; ways = ways + 1) begin

	    assign hit_per_way[ways]            = ((valid_use_per_way[ways] == 1'b1) &&  (tag_use_per_way[ways] == tag_write)) ? 1'b1 : 0;
	    assign we_per_way[ways]             = (evict == 1'b1) || (update == 1'b1) ? ((ways == way_to_update) ? (we) : 0) : 0;
	    assign data_write_per_way[ways]     = (evict == 1'b1) || (update == 1'b1) ? ((ways == way_to_update) ? data_write : 0) : 0;
	    assign write_from_mem_per_way[ways] = (evict == 1'b1) ? ((ways == way_to_update) ? 1 : 0) : 0;

	    /*VX_cache_data #(
	           .CACHE_SIZE(`CACHE_SIZE),
	           .CACHE_WAYS(`DCACHE_WAYS),
	           .CACHE_BLOCK(`CACHE_BLOCK),
	           .CACHE_BANKS(`CACHE_BANKS)) data_structures(*/
      VX_cache_data data_structures(
	        .clk       (clk),
          .rst       (rst),
	        // Inputs
	        .addr      (addr),
	        .we        (we_per_way[ways]),
	        .evict     (write_from_mem_per_way[ways]),
	        .data_write(data_write_per_way[ways]),
	        .tag_write (tag_write),
	        // Outputs
	        .tag_use   (tag_use_per_way[ways]),
	        .data_use  (data_use_per_way[ways]),
	        .valid_use (valid_use_per_way[ways]),
	        .dirty_use (dirty_use_per_way[ways])
	    );
	  end

    always @(posedge clk or posedge rst) begin
      if (rst) begin
        eviction_way_index <= 0;
      end else begin
      	if(miss && dirty_use && valid_use && !evict && valid_in) begin // can be either evict or invalid cache entries
     			if((eviction_way_index[addr]+1) == `DCACHE_WAYS) begin
     				eviction_way_index[addr] <= 0;
     			end else begin
     				eviction_way_index[addr] <= (eviction_way_index[addr] + 1);
     			end
      	end
      end
    end

endmodule
