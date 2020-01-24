

`include "../VX_define.v"

module VX_cache_data_per_index
    #(
        parameter CACHE_WAYS          = 1,
        parameter NUM_IND             = 8,
        parameter CACHE_WAY_INDEX     = 1, 
        parameter NUM_WORDS_PER_BLOCK = 4,
        parameter TAG_SIZE_START      = 0,
        parameter TAG_SIZE_END        = 16,
        parameter IND_SIZE_START      = 0,
        parameter IND_SIZE_END        = 7
    )
    (
	input wire clk,    // Clock
  input wire rst,
  input wire valid_in,
  input wire [3:0] state,
	// Addr
	input wire[IND_SIZE_END:IND_SIZE_START] 			addr,
	// WE
	input wire[NUM_WORDS_PER_BLOCK-1:0][3:0]   	we,
	input wire                             		evict,
	input wire[CACHE_WAY_INDEX-1:0]	   		way_to_update,
	// Data
	input wire[NUM_WORDS_PER_BLOCK-1:0][31:0] 	data_write, // Update Data
	input wire[TAG_SIZE_END:TAG_SIZE_START]             tag_write,


	output wire[TAG_SIZE_END:TAG_SIZE_START]           	tag_use,
	output wire[NUM_WORDS_PER_BLOCK-1:0][31:0] 	data_use,
	output wire                                 valid_use,
	output wire                                 dirty_use
	
);
    //localparam NUMBER_BANKS         = CACHE_BANKS;
    //localparam CACHE_BLOCK_PER_BANK = (CACHE_BLOCK / CACHE_BANKS);
    // localparam NUM_WORDS_PER_BLOCK  = CACHE_BLOCK / (CACHE_BANKS*4);
    //localparam NUMBER_INDEXES       = `DCACHE_NUM_IND;

    wire [CACHE_WAYS-1:0][TAG_SIZE_END:TAG_SIZE_START]          	tag_use_per_way;
    wire [CACHE_WAYS-1:0][NUM_WORDS_PER_BLOCK-1:0][31:0] 	data_use_per_way;
    wire [CACHE_WAYS-1:0] 									valid_use_per_way;
    wire [CACHE_WAYS-1:0] 									dirty_use_per_way;
    wire [CACHE_WAYS-1:0] 									hit_per_way;
    // reg  [CACHE_WAY_INDEX-1:0] 		eviction_way_index;
    wire [CACHE_WAYS-1:0][NUM_WORDS_PER_BLOCK-1:0][3:0] 	we_per_way;
    wire [CACHE_WAYS-1:0][NUM_WORDS_PER_BLOCK-1:0][31:0] 	data_write_per_way;
    wire [CACHE_WAYS-1:0] 									write_from_mem_per_way;
    wire invalid_found;

    wire [CACHE_WAY_INDEX-1:0]  way_index;
    wire [CACHE_WAY_INDEX-1:0] invalid_index;


    localparam CACHE_IDLE    = 0; // Idle
    localparam SEND_MEM_REQ  = 1; // Write back this block into memory
    localparam RECIV_MEM_RSP = 2;

	 generate
    if(CACHE_WAYS != 1) begin
        VX_generic_priority_encoder #(.N(CACHE_WAYS)) valid_index
        (
          .valids(~valid_use_per_way),
          .index (invalid_index),
          .found (invalid_found)
        );

        VX_generic_priority_encoder #(.N(CACHE_WAYS)) way_indexing
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
	 endgenerate




    // wire hit       = |hit_per_way;
    // wire miss      = ~hit;
    // wire update    = |we && !miss;
    // wire valid     = &valid_use_per_way;

    wire[CACHE_WAY_INDEX-1:0] way_use_Qual;

    assign way_use_Qual = (state != CACHE_IDLE) ? way_to_update : way_index;

    assign tag_use   = tag_use_per_way[way_use_Qual];
    assign data_use  = data_use_per_way[way_use_Qual];
    assign valid_use = valid_use_per_way[way_use_Qual];
    assign dirty_use = dirty_use_per_way[way_use_Qual];

    // assign tag_use   = hit ? tag_use_per_way[way_index]   : (valid ? tag_use_per_way[eviction_way_index] : (invalid_found ? tag_use_per_way[invalid_index] : 0));
    // assign data_use  = hit ? data_use_per_way[way_index]  : (valid ? data_use_per_way[eviction_way_index] : (invalid_found ? data_use_per_way[invalid_index] : 0));
    // assign valid_use = hit ? valid_use_per_way[way_index] : (valid ? valid_use_per_way[eviction_way_index] : (invalid_found ? valid_use_per_way[invalid_index] : 0));
    // assign dirty_use = hit ? dirty_use_per_way[way_index] : (valid ? dirty_use_per_way[eviction_way_index] : (invalid_found ? dirty_use_per_way[invalid_index] : 0));



    genvar ways;
	 generate
	  for(ways=0; ways < CACHE_WAYS; ways = ways + 1) begin : each_way


      assign hit_per_way[ways]            = ((valid_use_per_way[ways] == 1'b1) &&  (tag_use_per_way[ways] == tag_write)) ? 1'b1 : 0;
      

      assign write_from_mem_per_way[ways] = evict && (ways == way_use_Qual);
      assign we_per_way[ways]             = (ways == way_use_Qual) ? (we) : 0;
      assign data_write_per_way[ways]     = data_write;


	    // assign hit_per_way[ways]            = ((valid_use_per_way[ways] == 1'b1) &&  (tag_use_per_way[ways] == tag_write)) ? 1'b1 : 0;

	    // assign we_per_way[ways]             = (evict == 1'b1) || (update == 1'b1) ? ((ways == way_use_Qual) ? (we) : 0) : 0;
	    // assign data_write_per_way[ways]     = (evict == 1'b1) || (update == 1'b1) ? ((ways == way_use_Qual) ? data_write : 0) : 0;
	    // assign write_from_mem_per_way[ways] = (evict == 1'b1) ? ((ways == way_use_Qual) ? 1 : 0) : 0;

	    VX_cache_data #(
	           .NUM_IND             (NUM_IND),
             .NUM_WORDS_PER_BLOCK (NUM_WORDS_PER_BLOCK),
             .TAG_SIZE_START      (TAG_SIZE_START),
             .TAG_SIZE_END        (TAG_SIZE_END),
             .IND_SIZE_START      (IND_SIZE_START),
             .IND_SIZE_END        (IND_SIZE_END)) data_structures(
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
	  endgenerate

    // always @(posedge clk or posedge rst) begin
    //   if (rst) begin
    //     eviction_way_index <= 0;
    //   end else begin
    //     // if((miss && dirty_use && valid_use && !evict && valid_in)) begin // can be either evict or invalid cache entries
    //     if((state == SEND_MEM_REQ)) begin // can be either evict or invalid cache entries
    //  			if((eviction_way_index+1) == CACHE_WAYS) begin
    //  				eviction_way_index <= 0;
    //  			end else begin
    //  				eviction_way_index <= (eviction_way_index + 1);
    //  			end
    //   	end
    //   end
    // end

endmodule
