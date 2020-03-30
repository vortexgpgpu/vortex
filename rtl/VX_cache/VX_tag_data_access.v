`include "VX_cache_config.v"

module VX_tag_data_access
	#(
	// Size of cache in bytes
	parameter CACHE_SIZE_BYTES              = 1024, 
	// Size of line inside a bank in bytes
	parameter BANK_LINE_SIZE_BYTES          = 16, 
	// Number of banks {1, 2, 4, 8,...}
	parameter NUMBER_BANKS                  = 8, 
	// Size of a word in bytes
	parameter WORD_SIZE_BYTES               = 4, 
	// Number of Word requests per cycle {1, 2, 4, 8, ...}
	parameter NUMBER_REQUESTS               = 2, 
	// Number of cycles to complete stage 1 (read from memory)
	parameter STAGE_1_CYCLES                = 2, 
    // Function ID, {Dcache=0, Icache=1, Sharedmemory=2}
    parameter FUNC_ID                       = 0,

// Queues feeding into banks Knobs {1, 2, 4, 8, ...}

	// Core Request Queue Size
	parameter REQQ_SIZE                     = 8, 
	// Miss Reserv Queue Knob
	parameter MRVQ_SIZE                     = 8, 
	// Dram Fill Rsp Queue Size
	parameter DFPQ_SIZE                     = 2, 
	// Snoop Req Queue
	parameter SNRQ_SIZE                     = 8, 

// Queues for writebacks Knobs {1, 2, 4, 8, ...}
	// Core Writeback Queue Size
	parameter CWBQ_SIZE                     = 8, 
	// Dram Writeback Queue Size
	parameter DWBQ_SIZE                     = 4, 
	// Dram Fill Req Queue Size
	parameter DFQQ_SIZE                     = 8, 
	// Lower Level Cache Hit Queue Size
	parameter LLVQ_SIZE                     = 16, 

 	// Fill Invalidator Size {Fill invalidator must be active}
 	parameter FILL_INVALIDAOR_SIZE          = 16, 

// Dram knobs
	parameter SIMULATED_DRAM_LATENCY_CYCLES = 10


	)
	(
	input  wire                            clk,
	input  wire                            reset,
	input  wire                            stall,
	input  wire                            is_snp_st1e,
	// Initial Reading
	input  wire[31:0]                      readaddr_st10, 

	// Write/Read Logic
	input  wire                            valid_req_st1e,
	input  wire                            writefill_st1e,
	input  wire[31:0]                      writeaddr_st1e,
	input  wire[`WORD_SIZE_RNG]            writeword_st1e,
	input  wire[`DBANK_LINE_SIZE_RNG][31:0] writedata_st1e,
	input  wire[2:0]                       mem_write_st1e,
	input  wire[2:0]                       mem_read_st1e, 

	output wire[`WORD_SIZE_RNG]             readword_st1e,
	output wire[`DBANK_LINE_SIZE_RNG][31:0] readdata_st1e,
	output wire[`TAG_SELECT_SIZE_RNG]      readtag_st1e,
	output wire                            miss_st1e,
	output wire                            dirty_st1e,
	output wire                            fill_saw_dirty_st1e
	
);


	reg[`DBANK_LINE_SIZE_RNG][31:0] readdata_st[STAGE_1_CYCLES-1:0];

	reg                            read_valid_st1c[STAGE_1_CYCLES-1:0];
	reg                            read_dirty_st1c[STAGE_1_CYCLES-1:0];
	reg[`TAG_SELECT_SIZE_RNG]      read_tag_st1c  [STAGE_1_CYCLES-1:0];
	reg[`DBANK_LINE_SIZE_RNG][31:0] read_data_st1c [STAGE_1_CYCLES-1:0];


	wire                            qual_read_valid_st1;
	wire                            qual_read_dirty_st1;
	wire[`TAG_SELECT_SIZE_RNG]      qual_read_tag_st1;
	wire[`DBANK_LINE_SIZE_RNG][31:0] qual_read_data_st1;

	wire                            use_read_valid_st1e;
	wire                            use_read_dirty_st1e;
	wire[`TAG_SELECT_SIZE_RNG]      use_read_tag_st1e;
	wire[`DBANK_LINE_SIZE_RNG][31:0] use_read_data_st1e;
	wire[`DBANK_LINE_SIZE_RNG][3:0]  use_write_enable;
	wire[`DBANK_LINE_SIZE_RNG][31:0] use_write_data;

	wire sw, sb, sh;

	wire real_writefill = writefill_st1e && ((valid_req_st1e && !use_read_valid_st1e) || (valid_req_st1e && use_read_valid_st1e && (writeaddr_st1e[`TAG_SELECT_ADDR_RNG] != use_read_tag_st1e)));


	wire fill_sent;
	wire invalidate_line;
	VX_tag_data_structure  #(
        .CACHE_SIZE_BYTES             (CACHE_SIZE_BYTES),
        .BANK_LINE_SIZE_BYTES         (BANK_LINE_SIZE_BYTES),
        .NUMBER_BANKS                 (NUMBER_BANKS),
        .WORD_SIZE_BYTES              (WORD_SIZE_BYTES),
        .NUMBER_REQUESTS              (NUMBER_REQUESTS),
        .STAGE_1_CYCLES               (STAGE_1_CYCLES),
        .FUNC_ID                      (FUNC_ID),
        .REQQ_SIZE                    (REQQ_SIZE),
        .MRVQ_SIZE                    (MRVQ_SIZE),
        .DFPQ_SIZE                    (DFPQ_SIZE),
        .SNRQ_SIZE                    (SNRQ_SIZE),
        .CWBQ_SIZE                    (CWBQ_SIZE),
        .DWBQ_SIZE                    (DWBQ_SIZE),
        .DFQQ_SIZE                    (DFQQ_SIZE),
        .LLVQ_SIZE                    (LLVQ_SIZE),
        .FILL_INVALIDAOR_SIZE         (FILL_INVALIDAOR_SIZE),
        .SIMULATED_DRAM_LATENCY_CYCLES(SIMULATED_DRAM_LATENCY_CYCLES)
        )
		VX_tag_data_structure
		(
		.clk         (clk),
		.reset       (reset),

		.read_addr   (readaddr_st10),
		.read_valid  (qual_read_valid_st1),
		.read_dirty  (qual_read_dirty_st1),
		.read_tag    (qual_read_tag_st1),
		.read_data   (qual_read_data_st1),

		.invalidate  (invalidate_line),
		.write_enable(use_write_enable),
		.write_fill  (real_writefill),
		.write_addr  (writeaddr_st1e),
		.write_data  (use_write_data),
		.fill_sent   (fill_sent)
		);

	// VX_generic_register #(.N( 1 + 1 + `TAG_SELECT_NUM_BITS + (`DBANK_LINE_SIZE_WORDS*32) )) s0_1_c0 (
	VX_generic_register #(.N( 1 + 1 + `TAG_SELECT_NUM_BITS + (`DBANK_LINE_SIZE_WORDS*32) ), .Valid(0)) s0_1_c0 (
		.clk  (clk),
		.reset(reset),
		.stall(stall),
		.flush(0),
		.in   ({qual_read_valid_st1, qual_read_dirty_st1, qual_read_tag_st1, qual_read_data_st1}),
		.out  ({read_valid_st1c[0] , read_dirty_st1c[0] , read_tag_st1c[0] , read_data_st1c[0]})
	);

	genvar curr_stage;
	generate
		for (curr_stage = 1; curr_stage < STAGE_1_CYCLES-1; curr_stage = curr_stage + 1) begin
			VX_generic_register #(.N( 1 + 1 + `TAG_SELECT_NUM_BITS + (`DBANK_LINE_SIZE_WORDS*32) )) s0_1_cc (
				.clk  (clk),
				.reset(reset),
				.stall(stall),
				.flush(0),
				.in   ({read_valid_st1c[curr_stage-1] , read_dirty_st1c[curr_stage-1] , read_tag_st1c[curr_stage-1] , read_data_st1c[curr_stage-1]}),
				.out  ({read_valid_st1c[curr_stage]   , read_dirty_st1c[curr_stage]   , read_tag_st1c[curr_stage]   , read_data_st1c[curr_stage]  })
			);
		end
	endgenerate


	assign use_read_valid_st1e = read_valid_st1c[STAGE_1_CYCLES-1] || (FUNC_ID == `SFUNC_ID); // If shared memory, always valid
	assign use_read_dirty_st1e = read_dirty_st1c[STAGE_1_CYCLES-1] && (FUNC_ID != `SFUNC_ID); // Dirty only applies in Dcache
	assign use_read_tag_st1e   = (FUNC_ID == `SFUNC_ID) ? writeaddr_st1e[`TAG_SELECT_ADDR_RNG] : read_tag_st1c  [STAGE_1_CYCLES-1]; // Tag is always the same in SM

	genvar curr_w;
	for (curr_w = 0; curr_w < `DBANK_LINE_SIZE_WORDS; curr_w = curr_w+1) assign use_read_data_st1e[curr_w][31:0]  = read_data_st1c[STAGE_1_CYCLES-1][curr_w][31:0];
	// assign use_read_data_st1e  = read_data_st1c [STAGE_1_CYCLES-1];

/////////////////////// LOAD LOGIC ///////////////////

	wire[`OFFSET_SIZE_RNG]      byte_select  = writeaddr_st1e[`OFFSET_ADDR_RNG];
	wire[`WORD_SELECT_SIZE_RNG] block_offset = writeaddr_st1e[`WORD_SELECT_ADDR_RNG];

    wire lw  = valid_req_st1e && (mem_read_st1e == `LW_MEM_READ);
    wire lb  = valid_req_st1e && (mem_read_st1e == `LB_MEM_READ);
    wire lh  = valid_req_st1e && (mem_read_st1e == `LH_MEM_READ);
    wire lhu = valid_req_st1e && (mem_read_st1e == `LHU_MEM_READ);
    wire lbu = valid_req_st1e && (mem_read_st1e == `LBU_MEM_READ);

    wire b0 = (byte_select == 0);
    wire b1 = (byte_select == 1);
    wire b2 = (byte_select == 2);
    wire b3 = (byte_select == 3);

    wire[31:0] w0 = read_data_st1c[STAGE_1_CYCLES-1][0][31:0];
    wire[31:0] w1 = read_data_st1c[STAGE_1_CYCLES-1][1][31:0];
    wire[31:0] w2 = read_data_st1c[STAGE_1_CYCLES-1][2][31:0];
    wire[31:0] w3 = read_data_st1c[STAGE_1_CYCLES-1][3][31:0];

    wire[31:0] data_unmod  = read_data_st1c[STAGE_1_CYCLES-1][block_offset][31:0];

    wire[31:0] data_unQual = (b0 || lw) ? (data_unmod) :
                             b1 ? (data_unmod >> 8)    :
                             b2 ? (data_unmod >> 16)   :
                             (data_unmod >> 24);


    wire[31:0] lb_data     = (data_unQual[7] ) ? (data_unQual | 32'hFFFFFF00) : (data_unQual & 32'hFF);
    wire[31:0] lh_data     = (data_unQual[15]) ? (data_unQual | 32'hFFFF0000) : (data_unQual & 32'hFFFF);
    wire[31:0] lbu_data    = (data_unQual & 32'hFF);
    wire[31:0] lhu_data    = (data_unQual & 32'hFFFF);
    wire[31:0] lw_data     = (data_unQual);


    wire[31:0] sw_data     = writeword_st1e[31:0];

    wire[31:0] sb_data     = b1 ? {{16{1'b0}}, writeword_st1e[7:0], { 8{1'b0}}} :
                             b2 ? {{ 8{1'b0}}, writeword_st1e[7:0], {16{1'b0}}} :
                             b3 ? {{ 0{1'b0}}, writeword_st1e[7:0], {24{1'b0}}} :
                             writeword_st1e[31:0];

    wire[31:0] sh_data     = b2 ? {writeword_st1e[15:0], {16{1'b0}}} : writeword_st1e[31:0];



    wire[31:0] use_write_dat   = sb ? sb_data :
                                 sh ? sh_data :
                                 sw_data;


    wire[31:0] data_Qual   = lb  ? lb_data  :
                             lh  ? lh_data  :
                             lhu ? lhu_data :
                             lbu ? lbu_data :
                             lw_data;

/////////////////////// STORE LOGIC ///////////////////

    assign sw  = valid_req_st1e && (mem_write_st1e == `SW_MEM_WRITE);
    assign sb  = valid_req_st1e && (mem_write_st1e == `SB_MEM_WRITE);
    assign sh  = valid_req_st1e && (mem_write_st1e == `SH_MEM_WRITE);

    wire[3:0] sb_mask = (b0 ? 4'b0001 : (b1 ? 4'b0010 : (b2 ? 4'b0100 : 4'b1000)));
    wire[3:0] sh_mask = (b0 ? 4'b0011 : 4'b1100);

    wire should_write = (sw || sb || sh) && valid_req_st1e && use_read_valid_st1e && !miss_st1e;
    wire force_write  = real_writefill;

    wire[`DBANK_LINE_SIZE_RNG][3:0]  we;
    wire[`DBANK_LINE_SIZE_RNG][31:0] data_write;
	genvar g; 
	generate
		for (g = 0; g < `DBANK_LINE_SIZE_WORDS; g = g + 1) begin : write_enables
		    wire normal_write = (block_offset == g[`WORD_SELECT_SIZE_RNG]) && should_write && !real_writefill;

		    assign we[g]      = (force_write)        ? 4'b1111  : 
		    				 (should_write && !real_writefill && (FUNC_ID == `LLFUNC_ID)) ? 4'b1111  :
		                        (normal_write && sw) ? 4'b1111  :
		                        (normal_write && sb) ? sb_mask  :
		                        (normal_write && sh) ? sh_mask  :
		                        4'b0000;

		    if (!(FUNC_ID == `LLFUNC_ID)) assign data_write[g] = force_write ? writedata_st1e[g] : use_write_dat;
		end
		if ((FUNC_ID == `LLFUNC_ID)) begin
			assign data_write = force_write ? writedata_st1e : writeword_st1e;
		end
	endgenerate

	assign use_write_enable = (writefill_st1e && !real_writefill) ? 0 : we;
	assign use_write_data   = data_write;

///////////////////////
	if (FUNC_ID == `LLFUNC_ID) begin
		assign readword_st1e = read_data_st1c[STAGE_1_CYCLES-1];
	end else begin
		assign readword_st1e = data_Qual;
	end


	wire[`TAG_SELECT_ADDR_RNG] writeaddr_tag = writeaddr_st1e[`TAG_SELECT_ADDR_RNG];

	wire tags_match   = writeaddr_tag != use_read_tag_st1e;
	
	wire snoop_hit    = valid_req_st1e &&  is_snp_st1e && tags_match;
	wire req_invalid  = valid_req_st1e && !is_snp_st1e && !use_read_valid_st1e && !writefill_st1e;
	wire req_miss     = valid_req_st1e && !is_snp_st1e &&  use_read_valid_st1e && !writefill_st1e && tags_match;
	
	assign miss_st1e           = snoop_hit || req_invalid || req_miss;
	assign dirty_st1e          = valid_req_st1e && use_read_valid_st1e && use_read_dirty_st1e;
	assign readdata_st1e       = use_read_data_st1e;
	assign readtag_st1e        = use_read_tag_st1e;
	assign fill_sent           = miss_st1e;
	assign fill_saw_dirty_st1e = real_writefill && dirty_st1e;
	assign invalidate_line     = is_snp_st1e && miss_st1e;

endmodule