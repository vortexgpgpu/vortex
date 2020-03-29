`include "VX_cache_config.v"
`include "VX_define.v"
module VX_bank 
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
	// Fill Forward SNP Queue
	parameter FFSQ_SIZE                     = 8,

 	// Fill Invalidator Size {Fill invalidator must be active}
 	parameter FILL_INVALIDAOR_SIZE          = 16, 

// Dram knobs
	parameter SIMULATED_DRAM_LATENCY_CYCLES = 10


	)
	(
	input wire clk,
	input wire reset,

	// Input Core Request
	input wire                                    delay_req,
	input wire [NUMBER_REQUESTS-1:0]              bank_valids,
	input wire [NUMBER_REQUESTS-1:0][31:0]        bank_addr,
	input wire [NUMBER_REQUESTS-1:0][`WORD_SIZE_RNG] bank_writedata,
	input wire [4:0]                              bank_rd,
	input wire [NUMBER_REQUESTS-1:0][1:0]         bank_wb,
	input wire [31:0]                             bank_pc,
	input wire [`NW_M1:0]                         bank_warp_num,
	input wire [NUMBER_REQUESTS-1:0][2:0]         bank_mem_read,  
	input wire [NUMBER_REQUESTS-1:0][2:0]         bank_mem_write,
	output wire                                   reqq_full,

	// Output Core WB
	input  wire                                   bank_wb_pop,
	output wire                                   bank_wb_valid,
	output wire [`vx_clog2(NUMBER_REQUESTS)-1:0]  bank_wb_tid,
	output wire [4:0]                             bank_wb_rd,
	output wire [1:0]                             bank_wb_wb,
	output wire [`NW_M1:0]                        bank_wb_warp_num,
	output wire [`WORD_SIZE_RNG]                  bank_wb_data,
	output wire [31:0]                            bank_wb_pc,
	output wire [31:0]                            bank_wb_address,

	// Dram Fill Requests
	output wire                                   dram_fill_req,
	output wire[31:0]                             dram_fill_req_addr,
    output wire                                   dram_because_of_snp,
    output wire                                   dram_snp_full,
	input  wire                                   dram_fill_req_queue_full,

	// Dram Fill Response
	input  wire                                   dram_fill_rsp,
	input  wire [31:0]                            dram_fill_addr,
	input  wire[`BANK_LINE_SIZE_RNG][`WORD_SIZE-1:0] dram_fill_rsp_data,
	output wire                                   dram_fill_accept,

	// Dram WB Requests
	input  wire                                   dram_wb_queue_pop,
	output wire                                   dram_wb_req,
	output wire[31:0]                             dram_wb_req_addr,
	output wire[`BANK_LINE_SIZE_RNG][`WORD_SIZE-1:0] dram_wb_req_data,

	// Snp Request
	input  wire                                   snp_req,
	input  wire[31:0]                             snp_req_addr,
	output wire                                   snrq_full,

	output wire                                   snp_fwd,
	output wire[31:0]                             snp_fwd_addr,
	input  wire                                   snp_fwd_pop
);




	wire       snrq_pop;
	wire       snrq_empty;

	wire       snrq_valid_st0;
	wire[31:0] snrq_addr_st0;

	reg        snrq_hazard_st0;

	assign snrq_valid_st0 = !snrq_empty;
	VX_generic_queue_ll #(.DATAW(32), .SIZE(SNRQ_SIZE)) snr_queue(
		.clk     (clk),
		.reset   (reset),
		.push    (snp_req),
		.in_data (snp_req_addr),
		.pop     (snrq_pop),
		.out_data(snrq_addr_st0),
		.empty   (snrq_empty),
		.full    (snrq_full)
		);

	wire                            dfpq_pop;
	wire                            dfpq_empty;
	wire                            dfpq_full;
	wire[31:0]                      dfpq_addr_st0;
	wire[`BANK_LINE_SIZE_RNG][`WORD_SIZE-1:0] dfpq_filldata_st0;
	reg                             dfpq_hazard_st0;

	assign dram_fill_accept = !dfpq_full;

	VX_generic_queue_ll #(.DATAW(32+(`BANK_LINE_SIZE_WORDS*`WORD_SIZE)), .SIZE(DFPQ_SIZE)) dfp_queue(
		.clk     (clk),
		.reset   (reset),
		.push    (dram_fill_rsp),
		.in_data ({dram_fill_addr, dram_fill_rsp_data}),
		.pop     (dfpq_pop),
		.out_data({dfpq_addr_st0, dfpq_filldata_st0}),
		.empty   (dfpq_empty),
		.full    (dfpq_full)
		);


	wire                                  reqq_pop;
	wire                                  reqq_push;
	wire                                  reqq_empty;
	wire                                  reqq_req_st0;
	wire[`vx_clog2(NUMBER_REQUESTS)-1:0] reqq_req_tid_st0;
	wire [31:0]                           reqq_req_addr_st0;
	wire [`WORD_SIZE_RNG]                 reqq_req_writeword_st0;
	wire [4:0]                            reqq_req_rd_st0;
	wire [1:0]                            reqq_req_wb_st0;
	wire [`NW_M1:0]                       reqq_req_warp_num_st0;
	wire [2:0]                            reqq_req_mem_read_st0;  
	wire [2:0]                            reqq_req_mem_write_st0;
	reg                                   reqq_hazard_st0;
	wire [31:0]                           reqq_req_pc_st0;

	assign reqq_push = !delay_req && (|bank_valids);

	VX_cache_req_queue  #(
        .CACHE_SIZE_BYTES             (CACHE_SIZE_BYTES),
        .BANK_LINE_SIZE_BYTES         (BANK_LINE_SIZE_BYTES),
        .NUMBER_BANKS                 (NUMBER_BANKS),
        .WORD_SIZE_BYTES              (WORD_SIZE_BYTES),
        .NUMBER_REQUESTS              (NUMBER_REQUESTS),
        .STAGE_1_CYCLES               (STAGE_1_CYCLES),
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
		req_queue
		(
		.clk                   (clk),
		.reset                 (reset),
		// Enqueue
		.reqq_push             (reqq_push),
		.bank_valids           (bank_valids),
		.bank_addr             (bank_addr),
		.bank_writedata        (bank_writedata),
		.bank_rd               (bank_rd),
		.bank_pc               (bank_pc),
		.bank_wb               (bank_wb),
		.bank_warp_num         (bank_warp_num),
		.bank_mem_read         (bank_mem_read),
		.bank_mem_write        (bank_mem_write),

		// Dequeue
		.reqq_pop              (reqq_pop),
		.reqq_req_st0          (reqq_req_st0),
		.reqq_req_tid_st0      (reqq_req_tid_st0),
		.reqq_req_addr_st0     (reqq_req_addr_st0),
		.reqq_req_writedata_st0(reqq_req_writeword_st0),
		.reqq_req_rd_st0       (reqq_req_rd_st0),
		.reqq_req_wb_st0       (reqq_req_wb_st0),
		.reqq_req_warp_num_st0 (reqq_req_warp_num_st0),
		.reqq_req_mem_read_st0 (reqq_req_mem_read_st0),
		.reqq_req_mem_write_st0(reqq_req_mem_write_st0),
		.reqq_req_pc_st0       (reqq_req_pc_st0),
		.reqq_empty            (reqq_empty),
		.reqq_full             (reqq_full)
		);

	wire                                  mrvq_pop;
	wire                                  mrvq_full;
	wire                                  mrvq_stop;
	wire                                  mrvq_valid_st0;
	wire[`vx_clog2(NUMBER_REQUESTS)-1:0] mrvq_tid_st0;
	wire [31:0]                           mrvq_addr_st0;
	wire [`WORD_SIZE_RNG]                 mrvq_writeword_st0;
	wire [4:0]                            mrvq_rd_st0;
	wire [1:0]                            mrvq_wb_st0;
	wire [31:0]                           miss_resrv_pc_st0;
	wire [`NW_M1:0]                       mrvq_warp_num_st0;
	wire [2:0]                            mrvq_mem_read_st0;  
	wire [2:0]                            mrvq_mem_write_st0;
	reg                                   mrvq_hazard_st0;

	wire                                  miss_add;
	wire[31:0]                            miss_add_addr;
	wire[`WORD_SIZE_RNG]                  miss_add_data;
	wire[`vx_clog2(NUMBER_REQUESTS)-1:0]  miss_add_tid;
	wire[4:0]                             miss_add_rd;
	wire[1:0]                             miss_add_wb;
	wire[`NW_M1:0]                        miss_add_warp_num;
	wire[2:0]                             miss_add_mem_read;
	wire[2:0]                             miss_add_mem_write;

	wire[31:0]                            miss_add_pc;

	wire[31:0]                      	addr_st2;
	wire                            	is_fill_st2;

	VX_cache_miss_resrv #(
        .CACHE_SIZE_BYTES             (CACHE_SIZE_BYTES),
        .BANK_LINE_SIZE_BYTES         (BANK_LINE_SIZE_BYTES),
        .NUMBER_BANKS                 (NUMBER_BANKS),
        .WORD_SIZE_BYTES              (WORD_SIZE_BYTES),
        .NUMBER_REQUESTS              (NUMBER_REQUESTS),
        .STAGE_1_CYCLES               (STAGE_1_CYCLES),
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
		mrvq_queue
		(
		.clk                     (clk),
		.reset                   (reset),
		// Enqueue
		.miss_add                (miss_add), // Need to do all 
		.miss_add_addr           (miss_add_addr),
		.miss_add_data           (miss_add_data),
		.miss_add_tid            (miss_add_tid),
		.miss_add_rd             (miss_add_rd),
		.miss_add_wb             (miss_add_wb),
		.miss_add_warp_num       (miss_add_warp_num),
		.miss_add_mem_read       (miss_add_mem_read),
		.miss_add_mem_write      (miss_add_mem_write),
		.miss_add_pc             (miss_add_pc),
		.miss_resrv_full         (mrvq_full),
		.miss_resrv_stop         (mrvq_stop),

		// Broadcast
		.is_fill_st1             (is_fill_st2),
		.fill_addr_st1           (addr_st2),

		// Dequeue
		.miss_resrv_pop          (mrvq_pop),
		.miss_resrv_valid_st0    (mrvq_valid_st0),
		.miss_resrv_addr_st0     (mrvq_addr_st0),
		.miss_resrv_data_st0     (mrvq_writeword_st0),
		.miss_resrv_tid_st0      (mrvq_tid_st0),
		.miss_resrv_rd_st0       (mrvq_rd_st0),
		.miss_resrv_wb_st0       (mrvq_wb_st0),
		.miss_resrv_pc_st0       (miss_resrv_pc_st0),
		.miss_resrv_warp_num_st0 (mrvq_warp_num_st0),
		.miss_resrv_mem_read_st0 (mrvq_mem_read_st0),
		.miss_resrv_mem_write_st0(mrvq_mem_write_st0)
		);

	wire stall_bank_pipe;
	reg  is_fill_in_pipe;

	wire            valid_st1         [STAGE_1_CYCLES-1:0];
	wire            is_fill_st1       [STAGE_1_CYCLES-1:0];
	wire            going_to_write_st1[STAGE_1_CYCLES-1:0];	
	wire [31:0]		addr_st1          [STAGE_1_CYCLES-1:0];

	reg[16:0] p_stage;
	always @(*) begin
		is_fill_in_pipe = 0;
		for (p_stage = 0; p_stage < STAGE_1_CYCLES; p_stage=p_stage+1) begin
			if (is_fill_st1[p_stage]) is_fill_in_pipe = 1;
		end

		if (is_fill_st2) is_fill_in_pipe = 1;
	end

//	assign is_fill_in_pipe = (|is_fill_st1) || is_fill_st2;
		

	assign mrvq_pop = mrvq_valid_st0 && !stall_bank_pipe && !mrvq_hazard_st0;
	assign dfpq_pop = !mrvq_pop && !dfpq_empty && !stall_bank_pipe && !dfpq_hazard_st0;
	assign reqq_pop = !mrvq_stop && !mrvq_pop && !dfpq_pop && !reqq_empty && reqq_req_st0 && !stall_bank_pipe && !is_fill_st1[0] && !(reqq_hazard_st0 || (mrvq_valid_st0 && mrvq_hazard_st0)) && !is_fill_in_pipe;
	assign snrq_pop = !reqq_pop && !reqq_pop && !mrvq_pop && !dfpq_pop && snrq_valid_st0 && !stall_bank_pipe && !snrq_hazard_st0;

	integer st1_cycle;

	always @(*) begin
		dfpq_hazard_st0 = 0;
		mrvq_hazard_st0 = 0;
		reqq_hazard_st0 = 0;
		snrq_hazard_st0 = 0;
		// for (st1_cycle = 0; st1_cycle < STAGE_1_CYCLES; st1_cycle = st1_cycle + 1) begin
		// 	if (valid_st1[st1_cycle] && going_to_write_st1[st1_cycle]) begin
		// 		if (dfpq_addr_st0    [31:`LINE_SELECT_ADDR_START] == addr_st1[st1_cycle][31:`LINE_SELECT_ADDR_START]) dfpq_hazard_st0 = 1;
		// 		if (mrvq_addr_st0    [31:`LINE_SELECT_ADDR_START] == addr_st1[st1_cycle][31:`LINE_SELECT_ADDR_START]) mrvq_hazard_st0 = 1;
		// 		if (reqq_req_addr_st0[31:`LINE_SELECT_ADDR_START] == addr_st1[st1_cycle][31:`LINE_SELECT_ADDR_START]) reqq_hazard_st0 = 1;
		// 		if (snrq_addr_st0    [31:`LINE_SELECT_ADDR_START] == addr_st1[st1_cycle][31:`LINE_SELECT_ADDR_START]) snrq_hazard_st0 = 1;
		// 	end
		// end
	end

	wire                                  qual_is_fill_st0;
	wire                                  qual_valid_st0;
	wire [31:0]                           qual_addr_st0;
	wire [`WORD_SIZE_RNG]                 qual_writeword_st0;
	wire [`BANK_LINE_SIZE_RNG][`WORD_SIZE-1:0] qual_writedata_st0;
	wire [`REQ_INST_META_SIZE-1:0]        qual_inst_meta_st0;
	wire                                  qual_going_to_write_st0;
	wire                                  qual_is_snp;
	wire [31:0]                           qual_pc_st0;

	wire [`WORD_SIZE_RNG]                 writeword_st1     [STAGE_1_CYCLES-1:0];
	wire [`REQ_INST_META_SIZE-1:0]        inst_meta_st1     [STAGE_1_CYCLES-1:0];	
	wire [`BANK_LINE_SIZE_RNG][`WORD_SIZE-1:0] writedata_st1[STAGE_1_CYCLES-1:0];
	wire                                  is_snp_st1        [STAGE_1_CYCLES-1:0];
	wire [31:0]                           pc_st1            [STAGE_1_CYCLES-1:0];

	assign qual_is_fill_st0 = dfpq_pop;

	// always @(*) begin
	// 	if (qual_is_fill_st0 && (FUNC_ID == 3)) begin
	// 		$display("WHAT THE FUCK FUNC_ID: %x", FUNC_ID);
	// 	end
	// end

	assign qual_valid_st0   = dfpq_pop || mrvq_pop || reqq_pop || snrq_pop;

	assign qual_addr_st0    = dfpq_pop ? dfpq_addr_st0     :
	                          mrvq_pop ? mrvq_addr_st0     :
	                          reqq_pop ? reqq_req_addr_st0 :
	                          snrq_pop ? snrq_addr_st0     :
	                          0;

	assign qual_writedata_st0 = dfpq_pop ? dfpq_filldata_st0 : 57;

	assign qual_inst_meta_st0 = mrvq_pop ? {mrvq_rd_st0    , mrvq_wb_st0    , mrvq_warp_num_st0    , mrvq_mem_read_st0    , mrvq_mem_write_st0    , mrvq_tid_st0    } :
								reqq_pop ? {reqq_req_rd_st0, reqq_req_wb_st0, reqq_req_warp_num_st0, reqq_req_mem_read_st0, reqq_req_mem_write_st0, reqq_req_tid_st0} :
								0;

	assign qual_going_to_write_st0 = dfpq_pop ? 1 :
										(mrvq_pop && (mrvq_mem_write_st0 != `NO_MEM_WRITE)) ? 1 :
											(reqq_pop && (reqq_req_mem_write_st0 != `NO_MEM_WRITE)) ? 1 :
												(snrq_pop) ? 1 :
													0;

	assign qual_pc_st0             = (reqq_pop) ? reqq_req_pc_st0         :
										(mrvq_pop) ? miss_resrv_pc_st0    :
											(dfpq_pop) ? 32'hdeadbeef     :
												(snrq_pop) ? 32'hb00b0000 :
													32'h0;
	assign qual_is_snp             =  snrq_pop ? 1 : 0;

	assign qual_writeword_st0 = mrvq_pop ? mrvq_writeword_st0     :
								reqq_pop ? reqq_req_writeword_st0 :
								0;

	VX_generic_register #(.N( 1 + 1 + 1 +  `WORD_SIZE + 32 + `REQ_INST_META_SIZE + (`BANK_LINE_SIZE_WORDS*`WORD_SIZE) + 1 + 32)) s0_1_c0 (
	.clk  (clk),
	.reset(reset),
	.stall(stall_bank_pipe),
	.flush(0),
	.in   ({qual_is_snp  , qual_going_to_write_st0, qual_valid_st0, qual_addr_st0, qual_writeword_st0, qual_inst_meta_st0, qual_is_fill_st0, qual_writedata_st0, qual_pc_st0 }),
	.out  ({is_snp_st1[0], going_to_write_st1[0]  , valid_st1[0]  , addr_st1[0]  , writeword_st1[0]  , inst_meta_st1[0]  , is_fill_st1[0]  , writedata_st1[0]  , pc_st1[0]})
	);

	genvar curr_stage;
	generate
		for (curr_stage = 1; curr_stage < STAGE_1_CYCLES; curr_stage = curr_stage + 1) begin
			VX_generic_register #(.N( 1 + 1 + 1 +  `WORD_SIZE + 32 + `REQ_INST_META_SIZE + (`BANK_LINE_SIZE_WORDS*`WORD_SIZE) + 1 + 32)) s0_1_cc (
			.clk  (clk),
			.reset(reset),
			.stall(stall_bank_pipe),
			.flush(0),
			.in   ({is_snp_st1[curr_stage-1], going_to_write_st1[curr_stage-1], valid_st1[curr_stage-1], addr_st1[curr_stage-1], writeword_st1[curr_stage-1], inst_meta_st1[curr_stage-1], is_fill_st1[curr_stage-1]  , writedata_st1[curr_stage-1], pc_st1[curr_stage-1]}),
			.out  ({is_snp_st1[curr_stage]  , going_to_write_st1[curr_stage]  , valid_st1[curr_stage]  , addr_st1[curr_stage]  , writeword_st1[curr_stage]  , inst_meta_st1[curr_stage]  , is_fill_st1[curr_stage]    , writedata_st1[curr_stage]  , pc_st1[curr_stage]})
			);
		end
	endgenerate


	wire[`WORD_SIZE_RNG]            readword_st1e;
	wire[`BANK_LINE_SIZE_RNG][`WORD_SIZE-1:0] readdata_st1e;
	wire[`TAG_SELECT_SIZE_RNG]      readtag_st1e;
	wire                            miss_st1e;
	wire                            dirty_st1e;
	wire[31:0]                      pc_st1e;


	wire [4:0]                             rd_st1e;
	wire [1:0]                             wb_st1e;
	wire [`NW_M1:0]                        warp_num_st1e;
	wire [2:0]                             mem_read_st1e;  
	wire [2:0]                             mem_write_st1e;
	wire [`vx_clog2(NUMBER_REQUESTS)-1:0]  tid_st1e;
	wire                                   fill_saw_dirty_st1e;
	wire                                   is_snp_st1e;

	assign is_snp_st1e = is_snp_st1[STAGE_1_CYCLES-1];
	assign pc_st1e     = pc_st1[STAGE_1_CYCLES-1];
	assign {rd_st1e, wb_st1e, warp_num_st1e, mem_read_st1e, mem_write_st1e, tid_st1e} = inst_meta_st1[STAGE_1_CYCLES-1];


	VX_tag_data_access  #(
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
		VX_tag_data_access
		(
		.clk           (clk),
		.reset         (reset),
		.stall         (stall_bank_pipe),

		// Initial Read
		.readaddr_st10 (addr_st1[0]),

		// Actual Read/Write
		.valid_req_st1e(valid_st1[STAGE_1_CYCLES-1]),
		.writefill_st1e(is_fill_st1[STAGE_1_CYCLES-1]),
		.writeaddr_st1e(addr_st1[STAGE_1_CYCLES-1]),
		.writeword_st1e(writeword_st1[STAGE_1_CYCLES-1]),
		.writedata_st1e(writedata_st1[STAGE_1_CYCLES-1]),

		.mem_write_st1e(mem_write_st1e),
		.mem_read_st1e (mem_read_st1e), 

		.is_snp_st1e   (is_snp_st1e),

		// Read Data
		.readword_st1e (readword_st1e),
		.readdata_st1e (readdata_st1e),
		.readtag_st1e  (readtag_st1e),
		.miss_st1e     (miss_st1e),
		.dirty_st1e    (dirty_st1e),
		.fill_saw_dirty_st1e(fill_saw_dirty_st1e)
		);

	wire qual_valid_st1e_2 = valid_st1[STAGE_1_CYCLES-1] && !is_fill_st1[STAGE_1_CYCLES-1];

	wire                            valid_st2;	
	wire[`WORD_SIZE_RNG]            writeword_st2;
	wire[`WORD_SIZE_RNG]            readword_st2;
	wire[`BANK_LINE_SIZE_RNG][`WORD_SIZE-1:0] readdata_st2;
	wire                            miss_st2;
	wire                            dirty_st2;
	wire[`REQ_INST_META_SIZE-1:0]   inst_meta_st2;
	wire[`TAG_SELECT_SIZE_RNG]      readtag_st2;	
	wire                            fill_saw_dirty_st2;
	wire                            is_snp_st2;
	wire [31:0]                     pc_st2;


	VX_generic_register #(.N( 1+1+1+1+32+`WORD_SIZE+`WORD_SIZE+(`BANK_LINE_SIZE_WORDS * `WORD_SIZE) + `REQ_INST_META_SIZE + `TAG_SELECT_NUM_BITS + 32 + 2)) st_1e_2 (
		.clk  (clk),
		.reset(reset),
		.stall(stall_bank_pipe),
		.flush(0),
		.in   ({is_snp_st1e, fill_saw_dirty_st1e, is_fill_st1[STAGE_1_CYCLES-1] , qual_valid_st1e_2, addr_st1[STAGE_1_CYCLES-1], writeword_st1[STAGE_1_CYCLES-1], readword_st1e, readdata_st1e, readtag_st1e, miss_st1e, dirty_st1e, pc_st1e, inst_meta_st1[STAGE_1_CYCLES-1]}),
		.out  ({is_snp_st2 , fill_saw_dirty_st2 , is_fill_st2                   , valid_st2        , addr_st2                  , writeword_st2                  , readword_st2 , readdata_st2 , readtag_st2 , miss_st2 , dirty_st2 , pc_st2 , inst_meta_st2                  })
		);

	wire cwbq_full;
	wire dwbq_full;
	wire ffsq_full;
	wire invalidate_fill;

	// Enqueue to miss reserv if it's a valid miss
	assign miss_add       = valid_st2 && miss_st2 && !mrvq_full && !((is_snp_st2 && valid_st2 && ffsq_full) ||((valid_st2 && !miss_st2) && cwbq_full) || (((valid_st2 && miss_st2 && dirty_st2) || fill_saw_dirty_st2) && dwbq_full) || (valid_st2 && miss_st2 && mrvq_full));
	assign miss_add_pc    = pc_st2;
	assign miss_add_addr  = addr_st2;
	assign miss_add_data  = writeword_st2;
	assign {miss_add_rd, miss_add_wb, miss_add_warp_num, miss_add_mem_read, miss_add_mem_write, miss_add_tid} = inst_meta_st2;


	// Enqueue to CWB Queue
	wire                                   cwbq_push      = (valid_st2 && !miss_st2) && !cwbq_full && !((FUNC_ID == `LLFUNC_ID) && (miss_add_wb == 0)) && !((is_snp_st2 && valid_st2 && ffsq_full) || (((valid_st2 && miss_st2 && dirty_st2) || fill_saw_dirty_st2) && dwbq_full) || (valid_st2 && miss_st2 && mrvq_full) || (valid_st2 && miss_st2 && !invalidate_fill && dram_fill_req_queue_full));
	wire [`WORD_SIZE_RNG]                  cwbq_data      = readword_st2;
	wire [`vx_clog2(NUMBER_REQUESTS)-1:0]  cwbq_tid       = miss_add_tid;
	wire [4:0]                             cwbq_rd        = miss_add_rd;
	wire [1:0]                             cwbq_wb        = miss_add_wb;
	wire [`NW_M1:0]                        cwbq_warp_num  = miss_add_warp_num;
	wire [31:0]                            cwbq_pc        = pc_st2;
	
	wire                                   cwbq_empty;
	assign bank_wb_valid = !cwbq_empty;
	VX_generic_queue_ll #(.DATAW( `vx_clog2(NUMBER_REQUESTS) + 5 + 2 + (`NW_M1+1) + `WORD_SIZE + 32 + 32), .SIZE(CWBQ_SIZE)) cwb_queue(
		.clk     (clk),
		.reset   (reset),

		.push    (cwbq_push),
		.in_data ({cwbq_tid, cwbq_rd, cwbq_wb, cwbq_warp_num, cwbq_data, cwbq_pc, addr_st2}),

		.pop     (bank_wb_pop),
		.out_data({bank_wb_tid, bank_wb_rd, bank_wb_wb, bank_wb_warp_num, bank_wb_data, bank_wb_pc, bank_wb_address}),
		.empty   (cwbq_empty),
		.full    (cwbq_full)
		);

	// Enqueue to DWB Queue
	wire                             dwbq_push     = ((valid_st2 && miss_st2 && dirty_st2) || fill_saw_dirty_st2) && !dwbq_full && !((is_snp_st2 && valid_st2 && ffsq_full) ||((valid_st2 && !miss_st2) && cwbq_full) || (valid_st2 && miss_st2 && mrvq_full) || (valid_st2 && miss_st2 && !invalidate_fill && dram_fill_req_queue_full));
	wire[31:0]                       dwbq_req_addr = {readtag_st2, addr_st2[`LINE_SELECT_ADDR_END:0]} & `BASE_ADDR_MASK;
	wire[`BANK_LINE_SIZE_RNG][`WORD_SIZE-1:0]  dwbq_req_data = readdata_st2;
	wire                             dwbq_empty;
	
    wire possible_fill = valid_st2 && miss_st2 && !dram_fill_req_queue_full;
	wire[31:0] fill_invalidator_addr = addr_st2 & `BASE_ADDR_MASK;
	VX_fill_invalidator  #(
        .CACHE_SIZE_BYTES             (CACHE_SIZE_BYTES),
        .BANK_LINE_SIZE_BYTES         (BANK_LINE_SIZE_BYTES),
        .NUMBER_BANKS                 (NUMBER_BANKS),
        .WORD_SIZE_BYTES              (WORD_SIZE_BYTES),
        .NUMBER_REQUESTS              (NUMBER_REQUESTS),
        .STAGE_1_CYCLES               (STAGE_1_CYCLES),
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
		VX_fill_invalidator
		(
		.clk               (clk),
		.reset             (reset),
		.possible_fill     (possible_fill),
		.success_fill      (is_fill_st2),
		.fill_addr         (fill_invalidator_addr),

		.invalidate_fill   (invalidate_fill)
		);

	// Enqueu in dram_fill_req
	assign dram_fill_req       = possible_fill && !invalidate_fill;
	assign dram_because_of_snp = is_snp_st2 && valid_st2 && miss_st2;
	assign dram_snp_full       = snrq_full && snp_req;
	assign dram_fill_req_addr  = addr_st2 & `BASE_ADDR_MASK;

	assign dram_wb_req = !dwbq_empty;
	VX_generic_queue_ll #(.DATAW( 32 + (`BANK_LINE_SIZE_WORDS * `WORD_SIZE)), .SIZE(DWBQ_SIZE)) dwb_queue(
		.clk     (clk),
		.reset   (reset),

		.push    (dwbq_push),
		.in_data ({dwbq_req_addr, dwbq_req_data}),

		.pop     (dram_wb_queue_pop),
		.out_data({dram_wb_req_addr, dram_wb_req_data}),
		.empty   (dwbq_empty),
		.full    (dwbq_full)
		);

	wire snp_fwd_push;	
	wire ffsq_empty;

	assign snp_fwd_push = is_snp_st2 && valid_st2 && !ffsq_full && !(((valid_st2 && !miss_st2) && cwbq_full) || (((valid_st2 && miss_st2 && dirty_st2) || fill_saw_dirty_st2) && dwbq_full) || (valid_st2 && miss_st2 && mrvq_full) || (valid_st2 && miss_st2 && !invalidate_fill && dram_fill_req_queue_full));
	assign snp_fwd      = !ffsq_empty;
	VX_generic_queue_ll #(.DATAW(32), .SIZE(FFSQ_SIZE)) ffs_queue(
		.clk     (clk),
		.reset   (reset),
		.push    (snp_fwd_push),
		.in_data ({addr_st2}),
		.pop     (snp_fwd_pop),
		.out_data({snp_fwd_addr}),
		.empty   (ffsq_empty),
		.full    (ffsq_full)
		);

	assign stall_bank_pipe = (is_snp_st2 && valid_st2 && ffsq_full) || ((valid_st2 && !miss_st2) && cwbq_full) || (((valid_st2 && miss_st2 && dirty_st2) || fill_saw_dirty_st2) && dwbq_full) || (valid_st2 && miss_st2 && mrvq_full) || (valid_st2 && miss_st2 && !invalidate_fill && dram_fill_req_queue_full);

endmodule : VX_bank