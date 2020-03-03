`include "VX_cache_config.v"

module VX_bank (
	input wire clk,
	input wire reset,

	// Input Core Request
	input wire                                    delay_req,
	input wire [`NUMBER_REQUESTS-1:0]             bank_valids,
	input wire [`NUMBER_REQUESTS-1:0][31:0]       bank_addr,
	input wire [`NUMBER_REQUESTS-1:0][31:0]       bank_writedata,
	input wire [4:0]                              bank_rd,
	input wire [1:0]                              bank_wb,
	input wire [`NW_M1:0]                         bank_warp_num,
	input wire [2:0]                              bank_mem_read,  
	input wire [2:0]                              bank_mem_write,
	output wire                                   reqq_full,

	// Output Core WB
	input  wire                                   bank_wb_pop,
	output wire [`vx_clog2(`NUMBER_REQUESTS)-1:0] bank_wb_tid,
	output wire [4:0]                             bank_wb_rd,
	output wire [1:0]                             bank_wb_wb,
	output wire [`NW_M1:0]                        bank_wb_warp_num,
	output wire [31:0]                            bank_wb_data,

	// Dram Fill Requests
	output wire                                   dram_fill_req,
	output wire[31:0]                             dram_fill_req_addr,
	input  wire                                   dram_fill_req_queue_full,

	// Dram Fill Response
	input  wire                                   dram_fill_rsp,
	input  wire [31:0]                            dram_fill_addr,
	input  wire[`BANK_LINE_SIZE_RNG][31:0]        dram_fill_rsp_data,
	output wire                                   dram_fill_accept,

	// Dram WB Requests
	input  wire                                   dram_wb_queue_pop,
	output wire                                   dram_wb_req,
	output wire[31:0]                             dram_wb_req_addr,
	output wire[`BANK_LINE_SIZE_RNG][31:0]        dram_wb_req_data
);


	wire                            dfpq_pop; // Use this
	wire                            dfpq_empty;
	wire                            dfpq_full;
	wire[31:0]                      dfpq_addr_st0;
	wire[`BANK_LINE_SIZE_RNG][31:0] dfpq_filldata_st0;

	assign dram_fill_accept = !dfpq_full;

	VX_generic_queue #(.DATAW(32+(`BANK_LINE_SIZE_WORDS*32)), .SIZE(`DFPQ_SIZE)) dfp_queue(
		.clk     (clk),
		.reset   (reset),
		.push    (dram_fill_rsp),
		.in_data ({dram_fill_addr, dram_fill_rsp_data}),
		.pop     (dfpq_pop),
		.out_data({dfpq_addr_st0, dfpq_filldata_st0}),
		.empty   (dfpq_empty),
		.full    (dfpq_full)
		);


	wire                                  reqq_pop; // Use this
	wire                                  reqq_push;
	wire                                  reqq_empty;
	wire                                  reqq_req_st0;
	wire[`vx_clog2(`NUMBER_REQUESTS)-1:0] reqq_req_tid_st0;
	wire [31:0]                           reqq_req_addr_st0;
	wire [31:0]                           reqq_req_writeword_st0;
	wire [4:0]                            reqq_req_rd_st0;
	wire [1:0]                            reqq_req_wb_st0;
	wire [`NW_M1:0]                       reqq_req_warp_num_st0;
	wire [2:0]                            reqq_req_mem_read_st0;  
	wire [2:0]                            reqq_req_mem_write_st0;

	assign reqq_push = !delay_req && (|bank_valids);

	VX_cache_req_queue mrvq_queue(
		.clk                   (clk),
		.reset                 (reset),
		// Enqueue
		.reqq_push             (reqq_push),
		.bank_valids           (bank_valids),
		.bank_addr             (bank_addr),
		.bank_writedata        (bank_writedata),
		.bank_rd               (bank_rd),
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
		.reqq_empty            (reqq_empty),
		.reqq_full             (reqq_full)
		);

	wire                                  mrvq_pop; // Use this
	wire                                  mrvq_full;
	wire                                  mrvq_valid_st0;
	wire[`vx_clog2(`NUMBER_REQUESTS)-1:0] mrvq_tid_st0;
	wire [31:0]                           mrvq_addr_st0;
	wire [31:0]                           mrvq_writeword_st0;
	wire [4:0]                            mrvq_rd_st0;
	wire [1:0]                            mrvq_wb_st0;
	wire [`NW_M1:0]                       mrvq_warp_num_st0;
	wire [2:0]                            mrvq_mem_read_st0;  
	wire [2:0]                            mrvq_mem_write_st0;

	VX_cache_miss_resrv mrvq_queue(
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
		.miss_resrv_full         (mrvq_full)

		// Broadcast
		.is_fill_st1             (is_fill_st1),
		.fill_addr_st1           (addr_st1),

		// Dequeue
		.miss_resrv_pop          (mrvq_pop),
		.miss_resrv_valid_st0    (mrvq_valid_st0),
		.miss_resrv_addr_st0     (mrvq_addr_st0),
		.miss_resrv_data_st0     (mrvq_writeword_st0),
		.miss_resrv_tid_st0      (mrvq_tid_st0),
		.miss_resrv_rd_st0       (mrvq_rd_st0),
		.miss_resrv_wb_st0       (mrvq_wb_st0),
		.miss_resrv_warp_num_st0 (mrvq_warp_num_st0),
		.miss_resrv_mem_read_st0 (mrvq_mem_read_st0),
		.miss_resrv_mem_write_st0(mrvq_mem_write_st0)
		);

	wire stall_st0;
	wire stall_st1;
	wire stall_st2;

	assign stall_st1 = stall_st2;
	assign stall_st0 = stall_st1;

	assign dfpq_pop =              !dfpq_empty    && !stall_st0;
	assign mrvq_pop = !dfpq_pop && mrvq_valid_st0 && !stall_st0;
	assign reqq_pop = !mrvq_pop && reqq_req_st0   && !stall_st0;


	wire                                  qual_is_fill_st0;
	wire                                  qual_valid_st0;
	wire [31:0]                           qual_addr_st0;
	wire [31:0]                           qual_writeword_st0;
	wire [`BANK_LINE_SIZE_RNG][31:0]      qual_writedata_st0;
	wire [`REQ_INST_META_SIZE-1:0]        qual_inst_meta_st0;

	wire                                  is_fill_st1;
	wire                                  valid_st1;
	wire [31:0]                           addr_st1;
	wire [31:0]                           writeword_st1;
	wire [`BANK_LINE_SIZE_RNG][31:0]      writedata_st1;
	wire [`REQ_INST_META_SIZE-1:0]        inst_meta_st1;

	assign qual_is_fill_st0 = dfpq_pop;
	assign qual_valid_st0   = dfpq_pop || mrvq_pop || reqq_pop;

	assign qual_addr_st0    = dfpq_pop ? dfpq_addr_st0     :
	                          mrvq_pop ? mrvq_addr_st0     :
	                          reqq_pop ? reqq_req_addr_st0 :
	                          0;

	assign qual_writeword_st0 = mrvq_pop ? mrvq_writeword_st0     :
								reqq_pop ? reqq_req_writeword_st0 :
								0;

	assign qual_writedata_st0 = dfpq_pop ? dfpq_filldata_st0 : 0;

	assign qual_inst_meta_st0 = mrvq_pop ? {mrvq_rd_st0, mrvq_wb_st0, mrvq_warp_num_st0, mrvq_mem_read_st0, mrvq_mem_write_st0, mrvq_tid_st0} :
								reqq_pop ? {reqq_rd_st0, reqq_wb_st0, reqq_warp_num_st0, reqq_mem_read_st0, reqq_mem_write_st0, reqq_tid_st0} :
								0;

	VX_generic_register #(.N(1)) s0_1 (
		.clk  (clk),
		.reset(reset),
		.stall(stall_st1),
		.flush(0),
		.in   ({qual_is_fill_st0, qual_valid_st0, qual_addr_st0, qual_writeword_st0, qual_writedata_st0, qual_inst_meta_st0}),
		.out  ({is_fill_st1     , valid_st1     , addr_st1     , writeword_st1     , writedata_st1     , inst_meta_st1     })
		);

endmodule





