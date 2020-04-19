`include "VX_define.vh"
`include "VX_cache_config.vh"

module Vortex
	#( 
		parameter CORE_ID = 0
	) (		
`ifdef SINGLE_CORE_BENCH

	// Clock
	input  wire           	clk,
	input  wire           	reset,

	// IO
	output wire        		io_valid,
	output wire [31:0]  	io_data,

	// DRAM Dcache Req
	output wire             			dram_req_read,
	output wire             			dram_req_write,	
	output wire [31:0]      			dram_req_addr,
	output wire [`DBANK_LINE_SIZE-1:0] 	dram_req_data,
	input  wire             			dram_req_full,

	// DRAM Dcache Rsp	
	input  wire             			dram_rsp_valid,
	input  wire [31:0]      			dram_rsp_addr,
	input  wire [`DBANK_LINE_SIZE-1:0]  dram_rsp_data,
	output wire             			dram_rsp_ready,

	// DRAM Icache Req
	output wire                         I_dram_req_read,
	output wire                         I_dram_req_write,	
	output wire [31:0]                  I_dram_req_addr,
	output wire [`IBANK_LINE_SIZE-1:0]  I_dram_req_data,
	input  wire             			I_dram_req_full,

	// DRAM Icache Rsp	
	input  wire                         I_dram_rsp_valid,
	input  wire [31:0]                  I_dram_rsp_addr,
	input  wire [`IBANK_LINE_SIZE-1:0] 	I_dram_rsp_data,
	output wire                         I_dram_rsp_ready,

	// LLC Snooping
	input  wire             			snp_req_valid,
	input  wire [31:0]      			snp_req_addr,
	output wire             			snp_req_full,

	output wire          			    out_ebreak

`else 

	input  wire           				clk,
	input  wire           				reset,
	// IO
	output wire        					io_valid,
	output wire[31:0]  					io_data,

	// DRAM Dcache Req
	output wire                         dram_req_read,
	output wire                         dram_req_write,	
	output wire [31:0]                  dram_req_addr,
	output wire [`DBANK_LINE_SIZE-1:0]  dram_req_data,
	input  wire                         dram_req_full,

	// DRAM Dcache Rsp	
	input  wire                         dram_rsp_valid,
	input  wire [31:0]                  dram_rsp_addr,
	input  wire [`DBANK_LINE_SIZE-1:0]  dram_rsp_data,
	output wire                         dram_rsp_ready,

	// DRAM Icache Req
	output wire                         I_dram_req_read,
	output wire                         I_dram_req_write,	
	output wire [31:0]                  I_dram_req_addr,
	output wire [`IBANK_LINE_SIZE-1:0]  I_dram_req_data,
	input  wire                         I_dram_req_full,

	// DRAM Icache Rsp
	output wire                         I_dram_rsp_ready,
	input  wire                         I_dram_rsp_valid,
	input  wire [31:0]                  I_dram_rsp_addr,
	input  wire [`IBANK_LINE_SIZE-1:0]  I_dram_rsp_data,	

	input  wire                         snp_req_valid,
	input  wire [31:0]                  snp_req_addr,
	output wire                         snp_req_full,

	output wire        					out_ebreak
`endif
);
/* verilator lint_off UNUSED */
	wire scheduler_empty;
/* verilator lint_on UNUSED */

	wire memory_delay;
	wire exec_delay;
	wire gpr_stage_delay;
	wire schedule_delay;

	// Dcache Interface
	VX_gpu_dcache_rsp_inter #(.NUM_REQUESTS(`DNUM_REQUESTS))  vx_dcache_rsp();
	VX_gpu_dcache_req_inter #(.NUM_REQUESTS(`DNUM_REQUESTS))  vx_dcache_req();
	VX_gpu_dcache_req_inter #(.NUM_REQUESTS(`DNUM_REQUESTS))  vx_dcache_req_qual();

	VX_gpu_dcache_dram_req_inter #(.BANK_LINE_WORDS(`DBANK_LINE_WORDS)) vx_gpu_dcache_dram_req();
	VX_gpu_dcache_dram_rsp_inter #(.BANK_LINE_WORDS(`DBANK_LINE_WORDS)) vx_gpu_dcache_dram_res();

	assign vx_gpu_dcache_dram_res.dram_rsp_valid = dram_rsp_valid;
	assign vx_gpu_dcache_dram_res.dram_rsp_addr  = dram_rsp_addr;

	assign dram_req_write  = vx_gpu_dcache_dram_req.dram_req_write;
	assign dram_req_read   = vx_gpu_dcache_dram_req.dram_req_read;
	assign dram_req_addr   = vx_gpu_dcache_dram_req.dram_req_addr;
	assign dram_rsp_ready  = vx_gpu_dcache_dram_req.dram_rsp_ready;

	assign vx_gpu_dcache_dram_req.dram_req_full = dram_req_full;

	genvar i;
	generate
		for (i = 0; i < `DBANK_LINE_WORDS; i=i+1) begin
			assign vx_gpu_dcache_dram_res.dram_rsp_data[i] = dram_rsp_data[i * 32 +: 32];
			assign dram_req_data[i * 32 +: 32]                  = vx_gpu_dcache_dram_req.dram_req_data[i];
		end
	endgenerate

	wire temp_io_valid = (!memory_delay) 
					  && (|vx_dcache_req.core_req_valid) 
					  && (vx_dcache_req.core_req_mem_write[0] != `NO_MEM_WRITE) 
					  && (vx_dcache_req.core_req_addr[0] == 32'h00010000);

	wire[31:0] temp_io_data = vx_dcache_req.core_req_writedata[0];
	assign io_valid         = temp_io_valid;
	assign io_data          = temp_io_data;

	assign vx_dcache_req_qual.core_req_valid     = vx_dcache_req.core_req_valid & {`NUM_THREADS{~io_valid}};
	assign vx_dcache_req_qual.core_req_addr      = vx_dcache_req.core_req_addr;
	assign vx_dcache_req_qual.core_req_writedata = vx_dcache_req.core_req_writedata;
	assign vx_dcache_req_qual.core_req_mem_read  = vx_dcache_req.core_req_mem_read;
	assign vx_dcache_req_qual.core_req_mem_write = vx_dcache_req.core_req_mem_write;
	assign vx_dcache_req_qual.core_req_rd        = vx_dcache_req.core_req_rd;
	assign vx_dcache_req_qual.core_req_wb        = vx_dcache_req.core_req_wb;
	assign vx_dcache_req_qual.core_req_warp_num  = vx_dcache_req.core_req_warp_num;
	assign vx_dcache_req_qual.core_req_pc        = vx_dcache_req.core_req_pc;
	assign vx_dcache_req_qual.core_no_wb_slot    = vx_dcache_req.core_no_wb_slot;

	VX_gpu_dcache_rsp_inter #(.NUM_REQUESTS(`INUM_REQUESTS))  vx_icache_rsp();
	VX_gpu_dcache_req_inter #(.NUM_REQUESTS(`INUM_REQUESTS))  vx_icache_req();

	VX_gpu_dcache_dram_req_inter #(.BANK_LINE_WORDS(`IBANK_LINE_WORDS)) vx_gpu_icache_dram_req();
	VX_gpu_dcache_dram_rsp_inter #(.BANK_LINE_WORDS(`IBANK_LINE_WORDS)) vx_gpu_icache_dram_res();

	assign vx_gpu_icache_dram_res.dram_rsp_valid      = I_dram_rsp_valid;
	assign vx_gpu_icache_dram_res.dram_rsp_addr = I_dram_rsp_addr;

	assign I_dram_req_write  = vx_gpu_icache_dram_req.dram_req_write;
	assign I_dram_req_read   = vx_gpu_icache_dram_req.dram_req_read;
	assign I_dram_req_addr   = vx_gpu_icache_dram_req.dram_req_addr;
	assign I_dram_rsp_ready  = vx_gpu_icache_dram_req.dram_rsp_ready;

	assign vx_gpu_icache_dram_req.dram_req_full = I_dram_req_full;

	genvar j;
	generate
		for (j = 0; j < `IBANK_LINE_WORDS; j = j + 1) begin
			assign vx_gpu_icache_dram_res.dram_rsp_data[j] = I_dram_rsp_data[j * 32 +: 32];
			assign I_dram_req_data[j * 32 +: 32]                = vx_gpu_icache_dram_req.dram_req_data[j];
		end
	endgenerate

/////////////////////////////////////////////////////////////////////////

// Front-end to Back-end
VX_frE_to_bckE_req_inter    vx_bckE_req(); // New instruction request to EXE/MEM

// Back-end to Front-end
VX_wb_inter                 vx_writeback_inter(); // Writeback to GPRs
VX_branch_response_inter    vx_branch_rsp();      // Branch Resolution to Fetch
VX_jal_response_inter   	vx_jal_rsp();         // Jump resolution to Fetch

// CSR Buses
// VX_csr_write_request_inter vx_csr_w_req();

VX_warp_ctl_inter        	vx_warp_ctl();
VX_gpu_snp_req_rsp          vx_gpu_icache_snp_req();
VX_gpu_snp_req_rsp          vx_gpu_dcache_snp_req();

assign vx_gpu_dcache_snp_req.snp_req_valid  = snp_req_valid;
assign vx_gpu_dcache_snp_req.snp_req_addr   = snp_req_addr;
assign snp_req_full                        = vx_gpu_dcache_snp_req.snp_req_full;

VX_front_end vx_front_end(
	.clk                 (clk),
	.reset               (reset),
	.vx_warp_ctl         (vx_warp_ctl),
	.vx_bckE_req         (vx_bckE_req),
	.schedule_delay      (schedule_delay),
	.vx_icache_rsp       (vx_icache_rsp),
	.vx_icache_req       (vx_icache_req),
	.vx_jal_rsp          (vx_jal_rsp),
	.vx_branch_rsp       (vx_branch_rsp),
	.fetch_ebreak        (out_ebreak)
);

VX_scheduler schedule(
	.clk               	(clk),
	.reset             	(reset),
	.memory_delay      	(memory_delay),
	.exec_delay        	(exec_delay),
	.gpr_stage_delay   	(gpr_stage_delay),
	.vx_bckE_req       	(vx_bckE_req),
	.vx_writeback_inter	(vx_writeback_inter),
	.schedule_delay    	(schedule_delay),
	.is_empty          	(scheduler_empty)
);

VX_back_end #(.CORE_ID(CORE_ID)) vx_back_end(
	.clk                 (clk),
	.reset               (reset),
	.schedule_delay      (schedule_delay),
	.vx_warp_ctl         (vx_warp_ctl),
	.vx_bckE_req         (vx_bckE_req),
	.vx_jal_rsp          (vx_jal_rsp),
	.vx_branch_rsp       (vx_branch_rsp),
	.vx_dcache_rsp       (vx_dcache_rsp),
	.vx_dcache_req       (vx_dcache_req),
	.vx_writeback_inter  (vx_writeback_inter),
	.out_mem_delay       (memory_delay),
	.out_exec_delay      (exec_delay),
	.gpr_stage_delay     (gpr_stage_delay)
);

VX_dmem_controller vx_dmem_controller(
	.clk                      (clk),
	.reset                    (reset),

	// Dram <-> Dcache
	.vx_gpu_dcache_dram_req   (vx_gpu_dcache_dram_req),
	.vx_gpu_dcache_dram_res   (vx_gpu_dcache_dram_res),
	.vx_gpu_dcache_snp_req    (vx_gpu_dcache_snp_req),

	// Dram <-> Icache
	.vx_gpu_icache_dram_req   (vx_gpu_icache_dram_req),
	.vx_gpu_icache_dram_res   (vx_gpu_icache_dram_res),
	.vx_gpu_icache_snp_req    (vx_gpu_icache_snp_req),

	// Core <-> Icache
	.vx_icache_req            (vx_icache_req),
	.vx_icache_rsp            (vx_icache_rsp),

	// Core <-> Dcache
	.vx_dcache_req            (vx_dcache_req_qual),
	.vx_dcache_rsp            (vx_dcache_rsp)
);

// VX_csr_handler vx_csr_handler(
// 		.clk                  (clk),
// 		.in_decode_csr_address(decode_csr_address),
// 		.vx_csr_w_req         (vx_csr_w_req),
// 		.in_wb_valid          (vx_writeback_inter.wb_valid[0]),
// 		.out_decode_csr_data  (csr_decode_csr_data)
// );

endmodule // Vortex





