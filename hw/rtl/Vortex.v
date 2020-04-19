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
`DEBUG_BEGIN
	wire scheduler_empty;
`DEBUG_END

	wire memory_delay;
	wire exec_delay;
	wire gpr_stage_delay;
	wire schedule_delay;

	// Dcache Interface
	VX_gpu_dcache_rsp_if #(.NUM_REQUESTS(`DNUM_REQUESTS))  dcache_rsp_if();
	VX_gpu_dcache_req_if #(.NUM_REQUESTS(`DNUM_REQUESTS))  dcache_req_if();
	VX_gpu_dcache_req_if #(.NUM_REQUESTS(`DNUM_REQUESTS))  dcache_req_qual_if();

	VX_gpu_dcache_dram_req_if #(.BANK_LINE_WORDS(`DBANK_LINE_WORDS)) gpu_dcache_dram_req_if();
	VX_gpu_dcache_dram_rsp_if #(.BANK_LINE_WORDS(`DBANK_LINE_WORDS)) gpu_dcache_dram_res_if();

	assign gpu_dcache_dram_res_if.dram_rsp_valid = dram_rsp_valid;
	assign gpu_dcache_dram_res_if.dram_rsp_addr  = dram_rsp_addr;

	assign dram_req_write  = gpu_dcache_dram_req_if.dram_req_write;
	assign dram_req_read   = gpu_dcache_dram_req_if.dram_req_read;
	assign dram_req_addr   = gpu_dcache_dram_req_if.dram_req_addr;
	assign dram_rsp_ready  = gpu_dcache_dram_req_if.dram_rsp_ready;

	assign gpu_dcache_dram_req_if.dram_req_full = dram_req_full;

	genvar i;
	generate
		for (i = 0; i < `DBANK_LINE_WORDS; i=i+1) begin
			assign gpu_dcache_dram_res_if.dram_rsp_data[i] = dram_rsp_data[i * 32 +: 32];
			assign dram_req_data[i * 32 +: 32]                  = gpu_dcache_dram_req_if.dram_req_data[i];
		end
	endgenerate

	wire temp_io_valid = (!memory_delay) 
					  && (|dcache_req_if.core_req_valid) 
					  && (dcache_req_if.core_req_mem_write[0] != `NO_MEM_WRITE) 
					  && (dcache_req_if.core_req_addr[0] == 32'h00010000);

	wire[31:0] temp_io_data = dcache_req_if.core_req_writedata[0];
	assign io_valid         = temp_io_valid;
	assign io_data          = temp_io_data;

	assign dcache_req_qual_if.core_req_valid     = dcache_req_if.core_req_valid & {`NUM_THREADS{~io_valid}};
	assign dcache_req_qual_if.core_req_addr      = dcache_req_if.core_req_addr;
	assign dcache_req_qual_if.core_req_writedata = dcache_req_if.core_req_writedata;
	assign dcache_req_qual_if.core_req_mem_read  = dcache_req_if.core_req_mem_read;
	assign dcache_req_qual_if.core_req_mem_write = dcache_req_if.core_req_mem_write;
	assign dcache_req_qual_if.core_req_rd        = dcache_req_if.core_req_rd;
	assign dcache_req_qual_if.core_req_wb        = dcache_req_if.core_req_wb;
	assign dcache_req_qual_if.core_req_warp_num  = dcache_req_if.core_req_warp_num;
	assign dcache_req_qual_if.core_req_pc        = dcache_req_if.core_req_pc;
	assign dcache_req_qual_if.core_no_wb_slot    = dcache_req_if.core_no_wb_slot;

	VX_gpu_dcache_rsp_if #(.NUM_REQUESTS(`INUM_REQUESTS))  icache_rsp_if();
	VX_gpu_dcache_req_if #(.NUM_REQUESTS(`INUM_REQUESTS))  icache_req_if();

	VX_gpu_dcache_dram_req_if #(.BANK_LINE_WORDS(`IBANK_LINE_WORDS)) gpu_icache_dram_req_if();
	VX_gpu_dcache_dram_rsp_if #(.BANK_LINE_WORDS(`IBANK_LINE_WORDS)) gpu_icache_dram_res_if();

	assign gpu_icache_dram_res_if.dram_rsp_valid      = I_dram_rsp_valid;
	assign gpu_icache_dram_res_if.dram_rsp_addr = I_dram_rsp_addr;

	assign I_dram_req_write  = gpu_icache_dram_req_if.dram_req_write;
	assign I_dram_req_read   = gpu_icache_dram_req_if.dram_req_read;
	assign I_dram_req_addr   = gpu_icache_dram_req_if.dram_req_addr;
	assign I_dram_rsp_ready  = gpu_icache_dram_req_if.dram_rsp_ready;

	assign gpu_icache_dram_req_if.dram_req_full = I_dram_req_full;

	genvar j;
	generate
		for (j = 0; j < `IBANK_LINE_WORDS; j = j + 1) begin
			assign gpu_icache_dram_res_if.dram_rsp_data[j] = I_dram_rsp_data[j * 32 +: 32];
			assign I_dram_req_data[j * 32 +: 32]                = gpu_icache_dram_req_if.dram_req_data[j];
		end
	endgenerate

/////////////////////////////////////////////////////////////////////////

// Front-end to Back-end
VX_frE_to_bckE_req_if    bckE_req_if(); // New instruction request to EXE/MEM

// Back-end to Front-end
VX_wb_if                 writeback_if(); // Writeback to GPRs
VX_branch_response_if    branch_rsp_if();      // Branch Resolution to Fetch
VX_jal_response_if   	jal_rsp_if();         // Jump resolution to Fetch

// CSR Buses
// VX_csr_write_request_if csr_w_req_if();

VX_warp_ctl_if        	warp_ctl_if();
VX_gpu_snp_req_rsp_if   gpu_icache_snp_req_if();
VX_gpu_snp_req_rsp_if   gpu_dcache_snp_req_if();

assign gpu_dcache_snp_req_if.snp_req_valid  = snp_req_valid;
assign gpu_dcache_snp_req_if.snp_req_addr   = snp_req_addr;
assign snp_req_full                         = gpu_dcache_snp_req_if.snp_req_full;

VX_front_end front_end(
	.clk                 (clk),
	.reset               (reset),
	.warp_ctl_if         (warp_ctl_if),
	.bckE_req_if         (bckE_req_if),
	.schedule_delay      (schedule_delay),
	.icache_rsp_if       (icache_rsp_if),
	.icache_req_if       (icache_req_if),
	.jal_rsp_if          (jal_rsp_if),
	.branch_rsp_if       (branch_rsp_if),
	.fetch_ebreak        (out_ebreak)
);

VX_scheduler schedule(
	.clk               	(clk),
	.reset             	(reset),
	.memory_delay      	(memory_delay),
	.exec_delay        	(exec_delay),
	.gpr_stage_delay   	(gpr_stage_delay),
	.bckE_req_if       	(bckE_req_if),
	.writeback_if	    (writeback_if),
	.schedule_delay    	(schedule_delay),
	.is_empty          	(scheduler_empty)
);

VX_back_end #(.CORE_ID(CORE_ID)) back_end(
	.clk                 (clk),
	.reset               (reset),
	.schedule_delay      (schedule_delay),
	.warp_ctl_if         (warp_ctl_if),
	.bckE_req_if         (bckE_req_if),
	.jal_rsp_if          (jal_rsp_if),
	.branch_rsp_if       (branch_rsp_if),
	.dcache_rsp_if       (dcache_rsp_if),
	.dcache_req_if       (dcache_req_if),
	.writeback_if        (writeback_if),
	.out_mem_delay       (memory_delay),
	.out_exec_delay      (exec_delay),
	.gpr_stage_delay     (gpr_stage_delay)
);

VX_dmem_controller dmem_controller(
	.clk                      (clk),
	.reset                    (reset),

	// Dram <-> Dcache
	.gpu_dcache_dram_req_if   (gpu_dcache_dram_req_if),
	.gpu_dcache_dram_res_if   (gpu_dcache_dram_res_if),
	.gpu_dcache_snp_req_if    (gpu_dcache_snp_req_if),

	// Dram <-> Icache
	.gpu_icache_dram_req_if   (gpu_icache_dram_req_if),
	.gpu_icache_dram_res_if   (gpu_icache_dram_res_if),
	.gpu_icache_snp_req_if    (gpu_icache_snp_req_if),

	// Core <-> Icache
	.icache_req_if            (icache_req_if),
	.icache_rsp_if            (icache_rsp_if),

	// Core <-> Dcache
	.dcache_req_if            (dcache_req_qual_if),
	.dcache_rsp_if            (dcache_rsp_if)
);

// VX_csr_handler csr_handler(
// 		.clk                  (clk),
// 		.in_decode_csr_address(decode_csr_address),
// 		.csr_w_req_if         (csr_w_req_if),
// 		.in_wb_valid          (writeback_if.wb_valid[0]),
// 		.out_decode_csr_data  (csr_decode_csr_data)
// );

endmodule // Vortex





