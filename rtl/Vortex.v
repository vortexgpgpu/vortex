`include "VX_define.v"
`include "VX_cache_config.v"

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
	output wire[31:0]  		io_data,

	// DRAM Dcache Req
	output wire             dram_req,
	output wire             dram_req_write,
	output wire             dram_req_read,
	output wire [31:0]      dram_req_addr,
	output wire [31:0]      dram_req_size,
	output wire [31:0]      dram_req_data[`DBANK_LINE_SIZE_RNG],
	output wire [31:0]      dram_expected_lat,

	input  wire             dram_req_delay,

	// DRAM Dcache Res
	output wire             dram_fill_accept,
	input  wire             dram_fill_rsp,
	input  wire [31:0]      dram_fill_rsp_addr,
	input  wire [31:0]      dram_fill_rsp_data[`DBANK_LINE_SIZE_RNG],

	// DRAM Icache Req
	output wire                              I_dram_req,
	output wire                              I_dram_req_write,
	output wire                              I_dram_req_read,
	output wire [31:0]                       I_dram_req_addr,
	output wire [31:0]                       I_dram_req_size,
	output wire [`IBANK_LINE_SIZE_RNG][31:0] I_dram_req_data,
	output wire [31:0]                       I_dram_expected_lat,

	// DRAM Icache Res
	output wire                              I_dram_fill_accept,
	input  wire                              I_dram_fill_rsp,
	input  wire [31:0]                       I_dram_fill_rsp_addr,
	input  wire [`IBANK_LINE_SIZE_RNG][31:0] I_dram_fill_rsp_data,

	// LLC Snooping
	input  wire             snp_req,
	input  wire [31:0]      snp_req_addr,
	output wire             snp_req_delay,

	input  wire                              I_snp_req,
	input  wire [31:0]                       I_snp_req_addr,
	output wire                              I_snp_req_delay,

	output wire             out_ebreak

`else 

	input  wire           					 clk,
	input  wire           					 reset,
	// IO
	output wire        						 io_valid,
	output wire[31:0]  						 io_data,

	// DRAM Dcache Req
	output wire                              dram_req,
	output wire                              dram_req_write,
	output wire                              dram_req_read,
	output wire [31:0]                       dram_req_addr,
	output wire [31:0]                       dram_req_size,
	output wire [`DBANK_LINE_SIZE_RNG][31:0] dram_req_data,
	output wire [31:0]                       dram_expected_lat,

	// DRAM Dcache Res
	output wire                              dram_fill_accept,
	input  wire                              dram_fill_rsp,
	input  wire [31:0]                       dram_fill_rsp_addr,
	input  wire [`DBANK_LINE_SIZE_RNG][31:0] dram_fill_rsp_data,


	// DRAM Icache Req
	output wire                              I_dram_req,
	output wire                              I_dram_req_write,
	output wire                              I_dram_req_read,
	output wire [31:0]                       I_dram_req_addr,
	output wire [31:0]                       I_dram_req_size,
	output wire [`IBANK_LINE_SIZE_RNG][31:0] I_dram_req_data,
	output wire [31:0]                       I_dram_expected_lat,

	// DRAM Icache Res
	output wire                              I_dram_fill_accept,
	input  wire                              I_dram_fill_rsp,
	input  wire [31:0]                       I_dram_fill_rsp_addr,
	input  wire [`IBANK_LINE_SIZE_RNG][31:0] I_dram_fill_rsp_data,

	input  wire             dram_req_delay,

	input  wire                              snp_req,
	input  wire [31:0]                       snp_req_addr,
	output wire                              snp_req_delay,

	input  wire                              I_snp_req,
	input  wire [31:0]                       I_snp_req_addr,
	output wire                              I_snp_req_delay,

	output wire        						 out_ebreak
`endif
);

	wire scheduler_empty;
	wire out_ebreak_unqual;

	assign out_ebreak = out_ebreak_unqual && (scheduler_empty && 1);


	wire memory_delay;
	wire exec_delay;
	wire gpr_stage_delay;
	wire schedule_delay;


	// Dcache Interface
	VX_gpu_dcache_res_inter #(.NUMBER_REQUESTS(`DNUMBER_REQUESTS))  VX_dcache_rsp();
	VX_gpu_dcache_req_inter #(.NUMBER_REQUESTS(`DNUMBER_REQUESTS))  VX_dcache_req();
	VX_gpu_dcache_req_inter #(.NUMBER_REQUESTS(`DNUMBER_REQUESTS))  VX_dcache_req_qual();

	VX_gpu_dcache_dram_req_inter #(.BANK_LINE_SIZE_WORDS(`DBANK_LINE_SIZE_WORDS)) VX_gpu_dcache_dram_req();
	VX_gpu_dcache_dram_res_inter #(.BANK_LINE_SIZE_WORDS(`DBANK_LINE_SIZE_WORDS)) VX_gpu_dcache_dram_res();


	assign VX_gpu_dcache_dram_res.dram_fill_rsp      = dram_fill_rsp;
	assign VX_gpu_dcache_dram_res.dram_fill_rsp_addr = dram_fill_rsp_addr;

	assign dram_req          = VX_gpu_dcache_dram_req.dram_req;
	assign dram_req_write    = VX_gpu_dcache_dram_req.dram_req_write;
	assign dram_req_read     = VX_gpu_dcache_dram_req.dram_req_read;
	assign dram_req_addr     = VX_gpu_dcache_dram_req.dram_req_addr;
	assign dram_req_size     = VX_gpu_dcache_dram_req.dram_req_size;
	assign dram_expected_lat = `DSIMULATED_DRAM_LATENCY_CYCLES;
	assign dram_fill_accept  = VX_gpu_dcache_dram_req.dram_fill_accept;

	assign VX_gpu_dcache_dram_req.dram_req_delay = dram_req_delay;

	genvar wordy;
	generate
		for (wordy = 0; wordy < `DBANK_LINE_SIZE_WORDS; wordy=wordy+1) begin
			assign VX_gpu_dcache_dram_res.dram_fill_rsp_data[wordy] = dram_fill_rsp_data[wordy];
			assign dram_req_data[wordy]                             = VX_gpu_dcache_dram_req.dram_req_data[wordy];
		end
	endgenerate

	wire temp_io_valid      = (!memory_delay) && (|VX_dcache_req.core_req_valid) && (VX_dcache_req.core_req_mem_write[0] != `NO_MEM_WRITE) && (VX_dcache_req.core_req_addr[0] == 32'h00010000);
	wire[31:0] temp_io_data = VX_dcache_req.core_req_writedata[0];
	assign io_valid         = temp_io_valid;
	assign io_data          = temp_io_data;

	assign VX_dcache_req_qual.core_req_valid        = VX_dcache_req.core_req_valid & {`NT{~io_valid}};
	assign VX_dcache_req_qual.core_req_addr         = VX_dcache_req.core_req_addr;
	assign VX_dcache_req_qual.core_req_writedata    = VX_dcache_req.core_req_writedata;
	assign VX_dcache_req_qual.core_req_mem_read     = VX_dcache_req.core_req_mem_read;
	assign VX_dcache_req_qual.core_req_mem_write    = VX_dcache_req.core_req_mem_write;
	assign VX_dcache_req_qual.core_req_rd           = VX_dcache_req.core_req_rd;
	assign VX_dcache_req_qual.core_req_wb           = VX_dcache_req.core_req_wb;
	assign VX_dcache_req_qual.core_req_warp_num     = VX_dcache_req.core_req_warp_num;
	assign VX_dcache_req_qual.core_req_pc           = VX_dcache_req.core_req_pc;
	assign VX_dcache_req_qual.core_no_wb_slot       = VX_dcache_req.core_no_wb_slot;


	VX_gpu_dcache_res_inter #(.NUMBER_REQUESTS(`INUMBER_REQUESTS))  VX_icache_rsp();
	VX_gpu_dcache_req_inter #(.NUMBER_REQUESTS(`INUMBER_REQUESTS))  VX_icache_req();

	VX_gpu_dcache_dram_req_inter #(.BANK_LINE_SIZE_WORDS(`IBANK_LINE_SIZE_WORDS)) VX_gpu_icache_dram_req();
	VX_gpu_dcache_dram_res_inter #(.BANK_LINE_SIZE_WORDS(`IBANK_LINE_SIZE_WORDS)) VX_gpu_icache_dram_res();


	assign VX_gpu_icache_dram_res.dram_fill_rsp      = I_dram_fill_rsp;
	assign VX_gpu_icache_dram_res.dram_fill_rsp_addr = I_dram_fill_rsp_addr;

	assign I_dram_req          = VX_gpu_icache_dram_req.dram_req;
	assign I_dram_req_write    = VX_gpu_icache_dram_req.dram_req_write;
	assign I_dram_req_read     = VX_gpu_icache_dram_req.dram_req_read;
	assign I_dram_req_addr     = VX_gpu_icache_dram_req.dram_req_addr;
	assign I_dram_req_size     = VX_gpu_icache_dram_req.dram_req_size;
	assign I_dram_expected_lat = `ISIMULATED_DRAM_LATENCY_CYCLES;
	assign I_dram_fill_accept  = VX_gpu_icache_dram_req.dram_fill_accept;

	assign VX_gpu_icache_dram_req.dram_req_delay = dram_req_delay;

	genvar iwordy;
	generate
		for (iwordy = 0; iwordy < `IBANK_LINE_SIZE_WORDS; iwordy=iwordy+1) begin
			assign VX_gpu_icache_dram_res.dram_fill_rsp_data[iwordy] = I_dram_fill_rsp_data[iwordy];
			assign I_dram_req_data[iwordy]                           = VX_gpu_icache_dram_req.dram_req_data[iwordy];
		end
	endgenerate


/////////////////////////////////////////////////////////////////////////

// Front-end to Back-end
VX_frE_to_bckE_req_inter      VX_bckE_req(); // New instruction request to EXE/MEM

// Back-end to Front-end
VX_wb_inter                   VX_writeback_inter(); // Writeback to GPRs
VX_branch_response_inter      VX_branch_rsp();      // Branch Resolution to Fetch
VX_jal_response_inter         VX_jal_rsp();         // Jump resolution to Fetch

// CSR Buses
// VX_csr_write_request_inter VX_csr_w_req();


VX_warp_ctl_inter        VX_warp_ctl();


VX_gpu_snp_req_rsp           VX_gpu_icache_snp_req();
VX_gpu_snp_req_rsp           VX_gpu_dcache_snp_req();

assign VX_gpu_icache_snp_req.snp_req      = I_snp_req;
assign VX_gpu_icache_snp_req.snp_req_addr = I_snp_req_addr;
assign I_snp_req_delay                    = VX_gpu_icache_snp_req.snp_delay;

assign VX_gpu_dcache_snp_req.snp_req      = snp_req;
assign VX_gpu_dcache_snp_req.snp_req_addr = snp_req_addr;
assign snp_req_delay                      = VX_gpu_dcache_snp_req.snp_delay;

VX_front_end vx_front_end(
	.clk                 (clk),
	.reset               (reset),
	.VX_warp_ctl         (VX_warp_ctl),
	.VX_bckE_req         (VX_bckE_req),
	.schedule_delay      (schedule_delay),
	.VX_icache_rsp       (VX_icache_rsp),
	.VX_icache_req       (VX_icache_req),
	.VX_jal_rsp          (VX_jal_rsp),
	.VX_branch_rsp       (VX_branch_rsp),
	.fetch_ebreak        (out_ebreak_unqual)
	);

VX_scheduler schedule(
	.clk               (clk),
	.reset             (reset),
	.memory_delay      (memory_delay),
	.exec_delay        (exec_delay),
	.gpr_stage_delay   (gpr_stage_delay),
	.VX_bckE_req       (VX_bckE_req),
	.VX_writeback_inter(VX_writeback_inter),
	.schedule_delay    (schedule_delay),
	.is_empty          (scheduler_empty)
	);

VX_back_end #(.CORE_ID(CORE_ID)) vx_back_end(
	.clk                 (clk),
	.reset               (reset),
	.schedule_delay      (schedule_delay),
	.VX_warp_ctl         (VX_warp_ctl),
	.VX_bckE_req         (VX_bckE_req),
	.VX_jal_rsp          (VX_jal_rsp),
	.VX_branch_rsp       (VX_branch_rsp),
	.VX_dcache_rsp       (VX_dcache_rsp),
	.VX_dcache_req       (VX_dcache_req),
	.VX_writeback_inter  (VX_writeback_inter),
	.out_mem_delay       (memory_delay),
	.out_exec_delay      (exec_delay),
	.gpr_stage_delay     (gpr_stage_delay)
	);


VX_dmem_controller VX_dmem_controller(
	.clk                      (clk),
	.reset                    (reset),

	// Dram <-> Dcache
	.VX_gpu_dcache_dram_req   (VX_gpu_dcache_dram_req),
	.VX_gpu_dcache_dram_res   (VX_gpu_dcache_dram_res),
	.VX_gpu_dcache_snp_req    (VX_gpu_dcache_snp_req),

	// Dram <-> Icache
	.VX_gpu_icache_dram_req   (VX_gpu_icache_dram_req),
	.VX_gpu_icache_dram_res   (VX_gpu_icache_dram_res),
	.VX_gpu_icache_snp_req    (VX_gpu_icache_snp_req),

	// Core <-> Icache
	.VX_icache_req            (VX_icache_req),
	.VX_icache_rsp            (VX_icache_rsp),

	// Core <-> Dcache
	.VX_dcache_req            (VX_dcache_req_qual),
	.VX_dcache_rsp            (VX_dcache_rsp)
	);

// VX_csr_handler vx_csr_handler(
// 		.clk                  (clk),
// 		.in_decode_csr_address(decode_csr_address),
// 		.VX_csr_w_req         (VX_csr_w_req),
// 		.in_wb_valid          (VX_writeback_inter.wb_valid[0]),

// 		.out_decode_csr_data  (csr_decode_csr_data)
// 	);

endmodule // Vortex





