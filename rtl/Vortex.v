`include "VX_define.v"
`include "VX_cache_config.v"

module Vortex
	(
	input  wire           clk,
	input  wire           reset,
	input  wire[31:0] icache_response_instruction,
	output wire[31:0] icache_request_pc_address,
	// IO
	output wire        io_valid,
	output wire[31:0]  io_data,

	// DRAM Dcache Req
    output wire                              dram_req,
    output wire                              dram_req_write,
    output wire                              dram_req_read,
    output wire [31:0]                       dram_req_addr,
    output wire [31:0]                       dram_req_size,
    output wire [31:0]                       dram_req_data[`DBANK_LINE_SIZE_RNG],
    output wire [31:0]                       dram_expected_lat,

	// DRAM Dcache Res
	output wire                              dram_fill_accept,
    input  wire                              dram_fill_rsp,
    input  wire [31:0]                       dram_fill_rsp_addr,
    input  wire [31:0]                       dram_fill_rsp_data[`DBANK_LINE_SIZE_RNG],


    // Req I Mem
    output reg [31:0]  o_m_read_addr_i,
    output reg [31:0]  o_m_evict_addr_i,
    output reg         o_m_valid_i,
    output reg [31:0]  o_m_writedata_i[`ICACHE_BANKS - 1:0][`ICACHE_NUM_WORDS_PER_BLOCK-1:0],
    output reg         o_m_read_or_write_i,

    // Rsp I Mem
    input  wire [31:0] i_m_readdata_i[`ICACHE_BANKS - 1:0][`ICACHE_NUM_WORDS_PER_BLOCK-1:0],
    input  wire        i_m_ready_i,
    output wire        out_ebreak
	);


	reg[31:0] icache_banks               = `ICACHE_BANKS;
	reg[31:0] icache_num_words_per_block = `ICACHE_NUM_WORDS_PER_BLOCK;
	reg[31:0] number_threads             = `NT;
	reg[31:0] number_warps               = `NW;

	always @(posedge clk) begin
		icache_banks               <= icache_banks;
		icache_num_words_per_block <= icache_num_words_per_block;

		number_threads             <= number_threads;
		number_warps               <= number_warps;
	end

	wire memory_delay;
	wire exec_delay;
	wire gpr_stage_delay;
	wire schedule_delay;


	// Dcache Interface
	VX_gpu_dcache_res_inter #(.NUMBER_REQUESTS(`DNUMBER_REQUESTS))  VX_dcache_rsp();
	VX_gpu_dcache_req_inter #(.NUMBER_REQUESTS(`DNUMBER_REQUESTS))  VX_dcache_req();

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

	genvar wordy;
	generate
		for (wordy = 0; wordy < `DBANK_LINE_SIZE_WORDS; wordy=wordy+1) begin
			assign VX_gpu_dcache_dram_res.dram_fill_rsp_data[wordy] = dram_fill_rsp_data[wordy];
			assign dram_req_data[wordy]                             = VX_gpu_dcache_dram_req.dram_req_data[wordy];
		end
	endgenerate

	wire temp_io_valid      = (!memory_delay) && (|VX_dcache_req.core_req_valid) && (VX_dcache_req.core_req_mem_write != `NO_MEM_WRITE) && (VX_dcache_req.core_req_addr[0] == 32'h00010000);
	wire[31:0] temp_io_data = VX_dcache_req.core_req_valid[0];
	assign io_valid         = temp_io_valid;
	assign io_data          = temp_io_data;


	VX_icache_response_inter icache_response_fe();
	VX_icache_request_inter  icache_request_fe();
	VX_dram_req_rsp_inter #(
		.NUMBER_BANKS(`ICACHE_BANKS),
		.NUM_WORDS_PER_BLOCK(`ICACHE_NUM_WORDS_PER_BLOCK))    VX_dram_req_rsp_icache();

	//assign icache_response_fe.instruction = icache_response_instruction;
	assign icache_request_pc_address      = icache_request_fe.pc_address;

	assign o_m_valid_i                      = VX_dram_req_rsp_icache.o_m_valid;
	assign o_m_read_addr_i                  = VX_dram_req_rsp_icache.o_m_read_addr;
	assign o_m_evict_addr_i                 = VX_dram_req_rsp_icache.o_m_evict_addr;
	assign o_m_read_or_write_i              = VX_dram_req_rsp_icache.o_m_read_or_write;
	assign VX_dram_req_rsp_icache.i_m_ready = i_m_ready_i;
	genvar curr_bank;
	genvar curr_word;


for (curr_bank = 0; curr_bank < `ICACHE_BANKS; curr_bank = curr_bank + 1) begin : icache_setup
	for (curr_word = 0; curr_word < `ICACHE_NUM_WORDS_PER_BLOCK; curr_word = curr_word + 1) begin : icache_banks_setup
		assign o_m_writedata_i[curr_bank][curr_word]                     = VX_dram_req_rsp_icache.o_m_writedata[curr_bank][curr_word];
		assign VX_dram_req_rsp_icache.i_m_readdata[curr_bank][curr_word] = i_m_readdata_i[curr_bank][curr_word]; // fixed
	end
end

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


VX_front_end vx_front_end(
	.clk                 (clk),
	.reset               (reset),
	.VX_warp_ctl         (VX_warp_ctl),
	.VX_bckE_req         (VX_bckE_req),
	.schedule_delay      (schedule_delay),
	.icache_response_fe  (icache_response_fe),
	.icache_request_fe   (icache_request_fe),
	.VX_jal_rsp          (VX_jal_rsp),
	.VX_branch_rsp       (VX_branch_rsp),
	.fetch_ebreak        (out_ebreak)
	);

VX_scheduler schedule(
	.clk               (clk),
	.reset             (reset),
	.memory_delay      (memory_delay),
	.exec_delay        (exec_delay),
	.gpr_stage_delay   (gpr_stage_delay),
	.VX_bckE_req       (VX_bckE_req),
	.VX_writeback_inter(VX_writeback_inter),
	.schedule_delay    (schedule_delay)
	);

VX_back_end vx_back_end(
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
	.VX_gpu_dcache_dram_req   (VX_gpu_dcache_dram_req),
	.VX_gpu_dcache_dram_res   (VX_gpu_dcache_dram_res),
	.VX_dram_req_rsp_icache   (VX_dram_req_rsp_icache),
	.VX_icache_req            (icache_request_fe),
	.VX_icache_rsp            (icache_response_fe),
	.VX_dcache_req            (VX_dcache_req),
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





