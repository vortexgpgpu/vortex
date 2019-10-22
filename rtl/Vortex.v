
`include "VX_define.v"


module Vortex(
	input  wire           clk,
	input  wire           reset,
	input  wire[31:0] icache_response_instruction,
	output wire[31:0] icache_request_pc_address,
	// Req
    output reg [31:0]  o_m_addr,
    output reg         o_m_valid,
    output reg [31:0]  o_m_writedata[`NUMBER_BANKS - 1:0][`NUM_WORDS_PER_BLOCK-1:0],
    output reg         o_m_read_or_write,

    // Rsp
    input  wire [31:0] i_m_readdata[`NUMBER_BANKS - 1:0][`NUM_WORDS_PER_BLOCK-1:0],
    input  wire        i_m_ready,
	// Remove Start
	input  wire[31:0]     in_cache_driver_out_data[`NT_M1:0],
	output wire[31:0]     out_cache_driver_in_address[`NT_M1:0],
	output wire[2:0]      out_cache_driver_in_mem_read,
	output wire[2:0]      out_cache_driver_in_mem_write,
	output wire           out_cache_driver_in_valid[`NT_M1:0],
	output wire[31:0]     out_cache_driver_in_data[`NT_M1:0],
	// Remove end
	output wire           out_ebreak
	);

// assign out_cache_driver_in_address   = 0;
assign out_cache_driver_in_mem_read  = `NO_MEM_READ;
assign out_cache_driver_in_mem_write = `NO_MEM_WRITE;
// assign out_cache_driver_in_valid     = 0;
// assign out_cache_driver_in_data      = 0;

// assign out_cache_driver_in_address   = VX_dcache_req.out_cache_driver_in_address;
// assign out_cache_driver_in_mem_read  = VX_dcache_req.out_cache_driver_in_mem_read;
// assign out_cache_driver_in_mem_write = VX_dcache_req.out_cache_driver_in_mem_write;
// assign out_cache_driver_in_valid     = VX_dcache_req.out_cache_driver_in_valid;
// assign out_cache_driver_in_data      = VX_dcache_req.out_cache_driver_in_data;

// assign VX_dcache_rsp.in_cache_driver_out_data = in_cache_driver_out_data;


// Dcache Interface

VX_dcache_response_inter VX_dcache_rsp();
VX_dcache_request_inter  VX_dcache_req();


VX_dram_req_rsp_inter    VX_dram_req_rsp();

assign o_m_addr          = VX_dram_req_rsp.o_m_addr;
assign o_m_valid         = VX_dram_req_rsp.o_m_valid;
assign o_m_read_or_write = VX_dram_req_rsp.o_m_read_or_write;

assign VX_dram_req_rsp.i_m_ready = i_m_ready;

genvar curr_bank;
genvar curr_word;
for (curr_bank = 0; curr_bank < `NUMBER_BANKS; curr_bank = curr_bank + 1) begin

	for (curr_word = 0; curr_word < `NUM_WORDS_PER_BLOCK; curr_word = curr_word + 1) begin

		assign o_m_writedata[curr_bank][curr_word]                = VX_dram_req_rsp.o_m_writedata[curr_bank][curr_word];
		assign VX_dram_req_rsp.i_m_readdata[curr_bank][curr_word] = i_m_readdata[curr_bank][curr_word];

	end
end

// Icache Interface

VX_icache_response_inter icache_response_fe();
VX_icache_request_inter  icache_request_fe();

assign icache_response_fe.instruction = icache_response_instruction;
assign icache_request_pc_address      = icache_request_fe.pc_address;

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


wire memory_delay;
wire gpr_stage_delay;
wire schedule_delay;


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
	.gpr_stage_delay     (gpr_stage_delay)
	);


VX_dmem_controller VX_dmem_controller(
	.clk            (clk),
	.reset          (reset),
	.VX_dram_req_rsp(VX_dram_req_rsp),
	.VX_dcache_req  (VX_dcache_req),
	.VX_dcache_rsp  (VX_dcache_rsp)
	);
// VX_csr_handler vx_csr_handler(
// 		.clk                  (clk),
// 		.in_decode_csr_address(decode_csr_address),
// 		.VX_csr_w_req         (VX_csr_w_req),
// 		.in_wb_valid          (VX_writeback_inter.wb_valid[0]),

// 		.out_decode_csr_data  (csr_decode_csr_data)
// 	);




endmodule // Vortex





