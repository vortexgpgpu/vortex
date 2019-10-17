
`include "VX_define.v"

module Vortex(
	input  wire           clk,
	input  wire           reset,
	input  wire[31:0] icache_response_instruction,
	output wire[31:0] icache_request_pc_address,
	input  wire[31:0]     in_cache_driver_out_data[`NT_M1:0],
	output wire[31:0]     out_cache_driver_in_address[`NT_M1:0],
	output wire[2:0]      out_cache_driver_in_mem_read,
	output wire[2:0]      out_cache_driver_in_mem_write,
	output wire           out_cache_driver_in_valid[`NT_M1:0],
	output wire[31:0]     out_cache_driver_in_data[`NT_M1:0],
	output wire           out_ebreak
	);

// Dcache Interface

VX_dcache_response_inter VX_dcache_rsp();
VX_dcache_request_inter VX_dcache_req();

assign out_cache_driver_in_address   = VX_dcache_req.out_cache_driver_in_address;
assign out_cache_driver_in_mem_read  = VX_dcache_req.out_cache_driver_in_mem_read;
assign out_cache_driver_in_mem_write = VX_dcache_req.out_cache_driver_in_mem_write;
assign out_cache_driver_in_valid     = VX_dcache_req.out_cache_driver_in_valid;
assign out_cache_driver_in_data      = VX_dcache_req.out_cache_driver_in_data;

assign VX_dcache_rsp.in_cache_driver_out_data = in_cache_driver_out_data;


// Icache Interface

VX_icache_response_inter icache_response_fe();
VX_icache_request_inter  icache_request_fe();

assign icache_response_fe.instruction = icache_response_instruction;
assign icache_request_pc_address      = icache_request_fe.pc_address;

/////////////////////////////////////////////////////////////////////////



// Front-end to Back-end
VX_frE_to_bckE_req_inter      VX_bckE_req(); // New instruction request to EXE/MEM
wire                          fetch_delay;


// Back-end to Front-end
VX_wb_inter                   VX_writeback_inter(); // Writeback to GPRs
VX_branch_response_inter      VX_branch_rsp();      // Branch Resolution to Fetch
VX_jal_response_inter         VX_jal_rsp();         // Jump resolution to Fetch
wire                          execute_branch_stall;
wire                          memory_delay;

// Forwarding Buses
VX_forward_reqeust_inter      VX_fwd_req_de(); // Forward request
VX_forward_response_inter     VX_fwd_rsp();    // Forward Response
VX_forward_exe_inter          VX_fwd_exe();    // Data available in EXE
VX_forward_mem_inter          VX_fwd_mem();    // Data available in MEM
VX_forward_wb_inter           VX_fwd_wb();     // Data available in WB
wire                          forwarding_fwd_stall;



// CSR Buses
VX_csr_write_request_inter VX_csr_w_req();
wire[31:0]                 csr_decode_csr_data;
wire[11:0]                 decode_csr_address;


VX_warp_ctl_inter        VX_warp_ctl();


wire out_gpr_stall;
wire schedule_delay;


VX_front_end vx_front_end(
	.clk                 (clk),
	.reset               (reset),
	.VX_warp_ctl         (VX_warp_ctl),
	.forwarding_fwd_stall(forwarding_fwd_stall),
	.execute_branch_stall(execute_branch_stall),
	.VX_bckE_req         (VX_bckE_req),
	.decode_csr_address  (decode_csr_address),
	.memory_delay        (memory_delay),
	.fetch_delay         (fetch_delay),
	.schedule_delay      (schedule_delay),
	.icache_response_fe  (icache_response_fe),
	.icache_request_fe   (icache_request_fe),
	.VX_jal_rsp          (VX_jal_rsp),
	.VX_branch_rsp       (VX_branch_rsp),
	.fetch_ebreak        (out_ebreak),
	.in_gpr_stall        (out_gpr_stall)
	);

VX_scheduler schedule(
	.clk               (clk),
	.VX_bckE_req       (VX_bckE_req),
	.VX_writeback_inter(VX_writeback_inter),
	.schedule_delay    (schedule_delay)
	);

VX_back_end vx_back_end(
	.clk                 (clk),
	.reset               (reset),
	.schedule_delay      (schedule_delay),
	.fetch_delay         (fetch_delay),
	.in_fwd_stall        (forwarding_fwd_stall),
	.VX_fwd_req_de       (VX_fwd_req_de),
	.VX_fwd_rsp          (VX_fwd_rsp),
	.VX_warp_ctl         (VX_warp_ctl),
	.VX_bckE_req         (VX_bckE_req),
	.VX_fwd_exe          (VX_fwd_exe),
	.csr_decode_csr_data (csr_decode_csr_data),
	.execute_branch_stall(execute_branch_stall),
	.VX_jal_rsp          (VX_jal_rsp),
	.VX_branch_rsp       (VX_branch_rsp),
	.VX_dcache_rsp       (VX_dcache_rsp),
	.VX_dcache_req       (VX_dcache_req),
	.VX_fwd_mem          (VX_fwd_mem),
	.VX_fwd_wb           (VX_fwd_wb),
	.VX_csr_w_req        (VX_csr_w_req),
	.VX_writeback_inter  (VX_writeback_inter),
	.out_mem_delay       (memory_delay),
	.out_gpr_stall       (out_gpr_stall)
	);

VX_forwarding vx_forwarding(
		.VX_fwd_req_de(VX_fwd_req_de),
		.VX_fwd_exe   (VX_fwd_exe),
		.VX_fwd_mem   (VX_fwd_mem),
		.VX_fwd_wb    (VX_fwd_wb),
		.VX_fwd_rsp   (VX_fwd_rsp),
		.out_fwd_stall(forwarding_fwd_stall)
	);

VX_csr_handler vx_csr_handler(
		.clk                  (clk),
		.in_decode_csr_address(decode_csr_address),
		.VX_csr_w_req         (VX_csr_w_req),
		.in_wb_valid          (VX_writeback_inter.wb_valid[0]),

		.out_decode_csr_data  (csr_decode_csr_data)
	);




endmodule // Vortex





