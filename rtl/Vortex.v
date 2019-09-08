
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

wire decode_clone_stall;
wire decode_branch_stall;
wire[11:0] decode_csr_address;

// From fetch
wire           fetch_delay;
wire           fetch_ebreak;
wire[`NW_M1:0] fetch_which_warp;


// From execute
wire            		execute_branch_stall;
wire[11:0]      		execute_csr_address;
wire            		execute_is_csr;
reg[31:0]       		execute_csr_result;
wire            execute_jal;
wire[31:0]      execute_jal_dest;


// From e_m_register
wire            e_m_jal;
wire[31:0]      e_m_jal_dest;
wire[11:0]      e_m_csr_address;
wire            e_m_is_csr;
wire[31:0]      e_m_csr_result;


// From memory
wire            memory_delay;

// From csr handler
wire[31:0]    csr_decode_csr_data;


// From forwarding
wire         forwarding_fwd_stall;


// Internal
wire total_freeze;

assign total_freeze = fetch_delay || memory_delay;
assign out_ebreak   = fetch_ebreak;




VX_inst_meta_inter       fe_inst_meta_fd();
VX_inst_meta_inter       fd_inst_meta_de();

VX_frE_to_bckE_req_inter VX_bckE_req();

VX_mem_req_inter         VX_exe_mem_req();
VX_mem_req_inter         VX_mem_req();


VX_inst_mem_wb_inter     VX_mem_wb();

VX_mw_wb_inter           VX_mw_wb();

VX_warp_ctl_inter        VX_warp_ctl();
VX_wb_inter              VX_writeback_inter();


VX_forward_reqeust_inter      VX_fwd_req_de();
VX_forward_exe_inter          VX_fwd_exe();
VX_forward_mem_inter          VX_fwd_mem();
VX_forward_wb_inter           VX_fwd_wb();
VX_forward_response_inter     VX_fwd_rsp();

VX_icache_response_inter icache_response_fe();
VX_icache_request_inter  icache_request_fe();


VX_branch_response_inter VX_branch_rsp();
VX_jal_response_inter    VX_jal_rsp();

assign icache_response_fe.instruction = icache_response_instruction;
assign icache_request_pc_address      = icache_request_fe.pc_address;

VX_fetch vx_fetch(
		.clk                (clk),
		.reset              (reset),
		.in_memory_delay    (memory_delay),
		.in_branch_stall    (decode_branch_stall),
		.in_fwd_stall       (forwarding_fwd_stall),
		.in_branch_stall_exe(execute_branch_stall),
		.in_clone_stall     (decode_clone_stall),
		.VX_jal_rsp         (VX_jal_rsp),
		.icache_response    (icache_response_fe),
		.VX_warp_ctl        (VX_warp_ctl),

		.icache_request     (icache_request_fe),
		.VX_branch_rsp      (VX_branch_rsp),
		.out_delay          (fetch_delay),
		.out_ebreak         (fetch_ebreak),
		.out_which_wspawn   (fetch_which_warp),
		.fe_inst_meta_fd    (fe_inst_meta_fd)
	);


VX_front_end vx_front_end(
	.clk                 (clk),
	.reset               (reset),
	.total_freeze        (total_freeze),
	.forwarding_fwd_stall(forwarding_fwd_stall),
	.fetch_which_warp    (fetch_which_warp),
	.execute_branch_stall(execute_branch_stall),
	.VX_warp_ctl         (VX_warp_ctl),
	.fe_inst_meta_fd     (fe_inst_meta_fd),
	.VX_writeback_inter  (VX_writeback_inter),
	.VX_fwd_req_de       (VX_fwd_req_de),
	.VX_fwd_rsp          (VX_fwd_rsp),
	.VX_bckE_req         (VX_bckE_req),
	.decode_clone_stall  (decode_clone_stall),
	.decode_branch_stall (decode_branch_stall),
	.decode_csr_address  (decode_csr_address)
	);


VX_execute vx_execute(
		.VX_bckE_req      (VX_bckE_req),
		.VX_fwd_exe       (VX_fwd_exe),
		.in_csr_data      (csr_decode_csr_data),

		.VX_exe_mem_req  (VX_exe_mem_req),
		.out_csr_address  (execute_csr_address),
		.out_is_csr       (execute_is_csr),
		.out_csr_result   (execute_csr_result),
		.out_jal          (execute_jal),
		.out_jal_dest     (execute_jal_dest),
		.out_branch_stall (execute_branch_stall)
	);


assign VX_jal_rsp.jal          = e_m_jal;
assign VX_jal_rsp.jal_dest     = e_m_jal_dest;
assign VX_jal_rsp.jal_warp_num = VX_mem_req.warp_num;

VX_e_m_reg vx_e_m_reg(
		.clk              (clk),
		.reset            (reset),
		.in_csr_address   (execute_csr_address),
		.in_is_csr        (execute_is_csr),
		.in_csr_result    (execute_csr_result),
		.in_jal           (execute_jal),
		.in_jal_dest      (execute_jal_dest),
		.in_freeze        (total_freeze),
		.VX_exe_mem_req   (VX_exe_mem_req),

		.VX_mem_req       (VX_mem_req),
		.out_csr_address  (e_m_csr_address),
		.out_is_csr       (e_m_is_csr),
		.out_csr_result   (e_m_csr_result),
		.out_jal          (e_m_jal),
		.out_jal_dest     (e_m_jal_dest)
	);

VX_memory vx_memory(
		.VX_mem_req                   (VX_mem_req),
		.VX_mem_wb                    (VX_mem_wb),
		.VX_fwd_mem                   (VX_fwd_mem),
		.out_delay                    (memory_delay),

		.VX_branch_rsp                (VX_branch_rsp),

		.in_cache_driver_out_data     (in_cache_driver_out_data),
		.out_cache_driver_in_address  (out_cache_driver_in_address),
		.out_cache_driver_in_mem_read (out_cache_driver_in_mem_read),
		.out_cache_driver_in_mem_write(out_cache_driver_in_mem_write),
		.out_cache_driver_in_data     (out_cache_driver_in_data),
		.out_cache_driver_in_valid    (out_cache_driver_in_valid)
	);

VX_m_w_reg vx_m_w_reg(
		.clk       (clk),
		.reset     (reset),
		.in_freeze (total_freeze),
		.VX_mem_wb (VX_mem_wb),
		.VX_mw_wb  (VX_mw_wb)
	);


VX_writeback vx_writeback(
		.VX_mw_wb          (VX_mw_wb),
		.VX_fwd_wb         (VX_fwd_wb),
		.VX_writeback_inter(VX_writeback_inter)
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
		.in_mem_csr_address   (e_m_csr_address),
		.in_mem_is_csr        (e_m_is_csr),
		.in_mem_csr_result    (e_m_csr_result),
		.in_wb_valid          (VX_mw_wb.valid[0]),

		.out_decode_csr_data  (csr_decode_csr_data)
	);




endmodule // Vortex





