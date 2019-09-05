
`include "VX_define.v"
`include "buses.vh"

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

// wire[31:0] in_cache_driver_out_data[`NT_M1:0];

// assign in_cache_driver_out_data[0] = in_cache_driver_out_data_0;
// assign in_cache_driver_out_data[1] = in_cache_driver_out_data_1;


// From fetch
wire           fetch_delay;
wire           fetch_ebreak;
wire[`NW_M1:0] fetch_which_warp;


// From decode
wire            decode_branch_stall;
wire            decode_clone_stall;


// From execute
wire            execute_branch_stall;
wire[11:0]      execute_csr_address;
wire            execute_is_csr;
reg[31:0]       execute_csr_result;
wire[`NT_M1:0][31:0]      execute_a_reg_data;
wire[`NT_M1:0][31:0]      execute_b_reg_data;
wire            execute_jal;
wire[31:0]      execute_jal_dest;


// From e_m_register
wire            e_m_jal;
wire[31:0]      e_m_jal_dest;
wire[11:0]      e_m_csr_address;
wire            e_m_is_csr;
wire[31:0]      e_m_csr_result;
/* verilator lint_off UNUSED */
wire[`NT_M1:0][31:0]      e_m_a_reg_data;
wire[`NT_M1:0][31:0]      e_m_b_reg_data;
/* verilator lint_on UNUSED */


// From memory
wire            memory_delay;
wire            memory_branch_dir;
wire[31:0]      memory_branch_dest;

// From m_w_register
wire[`NT_M1:0][31:0]      m_w_alu_result;
wire[`NT_M1:0][31:0]      m_w_mem_result;
wire[4:0]       m_w_rd;
wire[1:0]       m_w_wb;
/* verilator lint_off UNUSED */
wire[4:0]       m_w_rs1;
wire[4:0]       m_w_rs2;
/* verilator lint_on UNUSED */
wire[31:0]      m_w_PC_next;
wire[`NT_M1:0]  m_w_valid;
wire[`NW_M1:0]  m_w_warp_num;

// From csr handler
wire[31:0]    csr_decode_csr_data;


// From forwarding
wire         forwarding_fwd_stall;
wire         forwarding_src1_fwd;
wire         forwarding_src2_fwd;
/* verilator lint_off UNUSED */
wire         forwarding_csr_fwd;
wire[31:0]   forwarding_csr_fwd_data;
/* verilator lint_on UNUSED */
wire[`NT_M1:0][31:0]   forwarding_src1_fwd_data;
wire[`NT_M1:0][31:0]   forwarding_src2_fwd_data;


// Internal
wire total_freeze;
wire interrupt;
wire debug;

assign debug        = 1'b0;
assign interrupt    = 1'b0;
assign total_freeze = fetch_delay || memory_delay;
assign out_ebreak   = fetch_ebreak;


icache_response_t icache_response_fe;
icache_request_t  icache_request_fe;

VX_inst_meta_inter       fe_inst_meta_fd();
VX_inst_meta_inter       fd_inst_meta_de();

VX_frE_to_bckE_req_inter VX_frE_to_bckE_req();
VX_frE_to_bckE_req_inter VX_bckE_req();

VX_mem_req_inter         VX_exe_mem_req();
VX_mem_req_inter         VX_mem_req();


VX_inst_mem_wb_inter     VX_mem_wb();

VX_warp_ctl_inter        VX_warp_ctl();
VX_wb_inter              VX_writeback_inter();

assign icache_response_fe.instruction = icache_response_instruction;
assign icache_request_pc_address      = icache_request_fe.pc_address;

VX_fetch vx_fetch(
		.clk                (clk),
		.reset              (reset),
		.in_branch_dir      (memory_branch_dir),
		.in_freeze          (total_freeze),
		.in_branch_dest     (memory_branch_dest),
		.in_branch_stall    (decode_branch_stall),
		.in_fwd_stall       (forwarding_fwd_stall),
		.in_branch_stall_exe(execute_branch_stall),
		.in_clone_stall     (decode_clone_stall),
		.in_jal             (e_m_jal),
		.in_jal_dest        (e_m_jal_dest),
		.in_interrupt       (interrupt),
		.in_debug           (debug),
		.in_memory_warp_num (VX_mem_wb.warp_num),
		.icache_response    (icache_response_fe),
		.VX_warp_ctl        (VX_warp_ctl),

		.icache_request     (icache_request_fe),
		.out_delay          (fetch_delay),
		.out_ebreak         (fetch_ebreak),
		.out_which_wspawn   (fetch_which_warp),
		.fe_inst_meta_fd    (fe_inst_meta_fd)
	);


VX_f_d_reg vx_f_d_reg(
		.clk            (clk),
		.reset          (reset),
		.in_fwd_stall   (forwarding_fwd_stall),
		.in_freeze      (total_freeze),
		.in_clone_stall (decode_clone_stall),
		.fe_inst_meta_fd(fe_inst_meta_fd),
		.fd_inst_meta_de(fd_inst_meta_de)
	);


VX_decode vx_decode(
		.clk             (clk),
		.fd_inst_meta_de (fd_inst_meta_de),
		.VX_writeback_inter(VX_writeback_inter),
		.in_src1_fwd     (forwarding_src1_fwd),
		.in_src1_fwd_data(forwarding_src1_fwd_data),
		.in_src2_fwd     (forwarding_src2_fwd),
		.in_src2_fwd_data(forwarding_src2_fwd_data),
		.in_which_wspawn (fetch_which_warp),

		.VX_frE_to_bckE_req(VX_frE_to_bckE_req),
		.VX_warp_ctl       (VX_warp_ctl),
		.out_clone_stall   (decode_clone_stall),
		.out_branch_stall  (decode_branch_stall)
	);


VX_d_e_reg vx_d_e_reg(
		.clk            (clk),
		.reset          (reset),
		.in_fwd_stall   (forwarding_fwd_stall),
		.in_branch_stall(execute_branch_stall),
		.in_freeze      (total_freeze),
		.in_clone_stall (decode_clone_stall),
		.VX_frE_to_bckE_req(VX_frE_to_bckE_req),
		.VX_bckE_req       (VX_bckE_req)
	);

VX_execute vx_execute(
		.VX_bckE_req      (VX_bckE_req),
		.in_csr_data      (csr_decode_csr_data),

		.VX_exe_mem_req  (VX_exe_mem_req),
		.out_csr_address  (execute_csr_address),
		.out_is_csr       (execute_is_csr),
		.out_csr_result   (execute_csr_result),
		.out_jal          (execute_jal),
		.out_jal_dest     (execute_jal_dest),
		.out_branch_stall (execute_branch_stall),
		.out_a_reg_data   (execute_a_reg_data),
		.out_b_reg_data   (execute_b_reg_data)
	);

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
		.in_a_reg_data    (execute_a_reg_data),
		.in_b_reg_data    (execute_b_reg_data),

		.VX_mem_req       (VX_mem_req),
		.out_csr_address  (e_m_csr_address),
		.out_is_csr       (e_m_is_csr),
		.out_csr_result   (e_m_csr_result),
		.out_a_reg_data   (e_m_a_reg_data),
		.out_b_reg_data   (e_m_b_reg_data),
		.out_jal          (e_m_jal),
		.out_jal_dest     (e_m_jal_dest)
	);

// wire[31:0]  use_rd2[`NT_M1:0];

// assign use_rd2[0] = e_m_reg_data[1];
// assign use_rd2[1] = e_m_reg_data[3];

VX_memory vx_memory(
		.VX_mem_req                   (VX_mem_req),
		.VX_mem_wb                    (VX_mem_wb),

		.out_delay                    (memory_delay),

		.out_branch_dir               (memory_branch_dir),
		.out_branch_dest              (memory_branch_dest),

		.in_cache_driver_out_data     (in_cache_driver_out_data),
		.out_cache_driver_in_address  (out_cache_driver_in_address),
		.out_cache_driver_in_mem_read (out_cache_driver_in_mem_read),
		.out_cache_driver_in_mem_write(out_cache_driver_in_mem_write),
		.out_cache_driver_in_data     (out_cache_driver_in_data),
		.out_cache_driver_in_valid    (out_cache_driver_in_valid)
	);

VX_m_w_reg vx_m_w_reg(
		.clk           (clk),
		.reset         (reset),
		.VX_mem_wb     (VX_mem_wb),
		.in_freeze     (total_freeze),


		.out_alu_result(m_w_alu_result),
		.out_mem_result(m_w_mem_result),
		.out_rd        (m_w_rd),
		.out_wb        (m_w_wb),
		.out_rs1       (m_w_rs1),
		.out_rs2       (m_w_rs2),
		.out_PC_next   (m_w_PC_next),
		.out_valid     (m_w_valid),
		.out_warp_num  (m_w_warp_num)
	);


VX_writeback vx_writeback(
		.clk           (clk),
		.in_alu_result (m_w_alu_result),
		.in_mem_result (m_w_mem_result),
		.in_rd         (m_w_rd),
		.in_wb         (m_w_wb),
		.in_PC_next    (m_w_PC_next),
		.in_valid      (m_w_valid),
		.in_warp_num   (m_w_warp_num),
		.VX_writeback_inter(VX_writeback_inter)
	);


VX_forwarding vx_forwarding(
		.in_decode_src1         (VX_frE_to_bckE_req.rs1),
		.in_decode_src2         (VX_frE_to_bckE_req.rs2),
		.in_decode_csr_address  (VX_frE_to_bckE_req.csr_address),
		.in_decode_warp_num     (VX_frE_to_bckE_req.warp_num),

		.in_execute_dest        (VX_exe_mem_req.rd),
		.in_execute_wb          (VX_exe_mem_req.wb),
		.in_execute_alu_result  (VX_exe_mem_req.alu_result),
		.in_execute_PC_next     (VX_exe_mem_req.PC_next),
		.in_execute_is_csr      (execute_is_csr),
		.in_execute_csr_address (execute_csr_address),
		.in_execute_warp_num    (VX_exe_mem_req.warp_num),

		.in_memory_dest         (VX_mem_wb.rd),
		.in_memory_wb           (VX_mem_wb.wb),
		.in_memory_alu_result   (VX_mem_wb.alu_result),
		.in_memory_mem_data     (VX_mem_wb.mem_result),
		.in_memory_PC_next      (VX_mem_wb.PC_next),
		.in_memory_is_csr       (e_m_is_csr),
		.in_memory_csr_address  (e_m_csr_address),
		.in_memory_csr_result   (e_m_csr_result),
		.in_memory_warp_num     (VX_mem_wb.warp_num),

		.in_writeback_dest      (m_w_rd),
		.in_writeback_wb        (m_w_wb),
		.in_writeback_alu_result(m_w_alu_result),
		.in_writeback_mem_data  (m_w_mem_result),
		.in_writeback_PC_next   (m_w_PC_next),
		.in_writeback_warp_num  (VX_writeback_inter.wb_warp_num),

		.out_src1_fwd           (forwarding_src1_fwd),
		.out_src2_fwd           (forwarding_src2_fwd),
		.out_csr_fwd            (forwarding_csr_fwd),
		.out_src1_fwd_data      (forwarding_src1_fwd_data),
		.out_src2_fwd_data      (forwarding_src2_fwd_data),
		.out_csr_fwd_data       (forwarding_csr_fwd_data),
		.out_fwd_stall          (forwarding_fwd_stall)
	);

VX_csr_handler vx_csr_handler(
		.clk                  (clk),
		.in_decode_csr_address(VX_frE_to_bckE_req.csr_address),
		.in_mem_csr_address   (e_m_csr_address),
		.in_mem_is_csr        (e_m_is_csr),
		.in_mem_csr_result    (e_m_csr_result),
		.in_wb_valid          (m_w_valid[0]),

		.out_decode_csr_data  (csr_decode_csr_data)
	);




endmodule // Vortex





