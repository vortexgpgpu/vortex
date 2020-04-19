`include "VX_define.vh"

module VX_gpr_stage (
	input wire                 clk,
	input wire                 reset,
	input wire                 schedule_delay,

	input  wire                memory_delay,
	input  wire  			   exec_delay,
	input  wire                stall_gpr_csr,
	output wire                gpr_stage_delay,

	// inputs
	// Instruction Information
	VX_frE_to_bckE_req_if   bckE_req_if,

	// WriteBack inputs
	VX_wb_if                writeback_if,

	// Outputs
	VX_exec_unit_req_if   exec_unit_req_if,
	VX_lsu_req_if         lsu_req_if,
	VX_gpu_inst_req_if    gpu_inst_req_if,
	VX_csr_req_if         csr_req_if
);
`DEBUG_BEGIN
	wire[31:0] curr_PC = bckE_req_if.curr_PC;
	wire[2:0] branchType = bckE_req_if.branch_type;
	wire is_store = (bckE_req_if.mem_write != `NO_MEM_WRITE);
	wire is_load  = (bckE_req_if.mem_read  != `NO_MEM_READ);
	wire jalQual = bckE_req_if.jalQual;
`DEBUG_END

	VX_gpr_read_if gpr_read_if();
	assign gpr_read_if.rs1      = bckE_req_if.rs1;
	assign gpr_read_if.rs2      = bckE_req_if.rs2;
	assign gpr_read_if.warp_num = bckE_req_if.warp_num;

`ifndef ASIC
	VX_gpr_jal_if gpr_jal_if();
	assign gpr_jal_if.is_jal  = bckE_req_if.jalQual;
	assign gpr_jal_if.curr_PC = bckE_req_if.curr_PC;
`else 
	VX_gpr_jal_if gpr_jal_if();
	assign gpr_jal_if.is_jal  = exec_unit_req_if.jalQual;
	assign gpr_jal_if.curr_PC = exec_unit_req_if.curr_PC;
`endif

	VX_gpr_data_if           gpr_datf_if();

	VX_gpr_wrapper grp_wrapper (
		.clk               (clk),
		.reset             (reset),
		.writeback_if(writeback_if),
		.gpr_read_if       (gpr_read_if),
		.gpr_jal_if        (gpr_jal_if),

		.out_a_reg_data (gpr_datf_if.a_reg_data),
		.out_b_reg_data (gpr_datf_if.b_reg_data)
	);

	// assign bckE_req_if.is_csr   = is_csr;
	// assign bckE_req_out_if.csr_mask = (bckE_req_if.sr_immed == 1'b1) ?  {27'h0, bckE_req_if.rs1} : gpr_data_if.a_reg_data[0];

	// Outputs
	VX_exec_unit_req_if   exec_unit_req_temp_if();
	VX_lsu_req_if         lsu_req_temp_if();
	VX_gpu_inst_req_if    gpu_inst_req_temp_if();
	VX_csr_req_if         csr_req_temp_if();

	VX_inst_multiplex inst_mult(
		.bckE_req_if     (bckE_req_if),
		.gpr_data_if     (gpr_datf_if),
		.exec_unit_req_if(exec_unit_req_temp_if),
		.lsu_req_if      (lsu_req_temp_if),
		.gpu_inst_req_if (gpu_inst_req_temp_if),
		.csr_req_if      (csr_req_temp_if)
	);
`DEBUG_BEGIN
	wire is_lsu = (|lsu_req_temp_if.valid);
`DEBUG_END
	wire stall_rest = 0;
	wire flush_rest = schedule_delay;

	wire stall_lsu  = memory_delay;
	wire flush_lsu  = schedule_delay && !stall_lsu;

	wire stall_exec  = exec_delay;
	wire flush_exec  = schedule_delay && !stall_exec;

	wire stall_csr = stall_gpr_csr && bckE_req_if.is_csr && (|bckE_req_if.valid);

	assign gpr_stage_delay = stall_lsu || stall_exec || stall_csr;

`ifdef ASIC
		wire delayed_lsu_last_cycle;

		VX_generic_register #(
			.N(1)
		) delayed_reg (
			.clk  (clk),
			.reset(reset),
			.stall(stall_rest),
			.flush(stall_rest),
			.in   (stall_lsu),
			.out  (delayed_lsu_last_cycle)
		);

		wire[`NUM_THREADS-1:0][31:0] temp_store_data;
		wire[`NUM_THREADS-1:0][31:0] temp_base_address; // A reg data

		wire[`NUM_THREADS-1:0][31:0] real_store_data;
		wire[`NUM_THREADS-1:0][31:0] real_base_address; // A reg data

		wire store_curr_real = !delayed_lsu_last_cycle && stall_lsu;

		VX_generic_register #(
			.N(`NUM_THREADS*32*2)
		) lsu_data (
			.clk  (clk),
			.reset(reset),
			.stall(!store_curr_real),
			.flush(stall_rest),
			.in   ({real_store_data, real_base_address}),
			.out  ({temp_store_data, temp_base_address})
		);

		assign real_store_data   = lsu_req_temp_if.store_data;
		assign real_base_address = lsu_req_temp_if.base_address;

		assign lsu_req_if.store_data   = (delayed_lsu_last_cycle) ? temp_store_data   : real_store_data;
		assign lsu_req_if.base_address = (delayed_lsu_last_cycle) ? temp_base_address : real_base_address;

		VX_generic_register #(
			.N(77 + `NW_BITS-1 + 1 + (`NUM_THREADS))
		) lsu_reg (
			.clk  (clk),
			.reset(reset),
			.stall(stall_lsu),
			.flush(flush_lsu),
			.in   ({lsu_req_temp_if.valid, lsu_req_temp_if.lsu_pc, lsu_req_temp_if.warp_num, lsu_req_temp_if.offset, lsu_req_temp_if.mem_read, lsu_req_temp_if.mem_write, lsu_req_temp_if.rd, lsu_req_temp_if.wb}),
			.out  ({lsu_req_if.valid     , lsu_req_if.lsu_pc     ,lsu_req_if.warp_num     , lsu_req_if.offset     , lsu_req_if.mem_read     , lsu_req_if.mem_write     , lsu_req_if.rd     , lsu_req_if.wb     })
		);

		VX_generic_register #(
			.N(224 + `NW_BITS-1 + 1 + (`NUM_THREADS))
		) exec_unit_reg (
			.clk  (clk),
			.reset(reset),
			.stall(stall_exec),
			.flush(flush_exec),
			.in   ({exec_unit_req_temp_if.valid, exec_unit_req_temp_if.warp_num, exec_unit_req_temp_if.curr_PC, exec_unit_req_temp_if.PC_next, exec_unit_req_temp_if.rd, exec_unit_req_temp_if.wb, exec_unit_req_temp_if.alu_op, exec_unit_req_temp_if.rs1, exec_unit_req_temp_if.rs2, exec_unit_req_temp_if.rs2_src, exec_unit_req_temp_if.itype_immed, exec_unit_req_temp_if.upper_immed, exec_unit_req_temp_if.branch_type, exec_unit_req_temp_if.jalQual, exec_unit_req_temp_if.jal, exec_unit_req_temp_if.jal_offset, exec_unit_req_temp_if.ebreak, exec_unit_req_temp_if.wspawn, exec_unit_req_temp_if.is_csr, exec_unit_req_temp_if.csr_address, exec_unit_req_temp_if.csr_immed, exec_unit_req_temp_if.csr_mask}),
			.out  ({exec_unit_req_if.valid     , exec_unit_req_if.warp_num     , exec_unit_req_if.curr_PC     , exec_unit_req_if.PC_next     , exec_unit_req_if.rd     , exec_unit_req_if.wb     , exec_unit_req_if.alu_op     , exec_unit_req_if.rs1     , exec_unit_req_if.rs2     , exec_unit_req_if.rs2_src     , exec_unit_req_if.itype_immed     , exec_unit_req_if.upper_immed     , exec_unit_req_if.branch_type     , exec_unit_req_if.jalQual     , exec_unit_req_if.jal     , exec_unit_req_if.jal_offset     , exec_unit_req_if.ebreak     , exec_unit_req_if.wspawn     , exec_unit_req_if.is_csr     , exec_unit_req_if.csr_address     , exec_unit_req_if.csr_immed     , exec_unit_req_if.csr_mask     })
		);

		assign exec_unit_req_if.a_reg_data = real_base_address;
		assign exec_unit_req_if.b_reg_data = real_store_data;

		VX_generic_register #(
			.N(36 + `NW_BITS-1 + 1 + (`NUM_THREADS))
		) gpu_inst_reg (
			.clk  (clk),
			.reset(reset),
			.stall(stall_rest),
			.flush(flush_rest),
			.in   ({gpu_inst_req_temp_if.valid, gpu_inst_req_temp_if.warp_num, gpu_inst_req_temp_if.is_wspawn, gpu_inst_req_temp_if.is_tmc, gpu_inst_req_temp_if.is_split, gpu_inst_req_temp_if.is_barrier, gpu_inst_req_temp_if.pc_next}),
			.out  ({gpu_inst_req_if.valid     , gpu_inst_req_if.warp_num     , gpu_inst_req_if.is_wspawn     , gpu_inst_req_if.is_tmc     , gpu_inst_req_if.is_split     , gpu_inst_req_if.is_barrier     , gpu_inst_req_if.pc_next     })
		);

		assign gpu_inst_req_if.a_reg_data = real_base_address;
		assign gpu_inst_req_if.rd2        = real_store_data;

		VX_generic_register #(
			.N(`NW_BITS-1  + 1 + `NUM_THREADS + 58)
		) csr_reg (
			.clk  (clk),
			.reset(reset),
			.stall(stall_gpr_csr),
			.flush(flush_rest),
			.in   ({csr_req_temp_if.valid, csr_req_temp_if.warp_num, csr_req_temp_if.rd, csr_req_temp_if.wb, csr_req_temp_if.alu_op, csr_req_temp_if.is_csr, csr_req_temp_if.csr_address, csr_req_temp_if.csr_immed, csr_req_temp_if.csr_mask}),
			.out  ({csr_req_if.valid     , csr_req_if.warp_num     , csr_req_if.rd     , csr_req_if.wb     , csr_req_if.alu_op     , csr_req_if.is_csr     , csr_req_if.csr_address     , csr_req_if.csr_immed     , csr_req_if.csr_mask     })
		);

`else 

    // 341 
	VX_generic_register #(
		.N(77 + `NW_BITS-1 + 1 + 65*(`NUM_THREADS))
	) lsu_reg (
		.clk  (clk),
		.reset(reset),
		.stall(stall_lsu),
		.flush(flush_lsu),
		.in   ({lsu_req_temp_if.valid, lsu_req_temp_if.lsu_pc, lsu_req_temp_if.warp_num, lsu_req_temp_if.store_data, lsu_req_temp_if.base_address, lsu_req_temp_if.offset, lsu_req_temp_if.mem_read, lsu_req_temp_if.mem_write, lsu_req_temp_if.rd, lsu_req_temp_if.wb}),
		.out  ({lsu_req_if.valid     , lsu_req_if.lsu_pc     , lsu_req_if.warp_num     , lsu_req_if.store_data     , lsu_req_if.base_address     , lsu_req_if.offset     , lsu_req_if.mem_read     , lsu_req_if.mem_write     , lsu_req_if.rd     , lsu_req_if.wb     })
	);

	VX_generic_register #(
		.N(224 + `NW_BITS-1 + 1 + 65*(`NUM_THREADS))
	) exec_unit_reg (
		.clk  (clk),
		.reset(reset),
		.stall(stall_exec),
		.flush(flush_exec),
		.in   ({exec_unit_req_temp_if.valid, exec_unit_req_temp_if.warp_num, exec_unit_req_temp_if.curr_PC, exec_unit_req_temp_if.PC_next, exec_unit_req_temp_if.rd, exec_unit_req_temp_if.wb, exec_unit_req_temp_if.a_reg_data, exec_unit_req_temp_if.b_reg_data, exec_unit_req_temp_if.alu_op, exec_unit_req_temp_if.rs1, exec_unit_req_temp_if.rs2, exec_unit_req_temp_if.rs2_src, exec_unit_req_temp_if.itype_immed, exec_unit_req_temp_if.upper_immed, exec_unit_req_temp_if.branch_type, exec_unit_req_temp_if.jalQual, exec_unit_req_temp_if.jal, exec_unit_req_temp_if.jal_offset, exec_unit_req_temp_if.ebreak, exec_unit_req_temp_if.wspawn, exec_unit_req_temp_if.is_csr, exec_unit_req_temp_if.csr_address, exec_unit_req_temp_if.csr_immed, exec_unit_req_temp_if.csr_mask}),
		.out  ({exec_unit_req_if.valid     , exec_unit_req_if.warp_num     , exec_unit_req_if.curr_PC     , exec_unit_req_if.PC_next     , exec_unit_req_if.rd     , exec_unit_req_if.wb     , exec_unit_req_if.a_reg_data     , exec_unit_req_if.b_reg_data     , exec_unit_req_if.alu_op     , exec_unit_req_if.rs1     , exec_unit_req_if.rs2     , exec_unit_req_if.rs2_src     , exec_unit_req_if.itype_immed     , exec_unit_req_if.upper_immed     , exec_unit_req_if.branch_type     , exec_unit_req_if.jalQual     , exec_unit_req_if.jal     , exec_unit_req_if.jal_offset     , exec_unit_req_if.ebreak     , exec_unit_req_if.wspawn     , exec_unit_req_if.is_csr     , exec_unit_req_if.csr_address     , exec_unit_req_if.csr_immed     , exec_unit_req_if.csr_mask     })
	);

	VX_generic_register #(
		.N(68 + `NW_BITS-1 + 1 + 33*(`NUM_THREADS))
	) gpu_inst_reg (
		.clk  (clk),
		.reset(reset),
		.stall(stall_rest),
		.flush(flush_rest),
		.in   ({gpu_inst_req_temp_if.valid, gpu_inst_req_temp_if.warp_num, gpu_inst_req_temp_if.is_wspawn, gpu_inst_req_temp_if.is_tmc, gpu_inst_req_temp_if.is_split, gpu_inst_req_temp_if.is_barrier, gpu_inst_req_temp_if.pc_next, gpu_inst_req_temp_if.a_reg_data, gpu_inst_req_temp_if.rd2}),
		.out  ({gpu_inst_req_if.valid     , gpu_inst_req_if.warp_num     , gpu_inst_req_if.is_wspawn     , gpu_inst_req_if.is_tmc     , gpu_inst_req_if.is_split     , gpu_inst_req_if.is_barrier     , gpu_inst_req_if.pc_next     , gpu_inst_req_if.a_reg_data     , gpu_inst_req_if.rd2     })
	);

	VX_generic_register #(
		.N(`NW_BITS-1  + 1 + `NUM_THREADS + 58)
	) csr_reg (
		.clk  (clk),
		.reset(reset),
		.stall(stall_gpr_csr),
		.flush(flush_rest),
		.in   ({csr_req_temp_if.valid, csr_req_temp_if.warp_num, csr_req_temp_if.rd, csr_req_temp_if.wb, csr_req_temp_if.alu_op, csr_req_temp_if.is_csr, csr_req_temp_if.csr_address, csr_req_temp_if.csr_immed, csr_req_temp_if.csr_mask}),
		.out  ({csr_req_if.valid     , csr_req_if.warp_num     , csr_req_if.rd     , csr_req_if.wb     , csr_req_if.alu_op     , csr_req_if.is_csr     , csr_req_if.csr_address     , csr_req_if.csr_immed     , csr_req_if.csr_mask     })
	);

`endif

endmodule : VX_gpr_stage