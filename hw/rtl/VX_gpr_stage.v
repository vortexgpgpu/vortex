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
	VX_frE_to_bckE_req_inter   vx_bckE_req,

	// WriteBack inputs
	VX_wb_inter                vx_writeback_inter,

	// Outputs
	VX_exec_unit_req_inter   vx_exec_unit_req,
	VX_lsu_req_inter         vx_lsu_req,
	VX_gpu_inst_req_inter    vx_gpu_inst_req,
	VX_csr_req_inter         vx_csr_req
);
/* verilator lint_off UNUSED */
	wire[31:0] curr_PC = vx_bckE_req.curr_PC;
	wire[2:0] branchType = vx_bckE_req.branch_type;
	wire is_store = (vx_bckE_req.mem_write != `NO_MEM_WRITE);
	wire is_load  = (vx_bckE_req.mem_read  != `NO_MEM_READ);
	wire jalQual = vx_bckE_req.jalQual;
/* verilator lint_on UNUSED */	

	VX_gpr_read_inter vx_gpr_read();
	assign vx_gpr_read.rs1      = vx_bckE_req.rs1;
	assign vx_gpr_read.rs2      = vx_bckE_req.rs2;
	assign vx_gpr_read.warp_num = vx_bckE_req.warp_num;

`ifndef ASIC
	VX_gpr_jal_inter vx_gpr_jal();
	assign vx_gpr_jal.is_jal  = vx_bckE_req.jalQual;
	assign vx_gpr_jal.curr_PC = vx_bckE_req.curr_PC;
`else 
	VX_gpr_jal_inter vx_gpr_jal();
	assign vx_gpr_jal.is_jal  = vx_exec_unit_req.jalQual;
	assign vx_gpr_jal.curr_PC = vx_exec_unit_req.curr_PC;
`endif

	VX_gpr_data_inter           vx_gpr_datf();

	VX_gpr_wrapper vx_grp_wrapper (
		.clk               (clk),
		.reset             (reset),
		.vx_writeback_inter(vx_writeback_inter),
		.vx_gpr_read       (vx_gpr_read),
		.vx_gpr_jal        (vx_gpr_jal),

		.out_a_reg_data (vx_gpr_datf.a_reg_data),
		.out_b_reg_data (vx_gpr_datf.b_reg_data)
	);

	// assign vx_bckE_req.is_csr   = is_csr;
	// assign vx_bckE_req_out.csr_mask = (vx_bckE_req.sr_immed == 1'b1) ?  {27'h0, vx_bckE_req.rs1} : vx_gpr_data.a_reg_data[0];

	// Outputs
	VX_exec_unit_req_inter   vx_exec_unit_req_temp();
	VX_lsu_req_inter         vx_lsu_req_temp();
	VX_gpu_inst_req_inter    vx_gpu_inst_req_temp();
	VX_csr_req_inter         vx_csr_req_temp();

	VX_inst_multiplex vx_inst_mult(
		.vx_bckE_req     (vx_bckE_req),
		.vx_gpr_data     (vx_gpr_datf),
		.vx_exec_unit_req(vx_exec_unit_req_temp),
		.vx_lsu_req      (vx_lsu_req_temp),
		.vx_gpu_inst_req (vx_gpu_inst_req_temp),
		.vx_csr_req      (vx_csr_req_temp)
	);
/* verilator lint_off UNUSED */
	wire is_lsu = (|vx_lsu_req_temp.valid);
/* verilator lint_on UNUSED */
	wire stall_rest = 0;
	wire flush_rest = schedule_delay;

	wire stall_lsu  = memory_delay;
	wire flush_lsu  = schedule_delay && !stall_lsu;

	wire stall_exec  = exec_delay;
	wire flush_exec  = schedule_delay && !stall_exec;

	wire stall_csr = stall_gpr_csr && vx_bckE_req.is_csr && (|vx_bckE_req.valid);

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

		assign real_store_data   = vx_lsu_req_temp.store_data;
		assign real_base_address = vx_lsu_req_temp.base_address;

		assign vx_lsu_req.store_data   = (delayed_lsu_last_cycle) ? temp_store_data   : real_store_data;
		assign vx_lsu_req.base_address = (delayed_lsu_last_cycle) ? temp_base_address : real_base_address;

		VX_generic_register #(
			.N(77 + `NW_BITS-1 + 1 + (`NUM_THREADS))
		) lsu_reg (
			.clk  (clk),
			.reset(reset),
			.stall(stall_lsu),
			.flush(flush_lsu),
			.in   ({vx_lsu_req_temp.valid, vx_lsu_req_temp.lsu_pc, vx_lsu_req_temp.warp_num, vx_lsu_req_temp.offset, vx_lsu_req_temp.mem_read, vx_lsu_req_temp.mem_write, vx_lsu_req_temp.rd, vx_lsu_req_temp.wb}),
			.out  ({vx_lsu_req.valid     , vx_lsu_req.lsu_pc     ,vx_lsu_req.warp_num     , vx_lsu_req.offset     , vx_lsu_req.mem_read     , vx_lsu_req.mem_write     , vx_lsu_req.rd     , vx_lsu_req.wb     })
		);

		VX_generic_register #(
			.N(224 + `NW_BITS-1 + 1 + (`NUM_THREADS))
		) exec_unit_reg (
			.clk  (clk),
			.reset(reset),
			.stall(stall_exec),
			.flush(flush_exec),
			.in   ({vx_exec_unit_req_temp.valid, vx_exec_unit_req_temp.warp_num, vx_exec_unit_req_temp.curr_PC, vx_exec_unit_req_temp.PC_next, vx_exec_unit_req_temp.rd, vx_exec_unit_req_temp.wb, vx_exec_unit_req_temp.alu_op, vx_exec_unit_req_temp.rs1, vx_exec_unit_req_temp.rs2, vx_exec_unit_req_temp.rs2_src, vx_exec_unit_req_temp.itype_immed, vx_exec_unit_req_temp.upper_immed, vx_exec_unit_req_temp.branch_type, vx_exec_unit_req_temp.jalQual, vx_exec_unit_req_temp.jal, vx_exec_unit_req_temp.jal_offset, vx_exec_unit_req_temp.ebreak, vx_exec_unit_req_temp.wspawn, vx_exec_unit_req_temp.is_csr, vx_exec_unit_req_temp.csr_address, vx_exec_unit_req_temp.csr_immed, vx_exec_unit_req_temp.csr_mask}),
			.out  ({vx_exec_unit_req.valid     , vx_exec_unit_req.warp_num     , vx_exec_unit_req.curr_PC     , vx_exec_unit_req.PC_next     , vx_exec_unit_req.rd     , vx_exec_unit_req.wb     , vx_exec_unit_req.alu_op     , vx_exec_unit_req.rs1     , vx_exec_unit_req.rs2     , vx_exec_unit_req.rs2_src     , vx_exec_unit_req.itype_immed     , vx_exec_unit_req.upper_immed     , vx_exec_unit_req.branch_type     , vx_exec_unit_req.jalQual     , vx_exec_unit_req.jal     , vx_exec_unit_req.jal_offset     , vx_exec_unit_req.ebreak     , vx_exec_unit_req.wspawn     , vx_exec_unit_req.is_csr     , vx_exec_unit_req.csr_address     , vx_exec_unit_req.csr_immed     , vx_exec_unit_req.csr_mask     })
		);

		assign vx_exec_unit_req.a_reg_data = real_base_address;
		assign vx_exec_unit_req.b_reg_data = real_store_data;

		VX_generic_register #(
			.N(36 + `NW_BITS-1 + 1 + (`NUM_THREADS))
		) gpu_inst_reg (
			.clk  (clk),
			.reset(reset),
			.stall(stall_rest),
			.flush(flush_rest),
			.in   ({vx_gpu_inst_req_temp.valid, vx_gpu_inst_req_temp.warp_num, vx_gpu_inst_req_temp.is_wspawn, vx_gpu_inst_req_temp.is_tmc, vx_gpu_inst_req_temp.is_split, vx_gpu_inst_req_temp.is_barrier, vx_gpu_inst_req_temp.pc_next}),
			.out  ({vx_gpu_inst_req.valid     , vx_gpu_inst_req.warp_num     , vx_gpu_inst_req.is_wspawn     , vx_gpu_inst_req.is_tmc     , vx_gpu_inst_req.is_split     , vx_gpu_inst_req.is_barrier     , vx_gpu_inst_req.pc_next     })
		);

		assign vx_gpu_inst_req.a_reg_data = real_base_address;
		assign vx_gpu_inst_req.rd2        = real_store_data;

		VX_generic_register #(
			.N(`NW_BITS-1  + 1 + `NUM_THREADS + 58)
		) csr_reg (
			.clk  (clk),
			.reset(reset),
			.stall(stall_gpr_csr),
			.flush(flush_rest),
			.in   ({vx_csr_req_temp.valid, vx_csr_req_temp.warp_num, vx_csr_req_temp.rd, vx_csr_req_temp.wb, vx_csr_req_temp.alu_op, vx_csr_req_temp.is_csr, vx_csr_req_temp.csr_address, vx_csr_req_temp.csr_immed, vx_csr_req_temp.csr_mask}),
			.out  ({vx_csr_req.valid     , vx_csr_req.warp_num     , vx_csr_req.rd     , vx_csr_req.wb     , vx_csr_req.alu_op     , vx_csr_req.is_csr     , vx_csr_req.csr_address     , vx_csr_req.csr_immed     , vx_csr_req.csr_mask     })
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
		.in   ({vx_lsu_req_temp.valid, vx_lsu_req_temp.lsu_pc, vx_lsu_req_temp.warp_num, vx_lsu_req_temp.store_data, vx_lsu_req_temp.base_address, vx_lsu_req_temp.offset, vx_lsu_req_temp.mem_read, vx_lsu_req_temp.mem_write, vx_lsu_req_temp.rd, vx_lsu_req_temp.wb}),
		.out  ({vx_lsu_req.valid     , vx_lsu_req.lsu_pc     , vx_lsu_req.warp_num     , vx_lsu_req.store_data     , vx_lsu_req.base_address     , vx_lsu_req.offset     , vx_lsu_req.mem_read     , vx_lsu_req.mem_write     , vx_lsu_req.rd     , vx_lsu_req.wb     })
	);

	VX_generic_register #(
		.N(224 + `NW_BITS-1 + 1 + 65*(`NUM_THREADS))
	) exec_unit_reg (
		.clk  (clk),
		.reset(reset),
		.stall(stall_exec),
		.flush(flush_exec),
		.in   ({vx_exec_unit_req_temp.valid, vx_exec_unit_req_temp.warp_num, vx_exec_unit_req_temp.curr_PC, vx_exec_unit_req_temp.PC_next, vx_exec_unit_req_temp.rd, vx_exec_unit_req_temp.wb, vx_exec_unit_req_temp.a_reg_data, vx_exec_unit_req_temp.b_reg_data, vx_exec_unit_req_temp.alu_op, vx_exec_unit_req_temp.rs1, vx_exec_unit_req_temp.rs2, vx_exec_unit_req_temp.rs2_src, vx_exec_unit_req_temp.itype_immed, vx_exec_unit_req_temp.upper_immed, vx_exec_unit_req_temp.branch_type, vx_exec_unit_req_temp.jalQual, vx_exec_unit_req_temp.jal, vx_exec_unit_req_temp.jal_offset, vx_exec_unit_req_temp.ebreak, vx_exec_unit_req_temp.wspawn, vx_exec_unit_req_temp.is_csr, vx_exec_unit_req_temp.csr_address, vx_exec_unit_req_temp.csr_immed, vx_exec_unit_req_temp.csr_mask}),
		.out  ({vx_exec_unit_req.valid     , vx_exec_unit_req.warp_num     , vx_exec_unit_req.curr_PC     , vx_exec_unit_req.PC_next     , vx_exec_unit_req.rd     , vx_exec_unit_req.wb     , vx_exec_unit_req.a_reg_data     , vx_exec_unit_req.b_reg_data     , vx_exec_unit_req.alu_op     , vx_exec_unit_req.rs1     , vx_exec_unit_req.rs2     , vx_exec_unit_req.rs2_src     , vx_exec_unit_req.itype_immed     , vx_exec_unit_req.upper_immed     , vx_exec_unit_req.branch_type     , vx_exec_unit_req.jalQual     , vx_exec_unit_req.jal     , vx_exec_unit_req.jal_offset     , vx_exec_unit_req.ebreak     , vx_exec_unit_req.wspawn     , vx_exec_unit_req.is_csr     , vx_exec_unit_req.csr_address     , vx_exec_unit_req.csr_immed     , vx_exec_unit_req.csr_mask     })
	);

	VX_generic_register #(
		.N(68 + `NW_BITS-1 + 1 + 33*(`NUM_THREADS))
	) gpu_inst_reg (
		.clk  (clk),
		.reset(reset),
		.stall(stall_rest),
		.flush(flush_rest),
		.in   ({vx_gpu_inst_req_temp.valid, vx_gpu_inst_req_temp.warp_num, vx_gpu_inst_req_temp.is_wspawn, vx_gpu_inst_req_temp.is_tmc, vx_gpu_inst_req_temp.is_split, vx_gpu_inst_req_temp.is_barrier, vx_gpu_inst_req_temp.pc_next, vx_gpu_inst_req_temp.a_reg_data, vx_gpu_inst_req_temp.rd2}),
		.out  ({vx_gpu_inst_req.valid     , vx_gpu_inst_req.warp_num     , vx_gpu_inst_req.is_wspawn     , vx_gpu_inst_req.is_tmc     , vx_gpu_inst_req.is_split     , vx_gpu_inst_req.is_barrier     , vx_gpu_inst_req.pc_next     , vx_gpu_inst_req.a_reg_data     , vx_gpu_inst_req.rd2     })
	);

	VX_generic_register #(
		.N(`NW_BITS-1  + 1 + `NUM_THREADS + 58)
	) csr_reg (
		.clk  (clk),
		.reset(reset),
		.stall(stall_gpr_csr),
		.flush(flush_rest),
		.in   ({vx_csr_req_temp.valid, vx_csr_req_temp.warp_num, vx_csr_req_temp.rd, vx_csr_req_temp.wb, vx_csr_req_temp.alu_op, vx_csr_req_temp.is_csr, vx_csr_req_temp.csr_address, vx_csr_req_temp.csr_immed, vx_csr_req_temp.csr_mask}),
		.out  ({vx_csr_req.valid     , vx_csr_req.warp_num     , vx_csr_req.rd     , vx_csr_req.wb     , vx_csr_req.alu_op     , vx_csr_req.is_csr     , vx_csr_req.csr_address     , vx_csr_req.csr_immed     , vx_csr_req.csr_mask     })
	);

`endif

endmodule : VX_gpr_stage