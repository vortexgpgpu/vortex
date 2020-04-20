`include "VX_define.v"

module VX_inst_multiplex (
	// Inputs
	VX_frE_to_bckE_req_if bckE_req_if,
	VX_gpr_data_if        gpr_data_if,

	// Outputs
	VX_exec_unit_req_if   exec_unit_req_if,
	VX_lsu_req_if         lsu_req_if,
	VX_gpu_inst_req_if    gpu_inst_req_if,
	VX_csr_req_if         csr_req_if
);

	wire[`NUM_THREADS-1:0] is_mem_mask;
	wire[`NUM_THREADS-1:0] is_gpu_mask;
	wire[`NUM_THREADS-1:0] is_csr_mask;

	wire is_mem = (bckE_req_if.mem_write != `NO_MEM_WRITE) || (bckE_req_if.mem_read != `NO_MEM_READ);
	wire is_gpu = (bckE_req_if.is_wspawn || bckE_req_if.is_tmc || bckE_req_if.is_barrier || bckE_req_if.is_split);
	wire is_csr = bckE_req_if.is_csr;
	// wire is_gpu = 0;

	genvar currT;
	generate
	for (currT = 0; currT < `NUM_THREADS; currT = currT + 1) begin : mask_init
		assign is_mem_mask[currT] = is_mem;
		assign is_gpu_mask[currT] = is_gpu;
		assign is_csr_mask[currT] = is_csr;
	end
	endgenerate

	// LSU Unit
	assign lsu_req_if.valid        = bckE_req_if.valid & is_mem_mask;
	assign lsu_req_if.warp_num     = bckE_req_if.warp_num;
	assign lsu_req_if.base_address = gpr_data_if.a_reg_data;
	assign lsu_req_if.store_data   = gpr_data_if.b_reg_data;

	assign lsu_req_if.offset       = bckE_req_if.itype_immed;

	assign lsu_req_if.mem_read     = bckE_req_if.mem_read;
	assign lsu_req_if.mem_write    = bckE_req_if.mem_write;
	assign lsu_req_if.rd           = bckE_req_if.rd;
	assign lsu_req_if.wb           = bckE_req_if.wb;
	assign lsu_req_if.lsu_pc       = bckE_req_if.curr_PC;


	// Execute Unit
	assign exec_unit_req_if.valid       = bckE_req_if.valid & (~is_mem_mask & ~is_gpu_mask & ~is_csr_mask);
	assign exec_unit_req_if.warp_num    = bckE_req_if.warp_num;
	assign exec_unit_req_if.curr_PC     = bckE_req_if.curr_PC;
	assign exec_unit_req_if.PC_next     = bckE_req_if.PC_next;
	assign exec_unit_req_if.rd          = bckE_req_if.rd;
	assign exec_unit_req_if.wb          = bckE_req_if.wb;
	assign exec_unit_req_if.a_reg_data  = gpr_data_if.a_reg_data;
	assign exec_unit_req_if.b_reg_data  = gpr_data_if.b_reg_data;
	assign exec_unit_req_if.alu_op      = bckE_req_if.alu_op;
	assign exec_unit_req_if.rs1         = bckE_req_if.rs1;
	assign exec_unit_req_if.rs2         = bckE_req_if.rs2;
	assign exec_unit_req_if.rs2_src     = bckE_req_if.rs2_src;
	assign exec_unit_req_if.itype_immed = bckE_req_if.itype_immed;
	assign exec_unit_req_if.upper_immed = bckE_req_if.upper_immed;
	assign exec_unit_req_if.branch_type = bckE_req_if.branch_type;
	assign exec_unit_req_if.jalQual     = bckE_req_if.jalQual;
	assign exec_unit_req_if.jal         = bckE_req_if.jal;
	assign exec_unit_req_if.jal_offset  = bckE_req_if.jal_offset;
	assign exec_unit_req_if.ebreak      = bckE_req_if.ebreak;


	// GPR Req
	assign gpu_inst_req_if.valid       = bckE_req_if.valid & is_gpu_mask;
	assign gpu_inst_req_if.warp_num    = bckE_req_if.warp_num;
	assign gpu_inst_req_if.is_wspawn   = bckE_req_if.is_wspawn;
	assign gpu_inst_req_if.is_tmc      = bckE_req_if.is_tmc;
	assign gpu_inst_req_if.is_split    = bckE_req_if.is_split;
	assign gpu_inst_req_if.is_barrier  = bckE_req_if.is_barrier;
	assign gpu_inst_req_if.a_reg_data  = gpr_data_if.a_reg_data;
	assign gpu_inst_req_if.rd2         = gpr_data_if.b_reg_data[0];
	assign gpu_inst_req_if.pc_next     = bckE_req_if.PC_next;


	// CSR Req
	assign csr_req_if.valid           = bckE_req_if.valid & is_csr_mask;
	assign csr_req_if.warp_num        = bckE_req_if.warp_num;
	assign csr_req_if.rd              = bckE_req_if.rd;
	assign csr_req_if.wb              = bckE_req_if.wb;
	assign csr_req_if.alu_op          = bckE_req_if.alu_op;
	assign csr_req_if.is_csr          = bckE_req_if.is_csr;
	assign csr_req_if.csr_address     = bckE_req_if.csr_address;
	assign csr_req_if.csr_immed       = bckE_req_if.csr_immed;
	assign csr_req_if.csr_mask        = bckE_req_if.csr_mask;

endmodule




