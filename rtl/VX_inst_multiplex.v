`include "VX_define.v"

module VX_inst_multiplex (
	// Inputs
	VX_frE_to_bckE_req_inter VX_bckE_req,
	VX_gpr_data_inter        VX_gpr_data,

	// Outputs
	VX_exec_unit_req_inter   VX_exec_unit_req,
	VX_lsu_req_inter         VX_lsu_req,
	VX_gpu_inst_req_inter    VX_gpu_inst_req,
	VX_csr_req_inter         VX_csr_req
);

	wire[`NT_M1:0] is_mem_mask;
	wire[`NT_M1:0] is_gpu_mask;
	wire[`NT_M1:0] is_csr_mask;

	wire is_mem = (VX_bckE_req.mem_write != `NO_MEM_WRITE) || (VX_bckE_req.mem_read != `NO_MEM_READ);
	wire is_gpu = (VX_bckE_req.is_wspawn || VX_bckE_req.is_tmc || VX_bckE_req.is_barrier || VX_bckE_req.is_split);
	wire is_csr = VX_bckE_req.is_csr;
	// wire is_gpu = 0;

	genvar currT;
	for (currT = 0; currT < `NT; currT = currT + 1) begin
		assign is_mem_mask[currT] = is_mem;
		assign is_gpu_mask[currT] = is_gpu;
		assign is_csr_mask[currT] = is_csr;
	end

	// LSU Unit
	assign VX_lsu_req.valid        = VX_bckE_req.valid & is_mem_mask;
	assign VX_lsu_req.warp_num     = VX_bckE_req.warp_num;
	assign VX_lsu_req.base_address = VX_gpr_data.a_reg_data;
	assign VX_lsu_req.store_data   = VX_gpr_data.b_reg_data;

	assign VX_lsu_req.offset       = VX_bckE_req.itype_immed;

	assign VX_lsu_req.mem_read     = VX_bckE_req.mem_read;
	assign VX_lsu_req.mem_write    = VX_bckE_req.mem_write;
	assign VX_lsu_req.rd           = VX_bckE_req.rd;
	assign VX_lsu_req.wb           = VX_bckE_req.wb;


	// Execute Unit
	assign VX_exec_unit_req.valid       = VX_bckE_req.valid & (~is_mem_mask & ~is_gpu_mask & ~is_csr_mask);
	assign VX_exec_unit_req.warp_num    = VX_bckE_req.warp_num;
	assign VX_exec_unit_req.curr_PC     = VX_bckE_req.curr_PC;
	assign VX_exec_unit_req.PC_next     = VX_bckE_req.PC_next;
	assign VX_exec_unit_req.rd          = VX_bckE_req.rd;
	assign VX_exec_unit_req.wb          = VX_bckE_req.wb;
	assign VX_exec_unit_req.a_reg_data  = VX_gpr_data.a_reg_data;
	assign VX_exec_unit_req.b_reg_data  = VX_gpr_data.b_reg_data;
	assign VX_exec_unit_req.alu_op      = VX_bckE_req.alu_op;
	assign VX_exec_unit_req.rs1         = VX_bckE_req.rs1;
	assign VX_exec_unit_req.rs2         = VX_bckE_req.rs2;
	assign VX_exec_unit_req.rs2_src     = VX_bckE_req.rs2_src;
	assign VX_exec_unit_req.itype_immed = VX_bckE_req.itype_immed;
	assign VX_exec_unit_req.upper_immed = VX_bckE_req.upper_immed;
	assign VX_exec_unit_req.branch_type = VX_bckE_req.branch_type;
	assign VX_exec_unit_req.jalQual     = VX_bckE_req.jalQual;
	assign VX_exec_unit_req.jal         = VX_bckE_req.jal;
	assign VX_exec_unit_req.jal_offset  = VX_bckE_req.jal_offset;
	assign VX_exec_unit_req.ebreak      = VX_bckE_req.ebreak;


	// GPR Req
	assign VX_gpu_inst_req.valid       = VX_bckE_req.valid & is_gpu_mask;
	assign VX_gpu_inst_req.warp_num    = VX_bckE_req.warp_num;
	assign VX_gpu_inst_req.is_wspawn   = VX_bckE_req.is_wspawn;
	assign VX_gpu_inst_req.is_tmc      = VX_bckE_req.is_tmc;
	assign VX_gpu_inst_req.is_split    = VX_bckE_req.is_split;
	assign VX_gpu_inst_req.is_barrier  = VX_bckE_req.is_barrier;
	assign VX_gpu_inst_req.a_reg_data  = VX_gpr_data.a_reg_data;
	assign VX_gpu_inst_req.rd2         = VX_gpr_data.b_reg_data[0];
	assign VX_gpu_inst_req.pc_next     = VX_bckE_req.PC_next;


	// CSR Req
	assign VX_csr_req.valid           = VX_bckE_req.valid & is_csr_mask;
	assign VX_csr_req.warp_num        = VX_bckE_req.warp_num;
	assign VX_csr_req.rd              = VX_bckE_req.rd;
	assign VX_csr_req.wb              = VX_bckE_req.wb;
	assign VX_csr_req.is_csr          = VX_bckE_req.is_csr;
	assign VX_csr_req.csr_address     = VX_bckE_req.csr_address;
	assign VX_csr_req.csr_immed       = VX_bckE_req.csr_immed;
	assign VX_csr_req.csr_mask        = VX_bckE_req.csr_mask;

endmodule




