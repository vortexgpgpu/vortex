`include "VX_define.vh"

module VX_inst_multiplex (
	// Inputs
	VX_frE_to_bckE_req_inter vx_bckE_req,
	VX_gpr_data_inter        vx_gpr_data,

	// Outputs
	VX_exec_unit_req_inter   vx_exec_unit_req,
	VX_lsu_req_inter         vx_lsu_req,
	VX_gpu_inst_req_inter    vx_gpu_inst_req,
	VX_csr_req_inter         vx_csr_req
);

	wire[`NUM_THREADS-1:0] is_mem_mask;
	wire[`NUM_THREADS-1:0] is_gpu_mask;
	wire[`NUM_THREADS-1:0] is_csr_mask;

	wire is_mem = (vx_bckE_req.mem_write != `NO_MEM_WRITE) || (vx_bckE_req.mem_read != `NO_MEM_READ);
	wire is_gpu = (vx_bckE_req.is_wspawn || vx_bckE_req.is_tmc || vx_bckE_req.is_barrier || vx_bckE_req.is_split);
	wire is_csr = vx_bckE_req.is_csr;
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
	assign vx_lsu_req.valid        = vx_bckE_req.valid & is_mem_mask;
	assign vx_lsu_req.warp_num     = vx_bckE_req.warp_num;
	assign vx_lsu_req.base_address = vx_gpr_data.a_reg_data;
	assign vx_lsu_req.store_data   = vx_gpr_data.b_reg_data;

	assign vx_lsu_req.offset       = vx_bckE_req.itype_immed;

	assign vx_lsu_req.mem_read     = vx_bckE_req.mem_read;
	assign vx_lsu_req.mem_write    = vx_bckE_req.mem_write;
	assign vx_lsu_req.rd           = vx_bckE_req.rd;
	assign vx_lsu_req.wb           = vx_bckE_req.wb;
	assign vx_lsu_req.lsu_pc       = vx_bckE_req.curr_PC;


	// Execute Unit
	assign vx_exec_unit_req.valid       = vx_bckE_req.valid & (~is_mem_mask & ~is_gpu_mask & ~is_csr_mask);
	assign vx_exec_unit_req.warp_num    = vx_bckE_req.warp_num;
	assign vx_exec_unit_req.curr_PC     = vx_bckE_req.curr_PC;
	assign vx_exec_unit_req.PC_next     = vx_bckE_req.PC_next;
	assign vx_exec_unit_req.rd          = vx_bckE_req.rd;
	assign vx_exec_unit_req.wb          = vx_bckE_req.wb;
	assign vx_exec_unit_req.a_reg_data  = vx_gpr_data.a_reg_data;
	assign vx_exec_unit_req.b_reg_data  = vx_gpr_data.b_reg_data;
	assign vx_exec_unit_req.alu_op      = vx_bckE_req.alu_op;
	assign vx_exec_unit_req.rs1         = vx_bckE_req.rs1;
	assign vx_exec_unit_req.rs2         = vx_bckE_req.rs2;
	assign vx_exec_unit_req.rs2_src     = vx_bckE_req.rs2_src;
	assign vx_exec_unit_req.itype_immed = vx_bckE_req.itype_immed;
	assign vx_exec_unit_req.upper_immed = vx_bckE_req.upper_immed;
	assign vx_exec_unit_req.branch_type = vx_bckE_req.branch_type;
	assign vx_exec_unit_req.jalQual     = vx_bckE_req.jalQual;
	assign vx_exec_unit_req.jal         = vx_bckE_req.jal;
	assign vx_exec_unit_req.jal_offset  = vx_bckE_req.jal_offset;
	assign vx_exec_unit_req.ebreak      = vx_bckE_req.ebreak;


	// GPR Req
	assign vx_gpu_inst_req.valid       = vx_bckE_req.valid & is_gpu_mask;
	assign vx_gpu_inst_req.warp_num    = vx_bckE_req.warp_num;
	assign vx_gpu_inst_req.is_wspawn   = vx_bckE_req.is_wspawn;
	assign vx_gpu_inst_req.is_tmc      = vx_bckE_req.is_tmc;
	assign vx_gpu_inst_req.is_split    = vx_bckE_req.is_split;
	assign vx_gpu_inst_req.is_barrier  = vx_bckE_req.is_barrier;
	assign vx_gpu_inst_req.a_reg_data  = vx_gpr_data.a_reg_data;
	assign vx_gpu_inst_req.rd2         = vx_gpr_data.b_reg_data[0];
	assign vx_gpu_inst_req.pc_next     = vx_bckE_req.PC_next;


	// CSR Req
	assign vx_csr_req.valid           = vx_bckE_req.valid & is_csr_mask;
	assign vx_csr_req.warp_num        = vx_bckE_req.warp_num;
	assign vx_csr_req.rd              = vx_bckE_req.rd;
	assign vx_csr_req.wb              = vx_bckE_req.wb;
	assign vx_csr_req.alu_op          = vx_bckE_req.alu_op;
	assign vx_csr_req.is_csr          = vx_bckE_req.is_csr;
	assign vx_csr_req.csr_address     = vx_bckE_req.csr_address;
	assign vx_csr_req.csr_immed       = vx_bckE_req.csr_immed;
	assign vx_csr_req.csr_mask        = vx_bckE_req.csr_mask;

endmodule




