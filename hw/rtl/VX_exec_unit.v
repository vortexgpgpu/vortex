`include "VX_define.vh"

module VX_exec_unit (
	input wire            clk,
	input wire            reset,
		// Request
	VX_exec_unit_req_if   exec_unit_req_if,

	// Output
		// Writeback
	VX_inst_exec_wb_if    inst_exec_wb_if,
		// JAL Response
	VX_jal_rsp_if    jal_rsp_if,
		// Branch Response
	VX_branch_rsp_if branch_rsp_if,

	input wire 			  no_slot_exec_i,
	output wire 		  delay_o
);

	wire[`NUM_THREADS-1:0][31:0] in_a_reg_data;
	wire[`NUM_THREADS-1:0][31:0] in_b_reg_data;
	wire[4:0]            in_alu_op;
	wire                 in_rs2_src;
	wire[31:0]           in_itype_immed;
`DEBUG_BEGIN
	wire[2:0]            in_branch_type;
`DEBUG_END
	wire[19:0]           in_upper_immed;
	wire                 in_jal;
	wire[31:0]           in_jal_offset;
	wire[31:0]           in_curr_PC;

	assign in_a_reg_data  = exec_unit_req_if.a_reg_data;
	assign in_b_reg_data  = exec_unit_req_if.b_reg_data;
	assign in_alu_op      = exec_unit_req_if.alu_op;
	assign in_rs2_src     = exec_unit_req_if.rs2_src;
	assign in_itype_immed = exec_unit_req_if.itype_immed;
	assign in_branch_type = exec_unit_req_if.branch_type;
	assign in_upper_immed = exec_unit_req_if.upper_immed;
	assign in_jal         = exec_unit_req_if.jal;
	assign in_jal_offset  = exec_unit_req_if.jal_offset;
	assign in_curr_PC     = exec_unit_req_if.curr_PC;

	wire[`NUM_THREADS-1:0][31:0]  alu_result;
	wire[`NUM_THREADS-1:0]  alu_stall;
	genvar index_out_reg;
	generate
		for (index_out_reg = 0; index_out_reg < `NUM_THREADS; index_out_reg = index_out_reg + 1) begin : alu_defs
			VX_alu alu(
				.clk			(clk),
				.reset			(reset),
				.a_i          	(in_a_reg_data[index_out_reg]),
				.b_i          	(in_b_reg_data[index_out_reg]),
				.rs2_src_i   	(in_rs2_src),
				.itype_immed_i	(in_itype_immed),
				.upper_immed_i	(in_upper_immed),
				.alu_op_i     	(in_alu_op),
				.curr_PC_i    	(in_curr_PC),
				.alu_result_o	(alu_result[index_out_reg]),
				.alu_stall_o	(alu_stall[index_out_reg])
			);
		end
	endgenerate

	wire internal_stall;
	assign internal_stall = |alu_stall;

	assign delay_o = no_slot_exec_i || internal_stall;

`DEBUG_BEGIN
	wire [$clog2(`NUM_THREADS)-1:0] jal_branch_use_index;
	wire  jal_branch_found_valid;
`DEBUG_END

	VX_generic_priority_encoder #(
		.N(`NUM_THREADS)
	) choose_alu_result (
		.valids(exec_unit_req_if.valid),
		.index (jal_branch_use_index),
		.found (jal_branch_found_valid)
	);

	wire[31:0] branch_use_alu_result = alu_result[jal_branch_use_index];

	reg temp_branch_dir;
	always @(*)
	begin
		case (exec_unit_req_if.branch_type)
			`BEQ:       temp_branch_dir = (branch_use_alu_result     == 0) ? `TAKEN     : `NOT_TAKEN;
			`BNE:       temp_branch_dir = (branch_use_alu_result     == 0) ? `NOT_TAKEN : `TAKEN;
			`BLT:       temp_branch_dir = (branch_use_alu_result[31] == 0) ? `NOT_TAKEN : `TAKEN;
			`BGT:       temp_branch_dir = (branch_use_alu_result[31] == 0) ? `TAKEN     : `NOT_TAKEN;
			`BLTU:      temp_branch_dir = (branch_use_alu_result[31] == 0) ? `NOT_TAKEN : `TAKEN; 
			`BGTU:      temp_branch_dir = (branch_use_alu_result[31] == 0) ? `TAKEN     : `NOT_TAKEN;
			`NO_BRANCH: temp_branch_dir = `NOT_TAKEN;
			default:    temp_branch_dir = `NOT_TAKEN;
		endcase // in_branch_type
	end


	wire[`NUM_THREADS-1:0][31:0] duplicate_PC_data;
	genvar i;
	generate
		for (i = 0; i < `NUM_THREADS; i=i+1) begin : pc_data_setup
			assign duplicate_PC_data[i] = exec_unit_req_if.PC_next;
		end
	endgenerate


	// VX_inst_exec_wb_if    inst_exec_wb_temp_if();
		// JAL Response
	VX_jal_rsp_if    jal_rsp_temp_if();
		// Branch Response
	VX_branch_rsp_if branch_rsp_temp_if();

	// Actual Writeback
	assign inst_exec_wb_if.rd          = exec_unit_req_if.rd;
	assign inst_exec_wb_if.wb          = exec_unit_req_if.wb;
	assign inst_exec_wb_if.wb_valid    = exec_unit_req_if.valid & {`NUM_THREADS{!internal_stall}};
	assign inst_exec_wb_if.wb_warp_num = exec_unit_req_if.warp_num;
	assign inst_exec_wb_if.alu_result  = exec_unit_req_if.jal ? duplicate_PC_data : alu_result;

	assign inst_exec_wb_if.exec_wb_pc  = in_curr_PC;
	// Jal rsp
	assign jal_rsp_temp_if.jal           = in_jal;
	assign jal_rsp_temp_if.jal_dest      = $signed(in_a_reg_data[jal_branch_use_index]) + $signed(in_jal_offset);
	assign jal_rsp_temp_if.jal_warp_num  = exec_unit_req_if.warp_num;

	// Branch rsp
	assign branch_rsp_temp_if.valid_branch    = (exec_unit_req_if.branch_type != `NO_BRANCH) && (|exec_unit_req_if.valid);
	assign branch_rsp_temp_if.branch_dir      = temp_branch_dir;
	assign branch_rsp_temp_if.branch_warp_num = exec_unit_req_if.warp_num;
	assign branch_rsp_temp_if.branch_dest     = $signed(exec_unit_req_if.curr_PC) + ($signed(exec_unit_req_if.itype_immed) << 1); // itype_immed = branch_offset


	wire zero = 0;

	// VX_generic_register #(.N(174)) exec_reg(
	// 	.clk  (clk),
	// 	.reset(reset),
	// 	.stall(zero),
	// 	.flush(zero),
	// 	.in   ({inst_exec_wb_temp_if.rd, inst_exec_wb_temp_if.wb, inst_exec_wb_temp_if.wb_valid, inst_exec_wb_temp_if.wb_warp_num, inst_exec_wb_temp_if.alu_result, inst_exec_wb_temp_if.exec_wb_pc}),
	// 	.out  ({inst_exec_wb_if.rd     , inst_exec_wb_if.wb     , inst_exec_wb_if.wb_valid     , inst_exec_wb_if.wb_warp_num     , inst_exec_wb_if.alu_result     , inst_exec_wb_if.exec_wb_pc     })
	// 	);

	VX_generic_register #(
		.N(33 + `NW_BITS-1 + 1)
	) jal_reg (
		.clk  (clk),
		.reset(reset),
		.stall(zero),
		.flush(zero),
		.in   ({jal_rsp_temp_if.jal, jal_rsp_temp_if.jal_dest, jal_rsp_temp_if.jal_warp_num}),
		.out  ({jal_rsp_if.jal     , jal_rsp_if.jal_dest     , jal_rsp_if.jal_warp_num})
	);

	VX_generic_register #(
		.N(34 + `NW_BITS-1 + 1)
	) branch_reg (
		.clk  (clk),
		.reset(reset),
		.stall(zero),
		.flush(zero),
		.in   ({branch_rsp_temp_if.valid_branch, branch_rsp_temp_if.branch_dir, branch_rsp_temp_if.branch_warp_num, branch_rsp_temp_if.branch_dest}),
		.out  ({branch_rsp_if.valid_branch     , branch_rsp_if.branch_dir     , branch_rsp_if.branch_warp_num     , branch_rsp_if.branch_dest     })
	);

	// always @(*) begin
	// 	case(in_alu_op)
	// 		`CSR_ALU_RW: out_csr_result = in_csr_mask;
	// 		`CSR_ALU_RS: out_csr_result = in_csr_data | in_csr_mask;
	// 		`CSR_ALU_RC: out_csr_result = in_csr_data & (32'hFFFFFFFF - in_csr_mask);
	// 		default:     out_csr_result = 32'hdeadbeef;
	// 	endcase
	
	// end

	// assign out_is_csr        = exec_unit_req_if.is_csr;
	// assign out_csr_address   = exec_unit_req_if.csr_address;

endmodule : VX_exec_unit