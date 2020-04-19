`include "VX_define.vh"

module VX_execute_unit (
	input wire               clk,
	input wire               reset,
		// Request
	VX_exec_unit_req_inter   vx_exec_unit_req,

	// Output
		// Writeback
	VX_inst_exec_wb_inter    vx_inst_exec_wb,
		// JAL Response
	VX_jal_response_inter    vx_jal_rsp,
		// Branch Response
	VX_branch_response_inter vx_branch_rsp,

	input wire no_slot_exec,
	output wire out_delay
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

	assign in_a_reg_data  = vx_exec_unit_req.a_reg_data;
	assign in_b_reg_data  = vx_exec_unit_req.b_reg_data;
	assign in_alu_op      = vx_exec_unit_req.alu_op;
	assign in_rs2_src     = vx_exec_unit_req.rs2_src;
	assign in_itype_immed = vx_exec_unit_req.itype_immed;
	assign in_branch_type = vx_exec_unit_req.branch_type;
	assign in_upper_immed = vx_exec_unit_req.upper_immed;
	assign in_jal         = vx_exec_unit_req.jal;
	assign in_jal_offset  = vx_exec_unit_req.jal_offset;
	assign in_curr_PC     = vx_exec_unit_req.curr_PC;

	wire[`NUM_THREADS-1:0][31:0]  alu_result;
	wire[`NUM_THREADS-1:0]  alu_stall;
	genvar index_out_reg;
	generate
		for (index_out_reg = 0; index_out_reg < `NUM_THREADS; index_out_reg = index_out_reg + 1) begin : alu_defs
			VX_alu vx_alu(
				.clk(clk),
				.reset(reset),
				// .in_reg_data   (in_reg_data[1:0]),
				.in_1          (in_a_reg_data[index_out_reg]),
				.in_2          (in_b_reg_data[index_out_reg]),
				.in_rs2_src    (in_rs2_src),
				.in_itype_immed(in_itype_immed),
				.in_upper_immed(in_upper_immed),
				.in_alu_op     (in_alu_op),
				.in_curr_PC    (in_curr_PC),
				.out_alu_result(alu_result[index_out_reg]),
				.out_alu_stall(alu_stall[index_out_reg])
			);
		end
	endgenerate

	wire internal_stall;
	assign internal_stall = |alu_stall;

	assign out_delay = no_slot_exec || internal_stall;

`DEBUG_BEGIN
	wire [$clog2(`NUM_THREADS)-1:0] jal_branch_use_index;
	wire  jal_branch_found_valid;
`DEBUG_END

	VX_generic_priority_encoder #(
		.N(`NUM_THREADS)
	) choose_alu_result (
		.valids(vx_exec_unit_req.valid),
		.index (jal_branch_use_index),
		.found (jal_branch_found_valid)
		);

	wire[31:0] branch_use_alu_result = alu_result[jal_branch_use_index];

	reg temp_branch_dir;
	always @(*)
	begin
		case (vx_exec_unit_req.branch_type)
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
			assign duplicate_PC_data[i] = vx_exec_unit_req.PC_next;
		end
	endgenerate


	// VX_inst_exec_wb_inter    vx_inst_exec_wb_temp();
		// JAL Response
	VX_jal_response_inter    vx_jal_rsp_temp();
		// Branch Response
	VX_branch_response_inter vx_branch_rsp_temp();

	// Actual Writeback
	assign vx_inst_exec_wb.rd          = vx_exec_unit_req.rd;
	assign vx_inst_exec_wb.wb          = vx_exec_unit_req.wb;
	assign vx_inst_exec_wb.wb_valid    = vx_exec_unit_req.valid & {`NUM_THREADS{!internal_stall}};
	assign vx_inst_exec_wb.wb_warp_num = vx_exec_unit_req.warp_num;
	assign vx_inst_exec_wb.alu_result  = vx_exec_unit_req.jal ? duplicate_PC_data : alu_result;

	assign vx_inst_exec_wb.exec_wb_pc  = in_curr_PC;
	// Jal rsp
	assign vx_jal_rsp_temp.jal           = in_jal;
	assign vx_jal_rsp_temp.jal_dest      = $signed(in_a_reg_data[jal_branch_use_index]) + $signed(in_jal_offset);
	assign vx_jal_rsp_temp.jal_warp_num  = vx_exec_unit_req.warp_num;

	// Branch rsp
	assign vx_branch_rsp_temp.valid_branch    = (vx_exec_unit_req.branch_type != `NO_BRANCH) && (|vx_exec_unit_req.valid);
	assign vx_branch_rsp_temp.branch_dir      = temp_branch_dir;
	assign vx_branch_rsp_temp.branch_warp_num = vx_exec_unit_req.warp_num;
	assign vx_branch_rsp_temp.branch_dest     = $signed(vx_exec_unit_req.curr_PC) + ($signed(vx_exec_unit_req.itype_immed) << 1); // itype_immed = branch_offset


	wire zero = 0;

	// VX_generic_register #(.N(174)) exec_reg(
	// 	.clk  (clk),
	// 	.reset(reset),
	// 	.stall(zero),
	// 	.flush(zero),
	// 	.in   ({vx_inst_exec_wb_temp.rd, vx_inst_exec_wb_temp.wb, vx_inst_exec_wb_temp.wb_valid, vx_inst_exec_wb_temp.wb_warp_num, vx_inst_exec_wb_temp.alu_result, vx_inst_exec_wb_temp.exec_wb_pc}),
	// 	.out  ({vx_inst_exec_wb.rd     , vx_inst_exec_wb.wb     , vx_inst_exec_wb.wb_valid     , vx_inst_exec_wb.wb_warp_num     , vx_inst_exec_wb.alu_result     , vx_inst_exec_wb.exec_wb_pc     })
	// 	);

	VX_generic_register #(
		.N(33 + `NW_BITS-1 + 1)
	) jal_reg (
		.clk  (clk),
		.reset(reset),
		.stall(zero),
		.flush(zero),
		.in   ({vx_jal_rsp_temp.jal, vx_jal_rsp_temp.jal_dest, vx_jal_rsp_temp.jal_warp_num}),
		.out  ({vx_jal_rsp.jal     , vx_jal_rsp.jal_dest     , vx_jal_rsp.jal_warp_num})
	);

	VX_generic_register #(
		.N(34 + `NW_BITS-1 + 1)
	) branch_reg (
		.clk  (clk),
		.reset(reset),
		.stall(zero),
		.flush(zero),
		.in   ({vx_branch_rsp_temp.valid_branch, vx_branch_rsp_temp.branch_dir, vx_branch_rsp_temp.branch_warp_num, vx_branch_rsp_temp.branch_dest}),
		.out  ({vx_branch_rsp.valid_branch     , vx_branch_rsp.branch_dir     , vx_branch_rsp.branch_warp_num     , vx_branch_rsp.branch_dest     })
	);

	// always @(*) begin
	// 	case(in_alu_op)
	// 		`CSR_ALU_RW: out_csr_result = in_csr_mask;
	// 		`CSR_ALU_RS: out_csr_result = in_csr_data | in_csr_mask;
	// 		`CSR_ALU_RC: out_csr_result = in_csr_data & (32'hFFFFFFFF - in_csr_mask);
	// 		default:     out_csr_result = 32'hdeadbeef;
	// 	endcase
	
	// end

	// assign out_is_csr        = vx_exec_unit_req.is_csr;
	// assign out_csr_address   = vx_exec_unit_req.csr_address;

endmodule : VX_execute_unit