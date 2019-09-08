
`include "VX_define.v"


module VX_writeback (
		VX_mw_wb_inter      VX_mw_wb,
		VX_forward_wb_inter VX_fwd_wb,
		VX_wb_inter         VX_writeback_inter
	);

		

		wire[`NT_M1:0][31:0] in_alu_result = VX_mw_wb.alu_result;
		wire[`NT_M1:0][31:0] in_mem_result = VX_mw_wb.mem_result;
		wire[4:0]            in_rd         = VX_mw_wb.rd;
		wire[1:0]            in_wb         = VX_mw_wb.wb;
		wire[31:0]           in_PC_next    = VX_mw_wb.PC_next;
		wire[`NT_M1:0]       in_valid      = VX_mw_wb.valid;
		wire [`NW_M1:0]      in_warp_num   = VX_mw_wb.warp_num;

		wire is_jal;
		wire uses_alu;

		wire[`NT_M1:0][31:0] out_pc_data;


		genvar i;
		generate
			for (i = 0; i < `NT; i=i+1)
			begin
				assign out_pc_data[i] = in_PC_next;
			end
		endgenerate

		// assign out_pc_data[0] = in_PC_next;

		// assign out_pc_data[1] = in_PC_next;

		assign is_jal   = in_wb == `WB_JAL;
		assign uses_alu = in_wb == `WB_ALU;

		assign VX_writeback_inter.write_data = is_jal ? out_pc_data :
										uses_alu ? in_alu_result :
													in_mem_result;

		assign VX_writeback_inter.wb_valid    = in_valid;
		assign VX_writeback_inter.rd          = in_rd;
		assign VX_writeback_inter.wb          = in_wb;
		assign VX_writeback_inter.wb_warp_num = in_warp_num;


		assign VX_fwd_wb.dest        = VX_writeback_inter.rd;
		assign VX_fwd_wb.wb          = VX_writeback_inter.wb;
		assign VX_fwd_wb.alu_result  = in_alu_result;
		assign VX_fwd_wb.mem_data    = in_mem_result;
		assign VX_fwd_wb.PC_next     = in_PC_next;
		assign VX_fwd_wb.warp_num    = VX_writeback_inter.wb_warp_num;


endmodule // VX_writeback