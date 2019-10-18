
`include "VX_define.v"


module VX_writeback (
		// Mem WB info
		VX_inst_mem_wb_inter     VX_mem_wb,
		// EXEC Unit WB info
		VX_inst_exec_wb_inter    VX_inst_exec_wb,

		// Actual WB to GPR
		VX_wb_inter              VX_writeback_inter
	);



		wire exec_wb = (VX_inst_exec_wb.wb != 0) && (|VX_inst_exec_wb.wb_valid);
		wire mem_wb  = (VX_mem_wb.wb       != 0) && (|VX_mem_wb.wb_valid);


		assign VX_writeback_inter.write_data  = exec_wb ? VX_inst_exec_wb.alu_result :
		                                        mem_wb  ? VX_mem_wb.loaded_data      :
		                                        0;


		assign VX_writeback_inter.wb_valid    = exec_wb ? VX_inst_exec_wb.wb_valid :
		                                        mem_wb  ? VX_mem_wb.wb_valid       :
		                                        0;                             

		assign VX_writeback_inter.rd          = exec_wb ? VX_inst_exec_wb.rd :
		                                        mem_wb  ? VX_mem_wb.rd       :
		                                        0;

		assign VX_writeback_inter.wb          = exec_wb ? VX_inst_exec_wb.wb :
		                                        mem_wb  ? VX_mem_wb.wb       :
		                                        0;   

		assign VX_writeback_inter.wb_warp_num = exec_wb ? VX_inst_exec_wb.wb_warp_num :
		                                        mem_wb  ? VX_mem_wb.wb_warp_num       :
		                                        0;    		

		// wire[`NT_M1:0][31:0] in_alu_result = VX_mw_wb.alu_result;
		// wire[`NT_M1:0][31:0] in_mem_result = VX_mw_wb.mem_result;
		// wire[4:0]            in_rd         = VX_mw_wb.rd;
		// wire[1:0]            in_wb         = VX_mw_wb.wb;
		// wire[31:0]           in_PC_next    = VX_mw_wb.PC_next;
		// wire[`NT_M1:0]       in_valid      = VX_mw_wb.valid;
		// wire [`NW_M1:0]      in_warp_num   = VX_mw_wb.warp_num;

		// wire is_jal;
		// wire uses_alu;

		// wire[`NT_M1:0][31:0] out_pc_data;


		// genvar i;
		// generate
		// 	for (i = 0; i < `NT; i=i+1)
		// 	begin
		// 		assign out_pc_data[i] = in_PC_next;
		// 	end
		// endgenerate

		// // assign out_pc_data[0] = in_PC_next;

		// // assign out_pc_data[1] = in_PC_next;

		// assign is_jal   = in_wb == `WB_JAL;
		// assign uses_alu = in_wb == `WB_ALU;

		// assign VX_writeback_inter.write_data = is_jal ? out_pc_data :
		// 								uses_alu ? in_alu_result :
		// 											in_mem_result;

		// assign VX_writeback_inter.wb_valid    = in_valid;
		// assign VX_writeback_inter.rd          = in_rd;
		// assign VX_writeback_inter.wb          = in_wb;
		// assign VX_writeback_inter.wb_warp_num = in_warp_num;


endmodule // VX_writeback