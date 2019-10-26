
`include "VX_define.v"


module VX_writeback (
		// Mem WB info
		VX_inst_mem_wb_inter     VX_mem_wb,
		// EXEC Unit WB info
		VX_inst_exec_wb_inter    VX_inst_exec_wb,
		// CSR Unit WB info
		VX_csr_wb_inter          VX_csr_wb,

		// Actual WB to GPR
		VX_wb_inter              VX_writeback_inter,
		output wire              no_slot_mem
	);



		wire exec_wb = (VX_inst_exec_wb.wb != 0) && (|VX_inst_exec_wb.wb_valid);
		wire mem_wb  = (VX_mem_wb.wb       != 0) && (|VX_mem_wb.wb_valid);
		wire csr_wb  = (VX_csr_wb.wb       != 0) && (|VX_csr_wb.valid);


		assign no_slot_mem = mem_wb && (exec_wb || csr_wb);

		assign VX_writeback_inter.write_data  = exec_wb ? VX_inst_exec_wb.alu_result :
		                                        csr_wb  ? VX_csr_wb.csr_result       :
		                                        mem_wb  ? VX_mem_wb.loaded_data      :
		                                        0;


		assign VX_writeback_inter.wb_valid    = exec_wb ? VX_inst_exec_wb.wb_valid :
		                                        csr_wb  ? VX_csr_wb.valid          :
		                                        mem_wb  ? VX_mem_wb.wb_valid       :
		                                        0;    

		assign VX_writeback_inter.rd          = exec_wb ? VX_inst_exec_wb.rd :
		                                        csr_wb  ? VX_csr_wb.rd       :
		                                        mem_wb  ? VX_mem_wb.rd       :
		                                        0;

		assign VX_writeback_inter.wb          = exec_wb ? VX_inst_exec_wb.wb :
		                                        csr_wb  ? VX_csr_wb.wb       :
		                                        mem_wb  ? VX_mem_wb.wb       :
		                                        0;   

		assign VX_writeback_inter.wb_warp_num = exec_wb ? VX_inst_exec_wb.wb_warp_num :
		                                        csr_wb  ? VX_csr_wb.warp_num          :
		                                        mem_wb  ? VX_mem_wb.wb_warp_num       :
		                                        0;    		


endmodule // VX_writeback