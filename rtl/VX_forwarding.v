
`include "VX_define.v"

module VX_forwarding (
		// INFO FROM DECODE
		input wire[4:0]  in_decode_src1,
		input wire[4:0]  in_decode_src2,
		input wire[11:0] in_decode_csr_address, 

		// INFO FROM EXE
		input wire[4:0]  in_execute_dest,
		input wire[1:0]  in_execute_wb,
		input wire[31:0] in_execute_alu_result,
		input wire[31:0] in_execute_PC_next,
		input wire       in_execute_is_csr,
		input wire[11:0] in_execute_csr_address,

		// INFO FROM MEM
		input wire[4:0]  in_memory_dest,
		input wire[1:0]  in_memory_wb,
		input wire[31:0] in_memory_alu_result,
		input wire[31:0] in_memory_mem_data,
		input wire[31:0] in_memory_PC_next,
		input wire       in_memory_is_csr,
		input wire[11:0] in_memory_csr_address,
		input wire[31:0] in_memory_csr_result,

		// INFO FROM WB
		input wire[4:0]  in_writeback_dest,
		input wire[1:0]  in_writeback_wb,
		input wire[31:0] in_writeback_alu_result,
		input wire[31:0] in_writeback_mem_data,
		input wire[31:0] in_writeback_PC_next,

		// OUT SIGNALS
		output wire       out_src1_fwd,
		output wire       out_src2_fwd,
		output wire       out_csr_fwd,
		output wire[31:0] out_src1_fwd_data,
		output wire[31:0] out_src2_fwd_data,
		output wire[31:0] out_csr_fwd_data,
		output wire       out_fwd_stall
	);



		wire exe_mem_read;
		wire mem_mem_read;
		wire wb_mem_read ;
		wire exe_jal;
		wire mem_jal;
		wire wb_jal ;
		wire exe_csr;
		wire mem_csr;
		wire src1_exe_fwd;
		wire src1_mem_fwd;
		wire src1_wb_fwd;
		wire src2_exe_fwd;
		wire src2_mem_fwd;
		wire src2_wb_fwd;
		wire csr_exe_fwd;
		wire csr_mem_fwd;


		assign exe_mem_read = (in_execute_wb   == `WB_MEM);
		assign mem_mem_read = (in_memory_wb    == `WB_MEM);
		assign wb_mem_read  = (in_writeback_wb == `WB_MEM);

		assign exe_jal = (in_execute_wb   == `WB_JAL);
		assign mem_jal = (in_memory_wb    == `WB_JAL);
		assign wb_jal  = (in_writeback_wb == `WB_JAL);

		assign exe_csr = (in_execute_is_csr == 1'b1);
		assign mem_csr = (in_memory_is_csr == 1'b1);


		// SRC1
		assign src1_exe_fwd = ((in_decode_src1 == in_execute_dest) && 
								(in_decode_src1 != `ZERO_REG) &&
			                     (in_execute_wb != `NO_WB));

		assign src1_mem_fwd = ((in_decode_src1 == in_memory_dest) &&
							    (in_decode_src1 != `ZERO_REG) &&
			                      (in_memory_wb != `NO_WB) &&
			                      (!src1_exe_fwd));

		assign src1_wb_fwd = ((in_decode_src1 == in_writeback_dest) &&
							   (in_decode_src1 != `ZERO_REG) &&
			                  (in_writeback_wb != `NO_WB) &&
			                      (!src1_exe_fwd) &&
			                      (!src1_mem_fwd));


		assign out_src1_fwd  = src1_exe_fwd || src1_mem_fwd || src1_wb_fwd; // COMMENT





		// SRC2
		assign src2_exe_fwd = ((in_decode_src2 == in_execute_dest) && 
								(in_decode_src2 != `ZERO_REG) &&
			                     (in_execute_wb != `NO_WB));

		assign src2_mem_fwd = ((in_decode_src2 == in_memory_dest) &&
								(in_decode_src2 != `ZERO_REG) &&
			                      (in_memory_wb != `NO_WB) &&
			                      (!src2_exe_fwd));

		assign src2_wb_fwd = ((in_decode_src2 == in_writeback_dest) &&
							   (in_decode_src2 != `ZERO_REG) &&
			                  (in_writeback_wb != `NO_WB) &&
			                      (!src2_exe_fwd) &&
			                      (!src2_mem_fwd));


		assign out_src2_fwd  = src2_exe_fwd || src2_mem_fwd || src2_wb_fwd; // COMMENT



		// CSR
		assign csr_exe_fwd = (in_decode_csr_address == in_execute_csr_address) && exe_csr;
		assign csr_mem_fwd = (in_decode_csr_address == in_memory_csr_address)  && mem_csr && !csr_exe_fwd;

		assign out_csr_fwd      = csr_exe_fwd || csr_mem_fwd; // COMMENT


		wire exe_mem_read_stall = ((src1_exe_fwd || src2_exe_fwd) && exe_mem_read) ? `STALL : `NO_STALL;
		wire mem_mem_read_stall = ((src1_mem_fwd || src2_mem_fwd) && mem_mem_read) ? `STALL : `NO_STALL;

		assign out_fwd_stall = exe_mem_read_stall || mem_mem_read_stall; 

		// always @(*) begin
		// 	if (out_fwd_stall) $display("FWD STALL");
		// end

		assign out_src1_fwd_data = src1_exe_fwd ? ((exe_jal) ? in_execute_PC_next : in_execute_alu_result) :
			                          (src1_mem_fwd) ? ((mem_jal) ? in_memory_PC_next : (mem_mem_read ? in_memory_mem_data : in_memory_alu_result)) :
									    ( src1_wb_fwd ) ?  (wb_jal ? in_writeback_PC_next : (wb_mem_read ?  in_writeback_mem_data : in_writeback_alu_result)) :
										 	32'hdeadbeef; // COMMENT

		assign out_src2_fwd_data = src2_exe_fwd ? ((exe_jal) ? in_execute_PC_next : in_execute_alu_result) :
			                        (src2_mem_fwd) ? ((mem_jal) ? in_memory_PC_next : (mem_mem_read ? in_memory_mem_data : in_memory_alu_result)) :
									    ( src2_wb_fwd ) ?  (wb_jal ? in_writeback_PC_next : (wb_mem_read ?  in_writeback_mem_data : in_writeback_alu_result)) :
										 	32'hdeadbeef; // COMMENT
		
		assign out_csr_fwd_data = csr_exe_fwd ? in_execute_alu_result :
									 csr_mem_fwd ? in_memory_csr_result :
									 	    	32'hdeadbeef; // COMMENT



endmodule // VX_forwarding







