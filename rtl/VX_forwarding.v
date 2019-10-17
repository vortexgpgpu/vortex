
`include "VX_define.v"

module VX_forwarding (
		// INFO FROM DECODE
		VX_forward_reqeust_inter       VX_fwd_req_de,
		VX_forward_exe_inter           VX_fwd_exe,
		VX_forward_mem_inter           VX_fwd_mem,
		VX_forward_wb_inter            VX_fwd_wb,

		VX_forward_response_inter      VX_fwd_rsp,
		output wire                    out_fwd_stall
	);


		wire[4:0]      in_decode_src1        = VX_fwd_req_de.src1;
		wire[4:0]      in_decode_src2        = VX_fwd_req_de.src2;
		wire[`NW_M1:0] in_decode_warp_num    = VX_fwd_req_de.warp_num;

		wire[4:0]            in_execute_dest        = VX_fwd_exe.dest;
		wire[1:0]            in_execute_wb          = VX_fwd_exe.wb;
		wire[`NT_M1:0][31:0] in_execute_alu_result  = VX_fwd_exe.alu_result;
		wire[31:0]           in_execute_PC_next     = VX_fwd_exe.PC_next;
		wire[`NW_M1:0]       in_execute_warp_num    = VX_fwd_exe.warp_num;

		wire[4:0]            in_memory_dest         = VX_fwd_mem.dest;
		wire[1:0]            in_memory_wb           = VX_fwd_mem.wb;
		wire[`NT_M1:0][31:0] in_memory_alu_result   = VX_fwd_mem.alu_result;
		wire[`NT_M1:0][31:0] in_memory_mem_data     = VX_fwd_mem.mem_data;
		wire[31:0]           in_memory_PC_next      = VX_fwd_mem.PC_next;
		wire[`NW_M1:0]       in_memory_warp_num     = VX_fwd_mem.warp_num;

		wire[4:0]            in_writeback_dest       = VX_fwd_wb.dest;
		wire[1:0]            in_writeback_wb         = VX_fwd_wb.wb;
		wire[`NT_M1:0][31:0] in_writeback_alu_result = VX_fwd_wb.alu_result;
		wire[`NT_M1:0][31:0] in_writeback_mem_data   = VX_fwd_wb.mem_data;
		wire[31:0]           in_writeback_PC_next    = VX_fwd_wb.PC_next;
		wire[`NW_M1:0]       in_writeback_warp_num   = VX_fwd_wb.warp_num;


		wire                 out_src1_fwd;
		wire                 out_src2_fwd;
		wire[`NT_M1:0][31:0] out_src1_fwd_data;
		wire[`NT_M1:0][31:0] out_src2_fwd_data;


		assign VX_fwd_rsp.src1_fwd      = out_src1_fwd;
		assign VX_fwd_rsp.src2_fwd      = out_src2_fwd;
		assign VX_fwd_rsp.src1_fwd_data = out_src1_fwd_data;
		assign VX_fwd_rsp.src2_fwd_data = out_src2_fwd_data;




		wire exe_mem_read;
		wire mem_mem_read;
		wire wb_mem_read ;
		wire exe_jal;
		wire mem_jal;
		wire wb_jal ;
		wire src1_exe_fwd;
		wire src1_mem_fwd;
		wire src1_wb_fwd;
		wire src2_exe_fwd;
		wire src2_mem_fwd;
		wire src2_wb_fwd;

		wire[`NT_M1:0][31:0] use_execute_PC_next;
		wire[`NT_M1:0][31:0] use_memory_PC_next;
		wire[`NT_M1:0][31:0] use_writeback_PC_next;


		genvar index;
		generate  
		for (index=0; index < `NT; index=index+1)  
		  begin: gen_code_label  
			assign use_execute_PC_next[index]   = in_execute_PC_next;
			assign use_memory_PC_next[index]    = in_memory_PC_next;
			assign use_writeback_PC_next[index] = in_writeback_PC_next;
		  end  
		endgenerate  


		assign exe_mem_read = (in_execute_wb   == `WB_MEM);
		assign mem_mem_read = (in_memory_wb    == `WB_MEM);
		assign wb_mem_read  = (in_writeback_wb == `WB_MEM);

		assign exe_jal = (in_execute_wb   == `WB_JAL);
		assign mem_jal = (in_memory_wb    == `WB_JAL);
		assign wb_jal  = (in_writeback_wb == `WB_JAL);



		// SRC1
		assign src1_exe_fwd = ((in_decode_src1 == in_execute_dest) && 
								(in_decode_src1 != `ZERO_REG) &&
			                     (in_execute_wb != `NO_WB))   &&
						    (in_decode_warp_num == in_execute_warp_num);

		assign src1_mem_fwd = ((in_decode_src1 == in_memory_dest) &&
							    (in_decode_src1 != `ZERO_REG) &&
			                      (in_memory_wb != `NO_WB) &&
			                      (!src1_exe_fwd))  &&
		                    (in_decode_warp_num == in_memory_warp_num);

		assign src1_wb_fwd = ((in_decode_src1 == in_writeback_dest) &&
							   (in_decode_src1 != `ZERO_REG) &&
			                  (in_writeback_wb != `NO_WB) &&
			            (in_writeback_warp_num == in_decode_warp_num) &&
			                      (!src1_exe_fwd) &&
			                      (!src1_mem_fwd));


		// assign out_src1_fwd  = src1_exe_fwd || src1_mem_fwd || (src1_wb_fwd && 0);
		assign out_src1_fwd  = 0;





		// SRC2
		assign src2_exe_fwd = ((in_decode_src2 == in_execute_dest) && 
								(in_decode_src2 != `ZERO_REG) &&
			                     (in_execute_wb != `NO_WB)) &&
		                    (in_decode_warp_num == in_execute_warp_num);

		assign src2_mem_fwd = ((in_decode_src2 == in_memory_dest) &&
								(in_decode_src2 != `ZERO_REG) &&
			                      (in_memory_wb != `NO_WB) &&
			                      (!src2_exe_fwd)) &&
		                    (in_decode_warp_num == in_memory_warp_num);

		assign src2_wb_fwd = ((in_decode_src2 == in_writeback_dest) &&
							   (in_decode_src2 != `ZERO_REG) &&
			                  (in_writeback_wb != `NO_WB) &&
			                      (!src2_exe_fwd) &&
			                      (!src2_mem_fwd)) &&
		                (in_writeback_warp_num == in_decode_warp_num);


		// assign out_src2_fwd  = src2_exe_fwd || src2_mem_fwd || (src2_wb_fwd && 0);
		assign out_src2_fwd  = 0;




		// wire exe_mem_read_stall = ((src1_exe_fwd || src2_exe_fwd) && exe_mem_read) ? `STALL : `NO_STALL;
		// wire mem_mem_read_stall = ((src1_mem_fwd || src2_mem_fwd) && mem_mem_read) ? `STALL : `NO_STALL;
		wire exe_mem_read_stall = `NO_STALL;
		wire mem_mem_read_stall = `NO_STALL;

		// assign out_fwd_stall = exe_mem_read_stall || mem_mem_read_stall; 
		assign out_fwd_stall = 0; 

		// always @(*) begin
		// 	if (out_fwd_stall) $display("FWD STALL");
		// end

		assign out_src1_fwd_data = src1_exe_fwd ? ((exe_jal) ? use_execute_PC_next : in_execute_alu_result) :
			                          (src1_mem_fwd) ? ((mem_jal) ? use_memory_PC_next : (mem_mem_read ? in_memory_mem_data : in_memory_alu_result)) :
									    ( src1_wb_fwd ) ?  (wb_jal ? use_writeback_PC_next : (wb_mem_read ?  in_writeback_mem_data : in_writeback_alu_result)) :
										 	in_execute_alu_result; // last one should be deadbeef

		assign out_src2_fwd_data = src2_exe_fwd ? ((exe_jal) ? use_execute_PC_next : in_execute_alu_result) :
			                        (src2_mem_fwd) ? ((mem_jal) ? use_memory_PC_next : (mem_mem_read ? in_memory_mem_data : in_memory_alu_result)) :
									    ( src2_wb_fwd ) ?  (wb_jal ? use_writeback_PC_next : (wb_mem_read ?  in_writeback_mem_data : in_writeback_alu_result)) :
										 	in_execute_alu_result; // last one should be deadbeef




endmodule // VX_forwarding







