
`include "VX_define.v"


module VX_memory (
		VX_mem_req_inter     VX_mem_req,
		VX_inst_mem_wb_inter VX_mem_wb,

		VX_forward_mem_inter VX_fwd_mem,


		output wire          out_delay,

		VX_branch_response_inter VX_branch_rsp,


		input  wire[31:0]  in_cache_driver_out_data[`NT_M1:0],
		output wire[31:0] out_cache_driver_in_address[`NT_M1:0],
		output wire[2:0]  out_cache_driver_in_mem_read,
		output wire[2:0]  out_cache_driver_in_mem_write,
		output wire       out_cache_driver_in_valid[`NT_M1:0],
		output wire[31:0] out_cache_driver_in_data[`NT_M1:0]
	);	


		 genvar index;
		 for (index = 0; index <= `NT_M1; index = index + 1) begin
			assign out_cache_driver_in_address[index]   = VX_mem_req.alu_result[index];
			assign out_cache_driver_in_data[index]      = VX_mem_req.rd2[index];
			assign out_cache_driver_in_valid[index]     = VX_mem_req.valid[index];

			assign VX_mem_wb.mem_result[index]                = in_cache_driver_out_data[index];

		 end

		assign out_delay = 1'b0;

		assign out_cache_driver_in_mem_read  = VX_mem_req.mem_read;
		assign out_cache_driver_in_mem_write = VX_mem_req.mem_write;


		assign VX_mem_wb.alu_result = VX_mem_req.alu_result;
		assign VX_mem_wb.rd         = VX_mem_req.rd;
		assign VX_mem_wb.wb         = VX_mem_req.wb;
		assign VX_mem_wb.PC_next    = VX_mem_req.PC_next;
		assign VX_mem_wb.valid      = VX_mem_req.valid;
		assign VX_mem_wb.warp_num   = VX_mem_req.warp_num;

		assign VX_fwd_mem.dest        = VX_mem_wb.rd;
		assign VX_fwd_mem.wb          = VX_mem_wb.wb;
		assign VX_fwd_mem.alu_result  = VX_mem_wb.alu_result;
		assign VX_fwd_mem.mem_data    = VX_mem_wb.mem_result;
		assign VX_fwd_mem.PC_next     = VX_mem_wb.PC_next;
		assign VX_fwd_mem.warp_num    = VX_mem_wb.warp_num;


		reg temp_branch_dir;


		assign VX_branch_rsp.branch_dest = $signed(VX_mem_req.curr_PC) + ($signed(VX_mem_req.branch_offset) << 1);
		
		always @(*) begin
			case(VX_mem_req.branch_type)
				`BEQ:  temp_branch_dir = (VX_mem_req.alu_result[0] == 0)     ? `TAKEN     : `NOT_TAKEN;
				`BNE:  temp_branch_dir = (VX_mem_req.alu_result[0] == 0)     ? `NOT_TAKEN : `TAKEN;
				`BLT:  temp_branch_dir = (VX_mem_req.alu_result[0][31] == 0) ? `NOT_TAKEN : `TAKEN;
				`BGT:  temp_branch_dir = (VX_mem_req.alu_result[0][31] == 0) ? `TAKEN     : `NOT_TAKEN;
				`BLTU: temp_branch_dir = (VX_mem_req.alu_result[0][31] == 0) ? `NOT_TAKEN : `TAKEN; 
				`BGTU: temp_branch_dir = (VX_mem_req.alu_result[0][31] == 0) ? `TAKEN     : `NOT_TAKEN;
				`NO_BRANCH: temp_branch_dir = `NOT_TAKEN;
				default:    temp_branch_dir = `NOT_TAKEN;
			endcase // in_branch_type
		end

		assign VX_branch_rsp.branch_dir      = temp_branch_dir;
		assign VX_branch_rsp.branch_warp_num = VX_mem_req.warp_num;

endmodule // Memory


