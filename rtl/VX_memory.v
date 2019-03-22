
`include "VX_define.v"


module VX_memory (
		input wire[31:0]  in_alu_result,
		input wire[2:0]   in_mem_read, 
		input wire[2:0]   in_mem_write,
		input wire[4:0]   in_rd,
		input wire[1:0]   in_wb,
		input wire[4:0]   in_rs1,
		input wire[4:0]   in_rs2,
		input wire[31:0]  in_rd2,
		input wire[31:0]  in_PC_next,
		input wire[31:0]  in_curr_PC,
		input wire[31:0]  in_branch_offset,
		input wire[2:0]   in_branch_type, 
		input wire        in_valid,
		input wire[31:0]  in_cache_driver_out_data,

		output wire[31:0] out_alu_result,
		output wire[31:0] out_mem_result,
		output wire[4:0]  out_rd,
		output wire[1:0]  out_wb,
		output wire[4:0]  out_rs1,
		output wire[4:0]  out_rs2,
		output reg        out_branch_dir,
		output wire[31:0] out_branch_dest,
		output wire       out_delay,
		output wire[31:0] out_PC_next,
		output wire       out_valid,
		output wire[31:0] out_cache_driver_in_address,
		output wire[2:0]  out_cache_driver_in_mem_read,
		output wire[2:0]  out_cache_driver_in_mem_write,
		output wire[31:0] out_cache_driver_in_data 
	);	

		assign out_delay = 1'b0;

		assign out_cache_driver_in_address   = in_alu_result;
		assign out_cache_driver_in_mem_read  = in_mem_read;
		assign out_cache_driver_in_mem_write = in_mem_write;
		assign out_cache_driver_in_data      = in_rd2;


		assign out_mem_result = in_cache_driver_out_data;
		assign out_alu_result = in_alu_result;
		assign out_rd = in_rd;
		assign out_wb = in_wb;
		assign out_rs1 = in_rs1;
		assign out_rs2 = in_rs2;
		assign out_PC_next = in_PC_next;

		assign out_valid = in_valid;


		assign out_branch_dest = $signed(in_curr_PC) + ($signed(in_branch_offset) << 1);


		always @(*) begin
			case(in_branch_type)
				`BEQ:  out_branch_dir = (in_alu_result == 0)     ? `TAKEN     : `NOT_TAKEN;
				`BNE:  out_branch_dir = (in_alu_result == 0)     ? `NOT_TAKEN : `TAKEN;
				`BLT:  out_branch_dir = (in_alu_result[31] == 0) ? `NOT_TAKEN : `TAKEN;
				`BGT:  out_branch_dir = (in_alu_result[31] == 0) ? `TAKEN     : `NOT_TAKEN;
				`BLTU: out_branch_dir = (in_alu_result[31] == 0) ? `NOT_TAKEN : `TAKEN;
				`BGTU: out_branch_dir = (in_alu_result[31] == 0) ? `TAKEN     : `NOT_TAKEN;
				`NO_BRANCH: out_branch_dir = `NOT_TAKEN;
				default:    out_branch_dir = `NOT_TAKEN;
			endcase // in_branch_type
		end



endmodule // Memory


