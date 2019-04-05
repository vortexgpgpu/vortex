
`include "VX_define.v"



module VX_shared_memory(
		input  wire       clk,
		input  wire[31:0] in_address[`NT_M1:0],
		input  wire[2:0]  in_mem_read,
		input  wire[2:0]  in_mem_write,
		input  wire       in_valid[`NT_M1:0],
		input  wire[31:0] in_data[`NT_M1:0],

		output reg[31:0] out_data[`NT_M1:0]

	);



	reg[31:0] mem[255:0]; // 2^2 * 2^8 = 2^10 = 1kb of memory


	always @(posedge clk)
	begin
		if ((in_mem_write == `SW_MEM_WRITE) && in_valid)
		begin
			mem[in_address[0][9:2]] <= in_data;
		end

		if (in_mem_read == `LW_MEM_READ)
		begin
			assign out_data[0] = mem[in_address[0][9:2]];
		end
		
	end


endmodule // VX_shared_memory