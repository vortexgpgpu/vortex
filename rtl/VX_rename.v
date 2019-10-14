

module VX_rename (
	input wire           clk,
	input wire[`NW_M1:0] warp_num,
	input wire[4:0]      rs1,
	input wire[4:0]      rs2,
	input wire[4:0]      rd,

	output wire          stall,
);


	reg[31:0] rename[`NW-1:0];


	assign stall = rename[warp_num][rs1] || rename[warp_num][rs2];

	alwa


endmodule