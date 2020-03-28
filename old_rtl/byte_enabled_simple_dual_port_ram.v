
`include "VX_define.v"


module byte_enabled_simple_dual_port_ram
(
	input we, clk,
	input wire reset,
	input wire[4:0] waddr, raddr1, raddr2,
	input wire[`NT_M1:0] be,
	input wire[`NT_M1:0][31:0] wdata,
	output reg[`NT_M1:0][31:0] q1, q2
);

		// integer regi;
		// integer threadi;

	//     Thread   Byte  Bit
	logic [`NT_M1:0][3:0][7:0] GPR[31:0];

	// initial begin
	// 	for (ini = 0; ini < 32; ini = ini + 1) GPR[ini] = 0;
	// end

	integer ini;
	always@(posedge clk, posedge reset) begin
		if (reset) begin
			for (ini = 0; ini < 32; ini = ini + 1) GPR[ini] <= 0;
		end else if(we) begin
			integer thread_ind;
			for (thread_ind = 0; thread_ind <= `NT_M1; thread_ind = thread_ind + 1) begin
				if(be[thread_ind]) GPR[waddr][thread_ind][0] <= wdata[thread_ind][7:0];
				if(be[thread_ind]) GPR[waddr][thread_ind][1] <= wdata[thread_ind][15:8];
				if(be[thread_ind]) GPR[waddr][thread_ind][2] <= wdata[thread_ind][23:16];
				if(be[thread_ind]) GPR[waddr][thread_ind][3] <= wdata[thread_ind][31:24];
			end
		end			
		// $display("^^^^^^^^^^^^^^^^^^^^^^^");
		// for (regi = 0; regi <= 31; regi = regi + 1) begin
		// 	for (threadi = 0; threadi <= `NT_M1; threadi = threadi + 1) begin
		// 		if (GPR[regi][threadi] != 0) $display("$%d: %h",regi, GPR[regi][threadi]);
		// 	end
		// end

	end
	
	assign q1 = GPR[raddr1];
	assign q2 = GPR[raddr2];

	// assign q1 = (raddr1 == waddr && (we)) ? wdata : GPR[raddr1];
	// assign q2 = (raddr2 == waddr && (we)) ? wdata : GPR[raddr2];

endmodule
