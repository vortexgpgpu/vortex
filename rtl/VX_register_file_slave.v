




module VX_register_file_slave (
  input wire         clk,
  input wire         in_warp,
  input wire         in_valid,
  input wire         in_write_register,
  input wire[4:0]    in_rd,
  input wire[31:0]   in_data,
  input wire[4:0]    in_src1,
  input wire[4:0]    in_src2,
  input wire         in_clone,
  input wire         in_to_clone,
  input wire[31:0]   in_regs[31:0],

  output reg[31:0]   out_src1_data,
  output reg[31:0]   out_src2_data
);

	reg[31:0] registers[31:0];

	wire[31:0] write_data;

	wire[4:0] write_register;

	wire write_enable;

	// reg[5:0] i;
	// always @(posedge clk) begin
	// 	for (i = 0; i < 32; i++) begin
	// 		$display("%d: %h",i, registers[i[4:0]]);
	// 	end
	// end

	// integer i;

	assign write_data     = in_data;
	assign write_register = in_rd;

	assign write_enable   = (in_write_register && (in_rd != 5'h0)) && in_valid;

	always @(posedge clk) begin
		if(write_enable && !in_clone && in_warp) begin
			// $display("RF: Writing %h to %d",write_data, write_register);
			registers[write_register] <= write_data;
		end else if (in_clone && in_to_clone) begin
			// $display("CLONING IN SLAVE");
			registers <= in_regs;
		end
	end

	// always @(posedge clk) begin
	// 	for (i = 0; i < 32; i = i + 1)
	// 		$display("(%d): %x", i, registers[i]);
	
	// end

	always @(negedge clk) begin
		out_src1_data <= registers[in_src1];
		out_src2_data <= registers[in_src2];
	end


endmodule
