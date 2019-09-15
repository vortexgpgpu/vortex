

module VX_register_file (
  input wire        clk,
  input wire        in_wb_warp,
  input wire        in_valid,
  input wire        in_write_register,
  input wire[4:0]   in_rd,
  input wire[31:0]  in_data,
  input wire[4:0]   in_src1,
  input wire[4:0]   in_src2,

  output wire[31:0][31:0]  out_regs,
  output reg[31:0]   out_src1_data,
  output reg[31:0]   out_src2_data
);

	reg[31:0][31:0] registers;

	wire[31:0] write_data;

	wire[4:0] write_register;

	wire write_enable;

	reg[5:0] i;
	always @(posedge clk) begin
		$display("*************");
		if (write_enable && in_wb_warp)
			$display("writing: %d = %h",in_rd, in_data);

		for (i = 0; i < 32; i++) begin
			if (registers[i[4:0]] != 0)
				$display("%d: %h",i, registers[i[4:0]]);
		end
	end

	// always @(*) begin
	// 	$display("TID: %d: %h",10,registers[10]);
	// 	$display("WID: %d: %h",11,registers[11]);
	// end

	assign out_regs = registers;

	assign write_data     = in_data;
	assign write_register = in_rd;

	assign write_enable   = (in_write_register && (in_rd != 5'h0)) && in_valid;

	always @(posedge clk) begin
		if(write_enable && in_wb_warp) begin
			// $display("RF: Writing %h to %d",write_data, write_register);
			registers[write_register] <= write_data;
		end
	end

	// always @(negedge clk) begin
		assign out_src1_data = registers[in_src1];
		assign out_src2_data = registers[in_src2];
	// end

	always @(*) begin 
		$display("Reading Data 1: %d = %h",in_src1, out_src1_data);
		$display("Reading Data 2: %d = %h",in_src2, out_src2_data);
	end


endmodule
