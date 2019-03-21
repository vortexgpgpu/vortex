

module VX_register_file (
  input wire        clk,
  input wire        in_write_register,
  input wire[4:0]   in_rd,
  input wire[31:0]  in_data,
  input wire[4:0]   in_src1,
  input wire[4:0]   in_src2,

  output wire[31:0] out_src1_data,
  output wire[31:0] out_src2_data
);

	reg[31:0] registers[31:0];

	wire[31:0] write_data;

	wire[4:0] write_register;

	wire write_enable;



	assign write_data     = in_data;
	assign write_register = in_rd;

	assign write_enable   = in_write_register && (in_rd != 5'h0);

	always @(posedge clk) begin
		if(write_enable) begin
			registers[write_register] <= write_data;
		end
	end

	assign out_src1_data = registers[in_src1];
	assign out_src2_data = registers[in_src2];


endmodule
