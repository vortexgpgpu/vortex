
module VX_register_file_master_slave (
  input wire         clk,
  input wire         in_wb_warp,
  input wire         in_valid,
  input wire         in_write_register,
  input wire[4:0]    in_rd,
  input wire[31:0]   in_data,
  input wire[4:0]    in_src1,
  input wire[4:0]    in_src2,
  input wire         in_wspawn,
  input wire         in_to_wspawn,
  input wire[31:0][31:0]   in_wspawn_regs,

  output reg[31:0]   out_src1_data,
  output reg[31:0]   out_src2_data,
  output wire[31:0][31:0]  out_regs
);

	reg[31:0][31:0] registers;

	wire[31:0] write_data;

	wire[4:0] write_register;

	wire write_enable;


	assign out_regs = registers;

	// reg[5:0] i;
	// always @(posedge clk) begin
	// 	for (i = 0; i < 32; i++) begin
	// 		$display("%d: %h",i, registers[i[4:0]]);
	// 	end
	// end

	// integer i;

	assign write_data     = in_data;
	assign write_register = in_rd;

	// always @(*) begin
	// 	$display("TID: %d: %h",10,registers[10]);
	// 	$display("WID: %d: %h",11,registers[11]);
	// end

	assign write_enable   = (in_write_register && (in_rd != 5'h0)) && in_valid && in_wb_warp;

	always @(posedge clk) begin
		if(write_enable && !in_wspawn) begin
			// $display("RF: Writing %h to %d",write_data, write_register);
			registers[write_register] <= write_data;
		end else if (in_wspawn && in_to_wspawn) begin
			// $display("WSPAWN IN MASTER SLAVE");
			registers <= in_wspawn_regs;
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