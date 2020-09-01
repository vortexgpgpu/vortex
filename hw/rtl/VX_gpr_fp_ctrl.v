`include "VX_define.vh"

// control module to support multi-cycle read for fp register

module VX_gpr_fp_ctrl (
	input wire clk,
	input wire reset,

	input wire [`NUM_THREADS-1:0][31:0] rs1_data,
	input wire [`NUM_THREADS-1:0][31:0] rs2_data,

	// outputs 
	output wire [`NW_BITS+`NR_BITS-1:0]	raddr1,
	VX_gpr_read_if	gpr_read_if
);

    reg [`NUM_THREADS-1:0][31:0] rs1_tmp_data, rs2_tmp_data, rs3_tmp_data;
	reg read_rs1;
	reg [`NW_BITS-1:0] rs3_wid;

	wire rs3_delay = gpr_read_if.valid && gpr_read_if.use_rs3 && read_rs1;
	wire read_fire = gpr_read_if.valid && gpr_read_if.ready_out;

	always @(posedge clk) begin
		if (reset) begin
			rs1_tmp_data <= 0;
			rs2_tmp_data <= 0;
			rs3_tmp_data <= 0;
			read_rs1     <= 1;
			rs3_wid      <= 0;
		end else begin
			if (rs3_delay) begin
				read_rs1 <= 0;
				rs3_wid  <= gpr_read_if.wid;
			end else if (read_fire) begin
				read_rs1 <= 1;
			end

			if (read_rs1) begin
				rs1_tmp_data <= rs1_data;
			end
			rs2_tmp_data <= rs2_data;
			rs3_tmp_data <= rs1_data;

			assert(read_rs1 || rs3_wid == gpr_read_if.wid);
		end	
	end

	always @(posedge clk) begin
		
	end

	// outputs
	wire [`NR_BITS-1:0] rs1 = read_rs1 ? gpr_read_if.rs1 : gpr_read_if.rs3;
	assign raddr1 = {gpr_read_if.wid, rs1};
    assign gpr_read_if.ready_in = ~rs3_delay;
	assign gpr_read_if.rs1_data = rs1_tmp_data;
    assign gpr_read_if.rs2_data = rs2_tmp_data;
    assign gpr_read_if.rs3_data = rs3_tmp_data;

endmodule