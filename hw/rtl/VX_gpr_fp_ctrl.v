`include "VX_define.vh"

// control module to support multi-cycle read for fp register

module VX_gpr_fp_ctrl (
	input wire      clk,
	input wire      reset,

	input wire [`NUM_THREADS-1:0][31:0] rs1_int_data,
	input wire [`NUM_THREADS-1:0][31:0] rs2_int_data,
	input wire [`NUM_THREADS-1:0][31:0] rs1_fp_data,
	input wire [`NUM_THREADS-1:0][31:0] rs2_fp_data,

	// outputs 
	output wire [`NR_BITS-1:0]  raddr1,               
	output wire [`NR_BITS-1:0]  raddr2,               

	VX_gpr_read_if  gpr_read_if
);
    // param
	localparam GPR_DELAY_WID = 1;
	reg [GPR_DELAY_WID-1:0] multi_cyc_state;

	reg [`NUM_THREADS-1:0][31:0] tmp_rs1_data;
	reg [`NUM_THREADS-1:0][31:0] tmp_rs2_data;
    reg [`NUM_THREADS-1:0][31:0] rs1_data;
	reg [`NUM_THREADS-1:0][31:0] rs2_data;
	reg [`NUM_THREADS-1:0][31:0] rs3_data;

	wire gpr_delay;

	wire gpr_fire = gpr_read_if.valid && gpr_read_if.ready;

	always @(posedge clk) begin
		if (reset) begin
			multi_cyc_state <= 0;
        end else if (gpr_delay) begin
			multi_cyc_state <= 1;
        end else if (gpr_fire) begin
			multi_cyc_state <= 0;
        end
	end

    // select rs1 data

	always @(posedge clk) begin
		if (reset) begin
			tmp_rs1_data <= 0;
        end else begin
			if (gpr_delay) begin
				if (gpr_read_if.rs1_is_fp) begin
					tmp_rs1_data <= rs1_fp_data;
				end else begin
					tmp_rs1_data <= rs1_int_data;
				end
			end
		end
	end

	// select rs2 data

	always @(posedge clk) begin
		if(reset) begin
			tmp_rs2_data <= 0;
		end else begin
			if (gpr_delay) begin
				if (gpr_read_if.rs2_is_fp) begin
					tmp_rs2_data <= rs2_fp_data;
				end else begin
					tmp_rs2_data <= rs2_int_data;
				end
			end
		end
	end

	// outputs
	
    assign gpr_delay = gpr_read_if.valid && gpr_read_if.use_rs3 && (0 == multi_cyc_state);

	assign raddr1 = multi_cyc_state ? gpr_read_if.rs3 : gpr_read_if.rs1;
	assign raddr2 = gpr_read_if.rs2;

	always @(*) begin
		if (gpr_read_if.use_rs3) begin
			rs1_data = tmp_rs1_data;
			rs2_data = tmp_rs2_data;
			rs3_data = rs1_fp_data;
		end else begin
			rs1_data = gpr_read_if.rs1_is_fp ? rs1_fp_data : rs1_int_data;
			rs2_data = gpr_read_if.rs2_is_fp ? rs2_fp_data : rs2_int_data;
			rs3_data = {`NUM_THREADS{32'h8000_0000}}; // default value: -0 in single fp
		end
	end

    assign gpr_read_if.ready    = ~gpr_delay;
	assign gpr_read_if.rs1_data = rs1_data;
    assign gpr_read_if.rs2_data = rs2_data;
    assign gpr_read_if.rs3_data = rs3_data;

endmodule