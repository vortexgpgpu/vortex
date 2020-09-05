`include "VX_define.vh"

// control module to support multi-cycle read for fp register

module VX_gpr_fp_ctrl (
	input wire clk,
	input wire reset,

	input wire [`NUM_THREADS-1:0][31:0] rs1_data,
	input wire [`NUM_THREADS-1:0][31:0] rs2_data,
	VX_gpr_req_if   gpr_req_if,

	// outputs 
	output wire [`NW_BITS+`NR_BITS-1:0]	raddr1,
	VX_gpr_rsp_if	gpr_rsp_if
);

    reg [`NUM_THREADS-1:0][31:0] rsp_rs1_data, rsp_rs2_data, rsp_rs3_data;
	reg rsp_valid;
	reg [31:0] rsp_pc;	
	reg [`NW_BITS-1:0] rsp_wid;
	reg read_rs1;

	wire rs3_delay = gpr_req_if.valid && gpr_req_if.use_rs3 && read_rs1;
	wire read_fire = gpr_req_if.valid && gpr_rsp_if.ready;

	always @(posedge clk) begin
		if (reset) begin
			rsp_valid    <= 0;
			rsp_pc       <= 0;		
			rsp_rs1_data <= 0;
			rsp_rs2_data <= 0;
			rsp_rs3_data <= 0;	
			rsp_wid      <= 0;			
			read_rs1     <= 1;			
		end else begin
			if (rs3_delay) begin
				read_rs1 <= 0;
				rsp_wid  <= gpr_req_if.wid;
			end else if (read_fire) begin
				read_rs1 <= 1;
			end

			rsp_valid    <= gpr_req_if.valid;
			rsp_wid      <= gpr_req_if.wid;
			rsp_pc       <= gpr_req_if.PC;		

			if (read_rs1) begin
				rsp_rs1_data <= rs1_data;
			end			
			rsp_rs2_data <= rs2_data;
			rsp_rs3_data <= rs1_data;

			assert(read_rs1 || rsp_wid == gpr_req_if.wid);
		end	
	end

	always @(posedge clk) begin
		
	end

	// outputs
	wire [`NR_BITS-1:0] rs1 = read_rs1 ? gpr_req_if.rs1 : gpr_req_if.rs3;
	assign raddr1 = {gpr_req_if.wid, rs1};
    assign gpr_req_if.ready = ~rs3_delay;

	assign gpr_rsp_if.valid    = rsp_valid;
	assign gpr_rsp_if.wid      = rsp_wid;
	assign gpr_rsp_if.PC       = rsp_pc;
	assign gpr_rsp_if.rs1_data = rsp_rs1_data;
    assign gpr_rsp_if.rs2_data = rsp_rs2_data;
    assign gpr_rsp_if.rs3_data = rsp_rs3_data;

endmodule