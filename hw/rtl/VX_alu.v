`include "VX_define.vh"

module VX_alu(
	input wire clk,
	input wire reset,
	input wire[31:0]  in_1,
	input wire[31:0]  in_2,
	input wire        in_rs2_src,
	input wire[31:0]  in_itype_immed,
	input wire[19:0]  in_upper_immed,
	input wire[4:0]   in_alu_op,
	input wire[31:0]  in_curr_PC,
	output reg[31:0]  out_alu_result,
	output reg        out_alu_stall
	);

	localparam div_pipeline_len = 20;
	localparam mul_pipeline_len = 8;

	wire[31:0] unsigned_div_result;
	wire[31:0] unsigned_rem_result;
	wire[31:0] signed_div_result;
	wire[31:0] signed_rem_result;

	wire[63:0] mul_data_a, mul_data_b;
	wire[63:0] mul_result;

	wire[31:0] ALU_in1;
	wire[31:0] ALU_in2;

	VX_divide #(
		.WIDTHN(32),
		.WIDTHD(32),
		.SPEED("HIGHEST"),
		.PIPELINE(div_pipeline_len)
	) unsigned_div (
		.clock(clk),
		.aclr(1'b0),
		.clken(1'b1), // TODO this could be disabled on inactive instructions
		.numer(ALU_in1),
		.denom(ALU_in2),
		.quotient(unsigned_div_result),
		.remainder(unsigned_rem_result)
	);

	VX_divide #(
		.WIDTHN(32),
		.WIDTHD(32),
		.NREP("SIGNED"),
		.DREP("SIGNED"),
		.SPEED("HIGHEST"),
		.PIPELINE(div_pipeline_len)
	) signed_div (
		.clock(clk),
		.aclr(1'b0),
		.clken(1'b1), // TODO this could be disabled on inactive instructions
		.numer(ALU_in1),
		.denom(ALU_in2),
		.quotient(signed_div_result),
		.remainder(signed_rem_result)
	);

	VX_mult #(
		.WIDTHA(64),
		.WIDTHB(64),
		.WIDTHP(64),
		.SPEED("HIGHEST"),
		.FORCE_LE("YES"),
		.PIPELINE(mul_pipeline_len)
	) multiplier (
		.clock(clk),
		.aclr(1'b0),
		.clken(1'b1), // TODO this could be disabled on inactive instructions
		.dataa(mul_data_a),
		.datab(mul_data_b),
		.result(mul_result)
	);

	// MUL, MULH (signed*signed), MULHSU (signed*unsigned), MULHU (unsigned*unsigned)
	wire[63:0] alu_in1_signed = {{32{ALU_in1[31]}}, ALU_in1};
	wire[63:0] alu_in2_signed = {{32{ALU_in2[31]}}, ALU_in2};
	assign mul_data_a = (in_alu_op == `MULHU) ? {32'b0, ALU_in1} : alu_in1_signed;
	assign mul_data_b = (in_alu_op == `MULHU || in_alu_op == `MULHSU) ? {32'b0, ALU_in2} : alu_in2_signed;


	reg [15:0] curr_inst_delay;
	reg [15:0] inst_delay;
	reg inst_was_stalling;

	wire inst_delay_stall = inst_was_stalling ? inst_delay != 0 : curr_inst_delay != 0;
	assign out_alu_stall = inst_delay_stall;

	always @(*) begin
		case(in_alu_op)
			`DIV,
			`DIVU,
			`REM,
			`REMU: curr_inst_delay = div_pipeline_len;
			`MUL,
			`MULH,
			`MULHSU,
			`MULHU: curr_inst_delay = mul_pipeline_len;
			default: curr_inst_delay = 0;
		endcase // in_alu_op
	end

	always @(posedge clk) begin
		if (reset) begin
			inst_delay <= 0;
			inst_was_stalling <= 0;
		end
		else if (inst_delay_stall) begin
			if (inst_was_stalling) begin
				if (inst_delay > 0)
					inst_delay <= inst_delay - 1;
			end
			else begin
				inst_was_stalling <= 1;
				inst_delay <= curr_inst_delay - 1;
			end
		end
		else begin
			inst_was_stalling <= 0;
		end
	end

	`ifdef SYN_FUNC
	wire which_in2;
	wire[31:0] upper_immed;

	assign which_in2  = in_rs2_src == `RS2_IMMED;

	assign ALU_in1 = in_1;
	assign ALU_in2 = which_in2 ? in_itype_immed : in_2;

	assign upper_immed = {in_upper_immed, {12{1'b0}}};

	always @(*) begin
		case(in_alu_op)
			`ADD:        out_alu_result = $signed(ALU_in1) + $signed(ALU_in2);
			`SUB:        out_alu_result = $signed(ALU_in1) - $signed(ALU_in2);
			`SLLA:       out_alu_result = ALU_in1 << ALU_in2[4:0];
			`SLT:        out_alu_result = ($signed(ALU_in1) < $signed(ALU_in2)) ? 32'h1 : 32'h0;
			`SLTU:       out_alu_result = ALU_in1 < ALU_in2 ? 32'h1 : 32'h0;
			`XOR:        out_alu_result = ALU_in1 ^ ALU_in2;
			`SRL:        out_alu_result = ALU_in1 >> ALU_in2[4:0];
			`SRA:        out_alu_result = $signed(ALU_in1)  >>> ALU_in2[4:0];
			`OR:         out_alu_result = ALU_in1 | ALU_in2;
			`AND:        out_alu_result = ALU_in2 & ALU_in1;
			`SUBU:       out_alu_result = (ALU_in1 >= ALU_in2) ? 32'h0 : 32'hffffffff;
			`LUI_ALU:    out_alu_result = upper_immed;
			`AUIPC_ALU:  out_alu_result = $signed(in_curr_PC) + $signed(upper_immed);
			// TODO profitable to roll these exceptional cases into inst_delay to avoid pipeline when possible?
			`MUL:        out_alu_result = mul_result[31:0];
			`MULH:       out_alu_result = mul_result[63:32];
			`MULHSU:     out_alu_result = mul_result[63:32];
			`MULHU:      out_alu_result = mul_result[63:32];
			`DIV:        out_alu_result = (ALU_in2 == 0) ? 32'hffffffff : signed_div_result;
			`DIVU:       out_alu_result = (ALU_in2 == 0) ? 32'hffffffff : unsigned_div_result;
			`REM:        out_alu_result = (ALU_in2 == 0) ? ALU_in1 : signed_rem_result;
			`REMU:       out_alu_result = (ALU_in2 == 0) ? ALU_in1 : unsigned_rem_result;
			default: out_alu_result = 32'h0;
		endcase // in_alu_op
	end

`else

	wire which_in2;		
	wire[31:0] upper_immed;


	assign which_in2  = in_rs2_src == `RS2_IMMED;

	assign ALU_in1 = in_1;

	assign ALU_in2 = which_in2 ? in_itype_immed : in_2;


	assign upper_immed = {in_upper_immed, {12{1'b0}}};

	always @(*) begin
		case(in_alu_op)
			`ADD:        out_alu_result = $signed(ALU_in1) + $signed(ALU_in2);
			`SUB:        out_alu_result = $signed(ALU_in1) - $signed(ALU_in2);
			`SLLA:       out_alu_result = ALU_in1 << ALU_in2[4:0];
			`SLT:        out_alu_result = ($signed(ALU_in1) < $signed(ALU_in2)) ? 32'h1 : 32'h0;
			`SLTU:       out_alu_result = ALU_in1 < ALU_in2 ? 32'h1 : 32'h0;
			`XOR:        out_alu_result = ALU_in1 ^ ALU_in2;
			`SRL:        out_alu_result = ALU_in1 >> ALU_in2[4:0];
			`SRA:        out_alu_result = $signed(ALU_in1)  >>> ALU_in2[4:0];
			`OR:         out_alu_result = ALU_in1 | ALU_in2;
			`AND:        out_alu_result = ALU_in2 & ALU_in1;
			`SUBU:       out_alu_result = (ALU_in1 >= ALU_in2) ? 32'h0 : 32'hffffffff;
			`LUI_ALU:    out_alu_result = upper_immed;
			`AUIPC_ALU:  out_alu_result = $signed(in_curr_PC) + $signed(upper_immed);
			// TODO profitable to roll these exceptional cases into inst_delay to avoid pipeline when possible?
			`MUL:        out_alu_result = mul_result[31:0];
			`MULH:       out_alu_result = mul_result[63:32];
			`MULHSU:     out_alu_result = mul_result[63:32];
			`MULHU:      out_alu_result = mul_result[63:32];
			`DIV:        out_alu_result = (ALU_in2 == 0) ? 32'hffffffff : signed_div_result;
			`DIVU:       out_alu_result = (ALU_in2 == 0) ? 32'hffffffff : unsigned_div_result;
			`REM:        out_alu_result = (ALU_in2 == 0) ? ALU_in1 : signed_rem_result;
			`REMU:       out_alu_result = (ALU_in2 == 0) ? ALU_in1 : unsigned_rem_result;
			default: out_alu_result = 32'h0;
		endcase // in_alu_op
	end

`endif

endmodule : VX_alu