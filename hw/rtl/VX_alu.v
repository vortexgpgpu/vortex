`include "VX_define.v"

module VX_alu (
	input wire        clk,
	input wire        reset,
	input wire[31:0]  a_i,
	input wire[31:0]  b_i,
	input wire        rs2_src_i,
	input wire[31:0]  itype_immed_i,
	input wire[19:0]  upper_immed_i,
	input wire[4:0]   alu_op_i,
	input wire[31:0]  curr_PC_i,
	output reg[31:0]  alu_result_o,
	output reg        alu_stall_o
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
	assign mul_data_a = (alu_op_i == `MULHU) ? {32'b0, ALU_in1} : alu_in1_signed;
	assign mul_data_b = (alu_op_i == `MULHU || alu_op_i == `MULHSU) ? {32'b0, ALU_in2} : alu_in2_signed;

	reg [15:0] curr_inst_delay;
	reg [15:0] inst_delay;
	reg inst_was_stalling;

	wire inst_delay_stall = inst_was_stalling ? inst_delay != 0 : curr_inst_delay != 0;
	assign alu_stall_o = inst_delay_stall;

	always @(*) begin
		case(alu_op_i)
			`DIV,
			`DIVU,
			`REM,
			`REMU: curr_inst_delay = div_pipeline_len;
			`MUL,
			`MULH,
			`MULHSU,
			`MULHU: curr_inst_delay = mul_pipeline_len;
			default: curr_inst_delay = 0;
		endcase // alu_op_i
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

	assign which_in2  = rs2_src_i == `RS2_IMMED;

	assign ALU_in1 = a_i;
	assign ALU_in2 = which_in2 ? itype_immed_i : b_i;

	assign upper_immed = {upper_immed_i, {12{1'b0}}};

	always @(*) begin
		case(alu_op_i)
			`ADD:        alu_result_o = $signed(ALU_in1) + $signed(ALU_in2);
			`SUB:        alu_result_o = $signed(ALU_in1) - $signed(ALU_in2);
			`SLLA:       alu_result_o = ALU_in1 << ALU_in2[4:0];
			`SLT:        alu_result_o = ($signed(ALU_in1) < $signed(ALU_in2)) ? 32'h1 : 32'h0;
			`SLTU:       alu_result_o = ALU_in1 < ALU_in2 ? 32'h1 : 32'h0;
			`XOR:        alu_result_o = ALU_in1 ^ ALU_in2;
			`SRL:        alu_result_o = ALU_in1 >> ALU_in2[4:0];
			`SRA:        alu_result_o = $signed(ALU_in1)  >>> ALU_in2[4:0];
			`OR:         alu_result_o = ALU_in1 | ALU_in2;
			`AND:        alu_result_o = ALU_in2 & ALU_in1;
			`SUBU:       alu_result_o = (ALU_in1 >= ALU_in2) ? 32'h0 : 32'hffffffff;
			`LUI_ALU:    alu_result_o = upper_immed;
			`AUIPC_ALU:  alu_result_o = $signed(curr_PC_i) + $signed(upper_immed);
			// TODO profitable to roll these exceptional cases into inst_delay to avoid pipeline when possible?
			`MUL:        alu_result_o = mul_result[31:0];
			`MULH:       alu_result_o = mul_result[63:32];
			`MULHSU:     alu_result_o = mul_result[63:32];
			`MULHU:      alu_result_o = mul_result[63:32];
			`DIV:        alu_result_o = (ALU_in2 == 0) ? 32'hffffffff : signed_div_result;
			`DIVU:       alu_result_o = (ALU_in2 == 0) ? 32'hffffffff : unsigned_div_result;
			`REM:        alu_result_o = (ALU_in2 == 0) ? ALU_in1 : signed_rem_result;
			`REMU:       alu_result_o = (ALU_in2 == 0) ? ALU_in1 : unsigned_rem_result;
			default: alu_result_o = 32'h0;
		endcase // alu_op_i
	end

`else

	wire which_in2;		
	wire[31:0] upper_immed;

	assign which_in2  = rs2_src_i == `RS2_IMMED;

	assign ALU_in1 = a_i;

	assign ALU_in2 = which_in2 ? itype_immed_i : b_i;

	assign upper_immed = {upper_immed_i, {12{1'b0}}};

	always @(*) begin
		case(alu_op_i)
			`ADD:        alu_result_o = $signed(ALU_in1) + $signed(ALU_in2);
			`SUB:        alu_result_o = $signed(ALU_in1) - $signed(ALU_in2);
			`SLLA:       alu_result_o = ALU_in1 << ALU_in2[4:0];
			`SLT:        alu_result_o = ($signed(ALU_in1) < $signed(ALU_in2)) ? 32'h1 : 32'h0;
			`SLTU:       alu_result_o = ALU_in1 < ALU_in2 ? 32'h1 : 32'h0;
			`XOR:        alu_result_o = ALU_in1 ^ ALU_in2;
			`SRL:        alu_result_o = ALU_in1 >> ALU_in2[4:0];
			`SRA:        alu_result_o = $signed(ALU_in1)  >>> ALU_in2[4:0];
			`OR:         alu_result_o = ALU_in1 | ALU_in2;
			`AND:        alu_result_o = ALU_in2 & ALU_in1;
			`SUBU:       alu_result_o = (ALU_in1 >= ALU_in2) ? 32'h0 : 32'hffffffff;
			`LUI_ALU:    alu_result_o = upper_immed;
			`AUIPC_ALU:  alu_result_o = $signed(curr_PC_i) + $signed(upper_immed);
			// TODO profitable to roll these exceptional cases into inst_delay to avoid pipeline when possible?
			`MUL:        alu_result_o = mul_result[31:0];
			`MULH:       alu_result_o = mul_result[63:32];
			`MULHSU:     alu_result_o = mul_result[63:32];
			`MULHU:      alu_result_o = mul_result[63:32];
			`DIV:        alu_result_o = (ALU_in2 == 0) ? 32'hffffffff : signed_div_result;
			`DIVU:       alu_result_o = (ALU_in2 == 0) ? 32'hffffffff : unsigned_div_result;
			`REM:        alu_result_o = (ALU_in2 == 0) ? ALU_in1 : signed_rem_result;
			`REMU:       alu_result_o = (ALU_in2 == 0) ? ALU_in1 : unsigned_rem_result;
			default: alu_result_o = 32'h0;
		endcase // alu_op_i
	end

`endif

endmodule : VX_alu