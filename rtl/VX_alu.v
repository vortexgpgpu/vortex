
`include "VX_define.v"

module VX_alu(
	input wire[31:0]  in_1,
	input wire[31:0]  in_2,
	input wire        in_rs2_src,
	input wire[31:0]  in_itype_immed,
	input wire[19:0]  in_upper_immed,
	input wire[4:0]   in_alu_op,
	input wire[31:0]  in_csr_data, // done
	input wire[31:0]  in_curr_PC,
	output reg[31:0]  out_alu_result
	);


		wire which_in2;

		wire[31:0] ALU_in1;
		wire[31:0] ALU_in2;
		wire[31:0] upper_immed;


		assign which_in2  = in_rs2_src == `RS2_IMMED;

		assign ALU_in1 = in_1;

		assign ALU_in2 = which_in2 ? in_itype_immed : in_2;


		assign upper_immed = {in_upper_immed, {12{1'b0}}};



		// always @(*) begin
		// 	$display("EXECUTE CURR_PC: %h",in_curr_PC);
		// end

		/* verilator lint_off UNUSED */
		wire[63:0] mult_unsigned_result  = ALU_in1 * ALU_in2;
		wire[63:0] mult_signed_result    = $signed(ALU_in1) * $signed(ALU_in2);

		wire[63:0] alu_in1_signed = {{32{ALU_in1[31]}}, ALU_in1};

		wire[63:0] mult_signed_un_result = alu_in1_signed * ALU_in2;
		/* verilator lint_on UNUSED */

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
				`CSR_ALU_RW: out_alu_result = in_csr_data;
				`CSR_ALU_RS: out_alu_result = in_csr_data;
				`CSR_ALU_RC: out_alu_result = in_csr_data;
				`MUL:        begin out_alu_result = mult_signed_result[31:0]; end
				`MULH:       out_alu_result = mult_signed_result[63:32];
				`MULHSU:     out_alu_result = mult_signed_un_result[63:32];
				`MULHU:      out_alu_result = mult_unsigned_result[63:32];
				`DIV:        out_alu_result = (ALU_in2 == 0) ? 32'hffffffff : $signed($signed(ALU_in1) / $signed(ALU_in2));
				`DIVU:       out_alu_result = (ALU_in2 == 0) ? 32'hffffffff : ALU_in1 / ALU_in2;
				`REM:        out_alu_result = (ALU_in2 == 0) ? ALU_in1 : $signed($signed(ALU_in1) % $signed(ALU_in2));
				`REMU:       out_alu_result = (ALU_in2 == 0) ? ALU_in1 : ALU_in1 % ALU_in2;
				default: out_alu_result = 32'h0;
			endcase // in_alu_op
		end


endmodule // VX_alu