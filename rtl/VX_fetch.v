
`include "VX_define.v"

module VX_fetch (
	input  wire       clk,
	input  wire       reset,
	input  wire       in_branch_dir,
	input  wire       in_freeze,
	input  wire[31:0] in_branch_dest,
	input  wire       in_branch_stall,
	input  wire       in_fwd_stall,
	input  wire       in_branch_stall_exe,
	input  wire       in_jal,
	input  wire[31:0] in_jal_dest,
	input  wire       in_interrupt,
	input  wire       in_debug,
	input  wire[31:0] in_instruction,

	output wire[31:0]      out_instruction,
	output wire            out_delay,
	output wire[31:0]      out_curr_PC,
	output wire            out_valid[`NT_M1:0]
);


		reg       stall_reg;
		reg       delay_reg;
		reg[31:0] old;
		reg[4:0]  state;
		reg[31:0] real_PC;
		reg[31:0] JAL_reg;
		reg[31:0] BR_reg;
		reg       prev_debug;


		reg       delay;
		reg[31:0] PC_to_use;
		reg[31:0]  PC_to_use_temp;
		reg       stall;
		reg[31:0] temp_PC;
		reg[31:0] out_PC;
		reg[4:0]  temp_state;
		reg[4:0]  tempp_state;

		reg[`NT_M1:0] valid;


		// integer ini_cur_th = 0;
		genvar out_cur_th;

		initial begin
			// for (ini_cur_th = 0; ini_cur_th < `NT; ini_cur_th=ini_cur_th+1)
			// 	valid[ini_cur_th]   = 1; // Thread 1 active
			valid[0]   = 1;
			// valid[1]   = 0;
			stall_reg  = 0;
			delay_reg  = 0;
			old        = 0;
			state      = 0;
			real_PC    = 0;
			JAL_reg    = 0;
			BR_reg     = 0;
			prev_debug = 0;
		end

		always @(*) begin
			case(state)
				5'h00: PC_to_use_temp = real_PC;
				5'h01: PC_to_use_temp = JAL_reg;
				5'h02: PC_to_use_temp = BR_reg;
				5'h03: PC_to_use_temp = real_PC;
				5'h04: PC_to_use_temp = old;
				default: PC_to_use_temp = 32'h0;
			endcase // state
		end



		assign out_delay = 0;
		assign delay     = out_delay;

		always @(*) begin
			if ((delay_reg == 1'b1) && (in_freeze == 1'b0)) begin
				// $display("Using old cuz delay: PC: %h",old);
				PC_to_use = old;
			end else if (in_debug == 1'b1) begin
				if (prev_debug == 1'b1) begin
					PC_to_use = old;
				end else begin
					PC_to_use = real_PC;
				end
			end else if (stall_reg == 1'b1) begin
				// $display("Using old cuz stall: PC: %h\treal_pc: %h",old, real_PC);
				PC_to_use = old;
			end else begin
				PC_to_use = PC_to_use_temp;
			end
		end
		
		assign stall = in_branch_stall || in_fwd_stall || in_branch_stall_exe || in_interrupt || delay || in_freeze;

		assign out_instruction = stall ? 32'b0 : in_instruction;

		generate
			for (out_cur_th = 0; out_cur_th < `NT; out_cur_th = out_cur_th+1)
				assign out_valid[out_cur_th] = stall ? 1'b0  : valid[out_cur_th];
		endgenerate


		always @(*) begin

			if ((in_jal == 1'b1) && (delay_reg == 1'b0)) begin
				temp_PC = in_jal_dest;
			end else if ((in_branch_dir == 1'b1) && (delay_reg == 1'b0)) begin
				temp_PC = in_branch_dest;
			end else begin
				temp_PC = PC_to_use;
			end
		
		end

		assign out_PC = temp_PC;

		always @(*) begin
		
			if (in_jal == 1'b1) begin
				temp_state = 5'h1;
			end else if (in_branch_dir == 1'b1) begin
				temp_state = 5'h2;
			end else begin
				temp_state = 5'h0;
			end
		end





		assign tempp_state = in_interrupt ? 5'h3 : temp_state;

		assign out_curr_PC = out_PC;
		

		always @(posedge clk or posedge reset) begin
			if(reset) begin
				state      <= 0;
				stall_reg  <= 0;
				delay_reg  <= 0;
				old        <= 0;
				real_PC    <= 0;
				JAL_reg    <= 0;
				BR_reg     <= 0;
				prev_debug <= 0;

			end else begin

				if (in_debug == 1'b1) begin
					state <= 5'h3;
				end else begin
					if (prev_debug == 1'b1) begin
						state <= 5'h4;
					end else begin
						state <= tempp_state;
					end
				end

				stall_reg  <= stall;
				delay_reg  <= delay || in_freeze;
				old        <= out_PC;
				real_PC    <= PC_to_use       + 32'h4;
				JAL_reg    <= in_jal_dest     + 32'h4;
				BR_reg     <= in_branch_dest  + 32'h4;
				prev_debug <= in_debug;

			end
		end


		// always @(*) begin
		// 	$display("Fetch out pc: %h", out_PC);
		// end




endmodule