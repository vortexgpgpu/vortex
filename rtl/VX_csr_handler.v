

module VX_csr_handler (
		input wire        clk,
		input wire[11:0]  in_decode_csr_address, // done
		/* verilator lint_off UNUSED */
		input wire[11:0]  in_mem_csr_address,
		/* verilator lint_on UNUSED */
		input wire        in_mem_is_csr,
		/* verilator lint_off UNUSED */
		input wire[31:0]  in_mem_csr_result,
		/* verilator lint_on UNUSED */
		input wire        in_wb_valid,
		output wire[31:0] out_decode_csr_data // done
	);


		reg[11:0] csr[1024:0];
		reg[63:0] cycle;
		reg[63:0] instret;
		reg[11:0] decode_csr_address;


		wire read_cycle;
		wire read_cycleh;
		wire read_instret;
		wire read_instreth;

		initial begin
			cycle              = 0;
			instret            = 0;
			decode_csr_address = 0;
		end


		always @(posedge clk) begin
			cycle              <= cycle   + 1;
			decode_csr_address <= in_decode_csr_address;
			if (in_wb_valid) begin
				instret <= instret + 1;
			end
		end


		always @(posedge clk) begin
			if(in_mem_is_csr) begin
				/* verilator lint_off WIDTH */
				csr[in_mem_csr_address] <= in_mem_csr_result[11:0];
				/* verilator lint_on WIDTH */
			end
		end


		assign read_cycle    = decode_csr_address == 12'hC00;
		assign read_cycleh   = decode_csr_address == 12'hC80;
		assign read_instret  = decode_csr_address == 12'hC02;
		assign read_instreth = decode_csr_address == 12'hC82;


		/* verilator lint_off WIDTH */
    	assign out_decode_csr_data = read_cycle ? cycle[31:0] :
    									read_cycleh ? cycle[63:32] : 
    											read_instret ? instret[31:0] :
    													read_instreth ? instret[63:32] : 
    															{{20{1'b0}}, csr[decode_csr_address]};
    	/* verilator lint_on WIDTH */





endmodule // VX_csr_handler







