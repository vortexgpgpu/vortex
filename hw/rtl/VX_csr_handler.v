module VX_csr_handler (
	input wire        clk,
	input wire[`CSR_ADDR_SIZE-1:0] in_decode_csr_address, // done
	VX_csr_write_request_inter vx_csr_w_req,
	input wire        in_wb_valid,
	output wire[31:0] out_decode_csr_data // done
);
	wire       in_mem_is_csr;
	wire[`CSR_ADDR_SIZE-1:0] in_mem_csr_address;
	wire[31:0] in_mem_csr_result;

	assign in_mem_is_csr      = vx_csr_w_req.is_csr;
	assign in_mem_csr_address = vx_csr_w_req.csr_address;
	assign in_mem_csr_result  = vx_csr_w_req.csr_result;

	reg [`CSR_WIDTH-1:0] csr [`NUM_CSRS-1:0];
	
	reg [63:0] cycle;
	reg [63:0] instret;	
	reg [`CSR_ADDR_SIZE-1:0] decode_csr_address;

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

	reg[`CSR_WIDTH-1:0] data_read;

	always @(posedge clk) begin
		if (in_mem_is_csr) begin
			csr[in_mem_csr_address] <= in_mem_csr_result[11:0];
		end
	end

	assign data_read = csr[decode_csr_address];

	assign read_cycle    = decode_csr_address == `CSR_CYCL_L;
	assign read_cycleh   = decode_csr_address == `CSR_CYCL_H;
	assign read_instret  = decode_csr_address == `CSR_INST_L;
	assign read_instreth = decode_csr_address == `CSR_INST_H;

	assign out_decode_csr_data = read_cycle    ? cycle[31:0] :
						  		 read_cycleh   ? cycle[63:32] : 
								 read_instret  ? instret[31:0] :
								 read_instreth ? instret[63:32] : 
								 				 {{20{1'b0}}, data_read};

endmodule // VX_csr_handler







