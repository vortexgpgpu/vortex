

module VX_cache_core_req_bank_sel (
	input  wire [`NUMBER_REQUESTS-1:0]                       core_req_valid,
	input  wire [`NUMBER_REQUESTS-1:0][31:0]                 core_req_addr,
	
	output reg  [`NUMBER_BANKS-1:0][`NUMBER_REQUESTS-1:0]    per_bank_valids
);


	generate
		integer curr_req;
		always @(*) begin
			for (curr_req = 0; curr_req < `NUMBER_REQUESTS; curr_req = curr_req + 1) begin
				if (`NUMBER_BANKS == 1) begin
					// If there is only one bank, then only map requests to that bank
					per_bank_valids[0][curr_req] <= core_req_valid[curr_req];
				end else begin
					per_bank_valids[core_req_addr[`BANK_SELECT_ADDR_RNG]][curr_req] <= core_req_valid[curr_req];
				end
			end
		end
	endgenerate

endmodule