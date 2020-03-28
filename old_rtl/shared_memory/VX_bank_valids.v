`include "../VX_define.v"

// Converts in_valids to bank_valids
module VX_bank_valids
	#(
		parameter NB            = 4,
		parameter BITS_PER_BANK = 3
	)
	(
	input wire[`NT_M1:0] in_valids,
	input wire[`NT_M1:0][31:0] in_addr,
	output reg[NB:0][`NT_M1:0] bank_valids
	);

	
	integer i, j;
	always@(*) begin
		for(j = 0; j <= NB; j = j+1 ) begin
			for(i = 0; i <= `NT_M1; i = i+1) begin
				if(in_valids[i]) begin
					if(in_addr[i][(2+BITS_PER_BANK-1):2] == j[BITS_PER_BANK-1:0]) begin
						bank_valids[j][i] = 1'b1;
					end
					else begin
						bank_valids[j][i] = 1'b0;
					end

				end
				else begin
					bank_valids[j][i] = 1'b0;
				end
			end
		end
	end

endmodule