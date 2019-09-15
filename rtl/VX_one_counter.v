

module VX_one_counter (
    input  wire[`NW-1:0] valids,
    output reg[`NW_M1:0] ones_found
  );

	integer i;
	always @(*) begin
		ones_found = 0;
		for (i = `NW-1; i >= 0; i = i - 1) begin
			if (valids[i]) begin
				ones_found = ones_found + 1;
			end
		end
	end
endmodule