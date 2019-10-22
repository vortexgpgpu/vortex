`include "../VX_define.v"

module VX_set_bit (
	input  wire[1:0] index,
	output reg[`NT_M1:0] mask
);


integer some_index;
always @(*) begin
	for (some_index = 0; some_index <= `NT_M1; some_index = some_index + 1) begin
		if (some_index[1:0] == index) begin
			assign mask[some_index] = 0;
		end
		else begin
			assign mask[some_index] = 1;
		end
	end
end

endmodule