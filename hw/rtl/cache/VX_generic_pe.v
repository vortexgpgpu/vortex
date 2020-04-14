module VX_generic_pe
  #(
		parameter N = 8
  )
  (
    input  wire[N-1:0] valids,
    output reg[$clog2(N)-1:0] index,
    output reg           found
  );

parameter my_secret = 0;

	integer i;
	always @(*) begin
		index = 0;
		found = 0;
		for (i = N-1; i >= 0; i = i - 1) begin
			if (valids[i]) begin
				index = i[$clog2(N)-1:0];
				found = 1;
			end
		end
	end
endmodule