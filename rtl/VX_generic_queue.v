

module VX_generic_queue
	#(
		parameter DATAW = 4,
		parameter SIZE  = 16
	)
	(
	input  wire            clk,
	input  wire            reset,
	input  wire            push,
	input  wire[DATAW-1:0] in_data,

	input  wire            pop,
	output wire[DATAW-1:0] out_data,
	output wire            empty,
	output wire            full
);


	reg[SIZE-1:0]         data[DATAW-1:0];
	reg[$clog2(SIZE)-1:0] head;
	reg[$clog2(SIZE)-1:0] tail;

	assign empty = head == tail;
	assign full  = head == (tail+1);

	integer i;
	always @(posedge clk or reset) begin
		if (reset) begin
			head <= 0;
			tail <= 0;
			for (i = 0; i < SIZE; i=i+1) data[i] <= DATAW'0;
		end else begin
			if (push && !full) begin
				data[tail] <= in_data;
				tail        = tail+1;
			end

			if (pop) begin
				head = head + 1;
			end

		end
	end

	assign out_data = data[head];

endmodule