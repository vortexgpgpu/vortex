module VX_scope #(     
	parameter DATAW = 64, 
	parameter BUSW  = 64, 
	parameter SIZE  = 1024
) ( 
	input wire clk,
	input wire reset,
	input wire start,
	input wire [DATAW-1:0] data_in,
	input wire [BUSW-1:0] bus_in,
	output wire [BUSW-1:0] bus_out,	
	input wire bus_write,
	input wire bus_read
);
	reg [DATAW-1:0] mem [SIZE-1:0];

	reg [`CLOG2(SIZE)-1:0] raddr, waddr;

	reg started, running, done;

	reg [BUSW-1:0] delay_cntr;

	reg data_valid, data_end;

	reg [`LOG2UP(DATAW)-1:0] read_offset;

	wire [BUSW-3:0] data_part;

	always @(posedge clk) begin
		if (reset) begin
			raddr      	<= 0;	
			waddr      	<= 0;	
			started    	<= 0;
			running    	<= 0;
			done       	<= 0;
			delay_cntr 	<= 0;
			read_offset	<= 0;
		end else begin

			if (bus_write) begin
				delay_cntr <= bus_in;
			end

			if (start) begin
				started <= 1;				
			end

			if (start || started) begin
				if (0 == delay_cntr) begin
					running <= 1;
				end else begin
					delay_cntr <= delay_cntr - 1;
				end
			end

			if (running && !done) begin
				mem[waddr] <= data_in;
				waddr <= waddr + 1;
				if (waddr == $bits(waddr)'(SIZE-1)) begin
					done <= 1;
				end
			end

			if (bus_read) begin
				if (DATAW > (BUSW-2)) begin
					if (read_offset < $bits(read_offset)'(DATAW-(BUSW-2))) begin
						read_offset <= read_offset + $bits(read_offset)'(BUSW-2);
					end else begin
						read_offset <= 0;
						raddr       <= raddr + 1;
					end
				end else begin
					raddr <= raddr + 1;
				end
			end
		end		
	end

	assign data_valid = (waddr != 0) && (raddr <= waddr);

	assign data_end   = (0 == read_offset) || (raddr == waddr);

	assign data_part = (BUSW-2)'(mem[raddr] >> read_offset);

	assign bus_out = {data_valid, data_end, data_part};

endmodule