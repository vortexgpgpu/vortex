`include "../VX_define.v"

module VX_csr_data (
	input wire clk,    // Clock
	input wire reset,

	input wire[11:0]     in_read_csr_address,

	input wire           in_write_valid,
	input wire[31:0]     in_write_csr_data,
	input wire[11:0]     in_write_csr_address,

	output wire[31:0]    out_read_csr_data,

	// For instruction retire counting
	input wire           in_writeback_valid

);


	// wire[`NT_M1:0][31:0] thread_ids;
	// wire[`NT_M1:0][31:0] warp_ids;

	// genvar cur_t;
	// for (cur_t = 0; cur_t < `NT; cur_t = cur_t + 1) begin
	// 	assign thread_ids[cur_t] = cur_t;
	// end

	// genvar cur_tw;
	// for (cur_tw = 0; cur_tw < `NT; cur_tw = cur_tw + 1) begin
	// 	assign warp_ids[cur_tw] = {{(31-`NW_M1){1'b0}}, in_read_warp_num};
	// end

	reg[11:0] csr[1023:0];
	reg[63:0] cycle;
	reg[63:0] instret;


	wire read_cycle;
	wire read_cycleh;
	wire read_instret;
	wire read_instreth;

	assign read_cycle         = in_read_csr_address == 12'hC00;
	assign read_cycleh        = in_read_csr_address == 12'hC80;
	assign read_instret       = in_read_csr_address == 12'hC02;
	assign read_instreth      = in_read_csr_address == 12'hC82;

	// wire thread_select        = in_read_csr_address == 12'h20;
	// wire warp_select          = in_read_csr_address == 12'h21;

	// assign out_read_csr_data  =   thread_select ? thread_ids :
	// 					          warp_select   ? warp_ids   :
	// 					          0;

	integer curr_e;
	always @(posedge clk or posedge reset) begin
		if (reset) begin
			for (curr_e = 0; curr_e < 1024; curr_e=curr_e+1) begin
				csr[curr_e] <= 0;
			end
			cycle   <= 0;
			instret <= 0;
		end else begin
			cycle <= cycle + 1;
			if (in_write_valid) begin
				csr[in_write_csr_address] <= in_write_csr_data[11:0];
			end
			if (in_writeback_valid) begin
				instret <= instret + 1;
			end
		end
	end


    	assign out_read_csr_data = read_cycle ? cycle[31:0] :
    									read_cycleh ? cycle[63:32] : 
    											read_instret ? instret[31:0] :
    													read_instreth ? instret[63:32] : 
    															{{20{1'b0}}, csr[in_read_csr_address]};

endmodule
