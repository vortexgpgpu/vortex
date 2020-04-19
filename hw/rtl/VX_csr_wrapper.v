
`include "VX_define.vh"

module VX_csr_wrapper (
	VX_csr_req_if csr_req_if,
	VX_csr_wb_if  csr_wb_if
);


	wire[`NUM_THREADS-1:0][31:0] thread_ids;
	wire[`NUM_THREADS-1:0][31:0] warp_ids;

	genvar cur_t, cur_tw;
	generate
	for (cur_t = 0; cur_t < `NUM_THREADS; cur_t = cur_t + 1) begin : thread_ids_init
		assign thread_ids[cur_t] = cur_t;
	end

	for (cur_tw = 0; cur_tw < `NUM_THREADS; cur_tw = cur_tw + 1) begin : warp_ids_init
		assign warp_ids[cur_tw] = {{(31-`NW_BITS-1){1'b0}}, csr_req_if.warp_num};
	end
	endgenerate


	assign csr_wb_if.valid    = csr_req_if.valid;
	assign csr_wb_if.warp_num = csr_req_if.warp_num;
	assign csr_wb_if.rd       = csr_req_if.rd;
	assign csr_wb_if.wb       = csr_req_if.wb;


	wire thread_select        = csr_req_if.csr_address == 12'h20;
	wire warp_select          = csr_req_if.csr_address == 12'h21;

	assign csr_wb_if.csr_result = thread_select ? thread_ids :
						          warp_select   ? warp_ids   :
						          0;

endmodule