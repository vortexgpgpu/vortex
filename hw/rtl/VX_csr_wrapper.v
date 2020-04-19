
`include "VX_define.vh"

module VX_csr_wrapper (
	VX_csr_req_inter vx_csr_req,
	VX_csr_wb_inter  vx_csr_wb
);


	wire[`NUM_THREADS-1:0][31:0] thread_ids;
	wire[`NUM_THREADS-1:0][31:0] warp_ids;

	genvar cur_t, cur_tw;
	generate
	for (cur_t = 0; cur_t < `NUM_THREADS; cur_t = cur_t + 1) begin : thread_ids_init
		assign thread_ids[cur_t] = cur_t;
	end

	for (cur_tw = 0; cur_tw < `NUM_THREADS; cur_tw = cur_tw + 1) begin : warp_ids_init
		assign warp_ids[cur_tw] = {{(31-`NW_BITS-1){1'b0}}, vx_csr_req.warp_num};
	end
	endgenerate


	assign vx_csr_wb.valid    = vx_csr_req.valid;
	assign vx_csr_wb.warp_num = vx_csr_req.warp_num;
	assign vx_csr_wb.rd       = vx_csr_req.rd;
	assign vx_csr_wb.wb       = vx_csr_req.wb;


	wire thread_select        = vx_csr_req.csr_address == 12'h20;
	wire warp_select          = vx_csr_req.csr_address == 12'h21;

	assign vx_csr_wb.csr_result = thread_select ? thread_ids :
						          warp_select   ? warp_ids   :
						          0;

endmodule