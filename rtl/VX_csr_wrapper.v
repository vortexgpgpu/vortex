
`include "VX_define.v"

module VX_csr_wrapper (
	VX_csr_req_inter VX_csr_req,

	VX_csr_wb_inter  VX_csr_wb
);


	wire[`NT_M1:0][31:0] thread_ids;
	wire[`NT_M1:0][31:0] warp_ids;

	genvar cur_t;
	for (cur_t = 0; cur_t < `NT; cur_t = cur_t + 1) begin
		assign thread_ids[cur_t] = cur_t;
	end

	genvar cur_tw;
	for (cur_tw = 0; cur_tw < `NT; cur_tw = cur_tw + 1) begin
		assign warp_ids[cur_tw] = {{(31-`NW_M1){1'b0}}, VX_csr_req.warp_num};
	end


	assign VX_csr_wb.valid    = VX_csr_req.valid;
	assign VX_csr_wb.warp_num = VX_csr_req.warp_num;
	assign VX_csr_wb.rd       = VX_csr_req.rd;
	assign VX_csr_wb.wb       = VX_csr_req.wb;


	wire thread_select        = VX_csr_req.csr_address == 12'h20;
	wire warp_select          = VX_csr_req.csr_address == 12'h21;

	assign VX_csr_wb.csr_result = thread_select ? thread_ids :
						          warp_select   ? warp_ids   :
						          0;

endmodule