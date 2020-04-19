`include "VX_define.vh"

module VX_csr_pipe #(
	parameter CORE_ID = 0
) (
	input wire		clk,
	input wire 		reset,
	input wire 		no_slot_csr,
	VX_csr_req_if 	csr_req_if,
	VX_wb_if      	writeback_if,
	VX_csr_wb_if  	csr_wb_if,
	output wire 	stall_gpr_csr	
);

	wire[`NUM_THREADS-1:0] valid_s2;
	wire[`NW_BITS-1:0] warp_num_s2;
	wire[4:0]      rd_s2;
	wire[1:0]      wb_s2;
	wire           is_csr_s2;
	wire[`CSR_ADDR_SIZE-1:0] csr_address_s2;
	wire[31:0]     csr_read_data_s2;
	wire[31:0]     csr_updated_data_s2;

	wire[31:0] csr_read_data_unqual;
	wire[31:0] csr_read_data;

	assign stall_gpr_csr = no_slot_csr && csr_req_if.is_csr && |(csr_req_if.valid);

	assign csr_read_data = (csr_address_s2 == csr_req_if.csr_address) ? csr_updated_data_s2 : csr_read_data_unqual;

	wire writeback = |writeback_if.wb_valid;
	
	VX_csr_data csr_data(
		.clk                 (clk),
		.reset               (reset),
		.in_read_csr_address (csr_req_if.csr_address),
		.in_write_valid      (is_csr_s2),
		.in_write_csr_data   (csr_updated_data_s2[`CSR_WIDTH-1:0]),
		.in_write_csr_address(csr_address_s2),
		.out_read_csr_data   (csr_read_data_unqual),
		.in_writeback_valid  (writeback)
	);

	reg [31:0] csr_updated_data;

	always @(*) begin
		case (csr_req_if.alu_op)
			`CSR_ALU_RW: csr_updated_data = csr_req_if.csr_mask;
			`CSR_ALU_RS: csr_updated_data = csr_read_data | csr_req_if.csr_mask;
			`CSR_ALU_RC: csr_updated_data = csr_read_data & (32'hFFFFFFFF - csr_req_if.csr_mask);
			default:     csr_updated_data = 32'hdeadbeef;
		endcase
	end	

	wire zero = 0;

	VX_generic_register #(
		.N(32 + 32 + 12 + 1 + 2 + 5 + (`NW_BITS-1+1) + `NUM_THREADS)
	) csr_reg_s2 (
		.clk  (clk),
		.reset(reset),
		.stall(no_slot_csr),
		.flush(zero),
		.in   ({csr_req_if.valid, csr_req_if.warp_num, csr_req_if.rd, csr_req_if.wb, csr_req_if.is_csr, csr_req_if.csr_address, csr_read_data   , csr_updated_data   }),
		.out  ({valid_s2        , warp_num_s2        , rd_s2        , wb_s2        , is_csr_s2        , csr_address_s2        , csr_read_data_s2, csr_updated_data_s2})
	);

	wire [`NUM_THREADS-1:0][31:0] final_csr_data;

	wire [`NUM_THREADS-1:0][31:0] thread_ids;
	wire [`NUM_THREADS-1:0][31:0] warp_ids;
	wire [`NUM_THREADS-1:0][31:0] warp_idz;
	wire [`NUM_THREADS-1:0][31:0] csr_vec_read_data_s2;

	genvar cur_t;
	for (cur_t = 0; cur_t < `NUM_THREADS; cur_t = cur_t + 1) begin
		assign thread_ids[cur_t] = cur_t;
	end

	genvar cur_tw;
	for (cur_tw = 0; cur_tw < `NUM_THREADS; cur_tw = cur_tw + 1) begin
		assign warp_ids[cur_tw] = 32'(warp_num_s2);
		assign warp_idz[cur_tw] = 32'(warp_num_s2) + (CORE_ID * `NUM_WARPS);
	end

	genvar cur_v;
	for (cur_v = 0; cur_v < `NUM_THREADS; cur_v = cur_v + 1) begin
		assign csr_vec_read_data_s2[cur_v] = csr_read_data_s2;
	end

	wire thread_select        = csr_address_s2 == 12'h20;
	wire warp_select          = csr_address_s2 == 12'h21;
	wire warp_id_select       = csr_address_s2 == 12'h22;

	assign final_csr_data     = thread_select  ? thread_ids :
								warp_select    ? warp_ids   :
								warp_id_select ? warp_idz   :
							    csr_vec_read_data_s2;

	assign csr_wb_if.valid      = valid_s2;
	assign csr_wb_if.warp_num   = warp_num_s2;
	assign csr_wb_if.rd         = rd_s2;
	assign csr_wb_if.wb         = wb_s2;
	assign csr_wb_if.csr_result = final_csr_data;

endmodule
