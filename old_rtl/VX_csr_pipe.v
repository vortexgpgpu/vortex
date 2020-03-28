
module VX_csr_pipe (
	input wire clk,    // Clock
	input wire reset,
	input wire no_slot_csr,
	VX_csr_req_inter VX_csr_req,
	VX_wb_inter      VX_writeback,
	VX_csr_wb_inter  VX_csr_wb,
	output wire stall_gpr_csr
	
);

		wire[`NT_M1:0] valid_s2;
		wire[`NW_M1:0] warp_num_s2;
		wire[4:0]      rd_s2;
		wire[1:0]      wb_s2;
		wire[4:0]      alu_op_s2;
		wire           is_csr_s2;
		wire[11:0]     csr_address_s2;
		wire[31:0]     csr_read_data_s2;
		wire[31:0]     csr_updated_data_s2;

		wire[31:0] csr_read_data_unqual;
		wire[31:0] csr_read_data;

		assign stall_gpr_csr = no_slot_csr && VX_csr_req.is_csr && |(VX_csr_req.valid);

		assign csr_read_data = (csr_address_s2 == VX_csr_req.csr_address) ? csr_updated_data_s2 : csr_read_data_unqual;

		wire writeback          = |VX_writeback.wb_valid;
		VX_csr_data VX_csr_data(
			.clk                 (clk),
			.reset               (reset),
			.in_read_csr_address (VX_csr_req.csr_address),

			.in_write_valid      (is_csr_s2),
			.in_write_csr_data   (csr_updated_data_s2),
			.in_write_csr_address(csr_address_s2),

			.out_read_csr_data   (csr_read_data_unqual),

			.in_writeback_valid  (writeback)
			);



		reg[31:0] csr_updated_data;
		always @(*) begin
			case(VX_csr_req.alu_op)
				`CSR_ALU_RW: csr_updated_data = VX_csr_req.csr_mask;
				`CSR_ALU_RS: csr_updated_data = csr_read_data | VX_csr_req.csr_mask;
				`CSR_ALU_RC: csr_updated_data = csr_read_data & (32'hFFFFFFFF - VX_csr_req.csr_mask);
				default:     csr_updated_data = 32'hdeadbeef;
			endcase
		end	

		wire zero = 0;

		VX_generic_register #(.N(`NT + `NW_M1 + 1 + 5 + 2 + 5 + 12 + 64)) csr_reg_s2 (
			.clk  (clk),
			.reset(reset),
			.stall(no_slot_csr),
			.flush(zero),
			.in   ({VX_csr_req.valid, VX_csr_req.warp_num, VX_csr_req.rd, VX_csr_req.wb, VX_csr_req.is_csr, VX_csr_req.csr_address, csr_read_data   , csr_updated_data   }),
			.out  ({valid_s2        , warp_num_s2        , rd_s2        , wb_s2        , is_csr_s2        , csr_address_s2        , csr_read_data_s2, csr_updated_data_s2})
			);


		wire[`NT_M1:0][31:0] final_csr_data;

		wire[`NT_M1:0][31:0] thread_ids;
		wire[`NT_M1:0][31:0] warp_ids;
		wire[`NT_M1:0][31:0] csr_vec_read_data_s2;

		genvar cur_t;
		for (cur_t = 0; cur_t < `NT; cur_t = cur_t + 1) begin
			assign thread_ids[cur_t] = cur_t;
		end

		genvar cur_tw;
		for (cur_tw = 0; cur_tw < `NT; cur_tw = cur_tw + 1) begin
			assign warp_ids[cur_tw] = {{(31-`NW_M1){1'b0}}, warp_num_s2};
		end

		genvar cur_v;
		for (cur_v = 0; cur_v < `NT; cur_v = cur_v + 1) begin
			assign csr_vec_read_data_s2[cur_v] = csr_read_data_s2;
		end

		wire thread_select        = csr_address_s2 == 12'h20;
		wire warp_select          = csr_address_s2 == 12'h21;

		assign final_csr_data     =   thread_select ? thread_ids :
							          warp_select   ? warp_ids   :
							          csr_vec_read_data_s2;



	assign VX_csr_wb.valid      = valid_s2;
	assign VX_csr_wb.warp_num   = warp_num_s2;
	assign VX_csr_wb.rd         = rd_s2;
	assign VX_csr_wb.wb         = wb_s2;
	assign VX_csr_wb.csr_result = final_csr_data;

endmodule