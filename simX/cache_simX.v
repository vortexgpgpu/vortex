`include "VX_define.v"

module cache_simX (
	input       wire clk,    // Clock
	input       wire reset,

	// Icache
	input  wire[31:0] icache_pc_addr,
	input  wire       icache_valid_pc_addr,
	output wire       icache_stall,

	// Dcache
	input  wire[2:0]  dcache_mem_read,
	input  wire[2:0]  dcache_mem_write,
	input  wire       dcache_in_valid[`NT_M1:0],
	input  wire[31:0] dcache_in_addr[`NT_M1:0],
	output wire       dcache_stall	
);
	//////////////////// ICACHE ///////////////////

	VX_icache_request_inter  VX_icache_req;
	assign VX_icache_req.pc_address                    = icache_pc_addr;
	assign VX_icache_req.out_cache_driver_in_mem_read  = (icache_valid_pc_addr) ? `LW_MEM_READ : `NO_MEM_READ;
	assign VX_icache_req.out_cache_driver_in_mem_write = `NO_MEM_WRITE;
	assign VX_icache_req.out_cache_driver_in_valid     = icache_valid_pc_addr;
	assign VX_icache_req.out_cache_driver_in_data      = 0;

	VX_icache_response_inter VX_icache_rsp;
	assign icache_stall = VX_icache_rsp.delay;

	VX_dram_req_rsp_inter #(
		
		.NUMBER_BANKS(`ICACHE_BANKS),
		.NUM_WORDS_PER_BLOCK(`ICACHE_NUM_WORDS_PER_BLOCK)

	) VX_dram_req_rsp_icache();

	reg  icache_i_m_ready;

	assign VX_dram_req_rsp_icache.i_m_ready = icache_i_m_ready;

	//////////////////// DCACHE ///////////////////

	VX_dcache_request_inter  VX_dcache_req;
	assign VX_dcache_req.out_cache_driver_in_mem_read  = dcache_mem_read;
	assign VX_dcache_req.out_cache_driver_in_mem_write = dcache_mem_write;
	assign VX_dcache_req.out_cache_driver_in_data      = 0;

	genvar curr_t;
	for (curr_t = 0; curr_t < `NT; curr_t=curr_t+1)
	begin
		assign VX_dcache_req.out_cache_driver_in_address[curr_t]   = dcache_in_addr[curr_t];
		assign VX_dcache_req.out_cache_driver_in_valid[curr_t]     = dcache_in_valid[curr_t];
	end

	VX_dcache_response_inter VX_dcache_rsp;
	assign dcache_stall = VX_dcache_rsp.delay;

	VX_dram_req_rsp_inter #(

		.NUMBER_BANKS(`DCACHE_BANKS),
		.NUM_WORDS_PER_BLOCK(`DCACHE_NUM_WORDS_PER_BLOCK)

		) VX_dram_req_rsp();

	reg  dcache_i_m_ready;
	assign VX_dram_req_rsp.i_m_ready = dcache_i_m_ready;

	VX_dmem_controller dmem_ctrl (
		.clk                   (clk),
		.reset                 (reset),
		.VX_dram_req_rsp       (VX_dram_req_rsp),
		.VX_dram_req_rsp_icache(VX_dram_req_rsp_icache),
		.VX_icache_req         (VX_icache_req),
		.VX_icache_rsp         (VX_icache_rsp),
		.VX_dcache_req         (VX_dcache_req),
		.VX_dcache_rsp         (VX_dcache_rsp)
		);

	always @(posedge clk, posedge reset) begin
		if (reset)
		begin
			icache_i_m_ready = 0;
			dcache_i_m_ready = 0;
		end else begin

			if (VX_dram_req_rsp_icache.o_m_valid) begin
				icache_i_m_ready = 1;
				// $display("cache_simX.v: setting icache_i_m_ready = %d", icache_i_m_ready);
			end else if (icache_i_m_ready) begin
				icache_i_m_ready = 0;
			end else begin
				icache_i_m_ready = 0;
			end


			if (VX_dram_req_rsp.o_m_valid) begin
				dcache_i_m_ready = 1;
			end else if (dcache_i_m_ready) begin
				dcache_i_m_ready = 0;
			end else begin
				dcache_i_m_ready = 0;
			end

		end
	end

endmodule





