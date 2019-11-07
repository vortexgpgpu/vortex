
`include "VX_define.v"

module VX_dmem_controller (
	input wire               clk,
	input wire               reset,
	// MEM-RAM
	VX_dram_req_rsp_inter    VX_dram_req_rsp,
	// MEM-Processor
	VX_dcache_request_inter  VX_dcache_req,
	VX_dcache_response_inter VX_dcache_rsp
);


	wire to_shm = VX_dcache_req.out_cache_driver_in_address[0][31:24] == 8'hFF;

	wire[`NT_M1:0]       sm_driver_in_valid     = VX_dcache_req.out_cache_driver_in_valid & {`NT{to_shm}};
	wire[`NT_M1:0]       cache_driver_in_valid  = VX_dcache_req.out_cache_driver_in_valid & {`NT{~to_shm}};
	
	wire read_or_write   = (VX_dcache_req.out_cache_driver_in_mem_write != `NO_MEM_WRITE) && (|cache_driver_in_valid);



	wire[`NT_M1:0][31:0] cache_driver_in_address   = VX_dcache_req.out_cache_driver_in_address;
	wire[2:0]            cache_driver_in_mem_read  = !(|cache_driver_in_valid) ? `NO_MEM_READ  : VX_dcache_req.out_cache_driver_in_mem_read;
	wire[2:0]            cache_driver_in_mem_write = !(|cache_driver_in_valid) ? `NO_MEM_WRITE : VX_dcache_req.out_cache_driver_in_mem_write;
	wire[`NT_M1:0][31:0] cache_driver_in_data      = VX_dcache_req.out_cache_driver_in_data;


	wire[`NT_M1:0][31:0] cache_driver_out_data;
	wire[`NT_M1:0][31:0] sm_driver_out_data;
	wire[`NT_M1:0]       cache_driver_out_valid; // Not used for now
	wire                 sm_delay;
	wire                 cache_delay;


	wire valid_read_cache = !cache_delay && cache_driver_in_valid[0];


	VX_shared_memory #(.NB(7), .BITS_PER_BANK(3)) shared_memory (
		.clk       (clk),
		.reset     (reset),
		.in_valid  (sm_driver_in_valid),
		.in_address(cache_driver_in_address),
		.in_data   (cache_driver_in_data),
		.mem_read  (cache_driver_in_mem_read),
		.mem_write (cache_driver_in_mem_write),
		.out_valid (cache_driver_out_valid),
		.out_data  (sm_driver_out_data),
		.stall     (sm_delay)
		);


	VX_d_cache#(
          .CACHE_SIZE          (`DCACHE_SIZE),
          .CACHE_WAYS          (`DCACHE_WAYS),
          .CACHE_BLOCK         (`DCACHE_BLOCK),
          .CACHE_BANKS         (`DCACHE_BANKS),
          .LOG_NUM_BANKS       (`DCACHE_LOG_NUM_BANKS),
          .NUM_REQ             (`DCACHE_NUM_REQ),
          .LOG_NUM_REQ         (`DCACHE_LOG_NUM_REQ),
          .NUM_IND			   (`DCACHE_NUM_IND),
          .CACHE_WAY_INDEX     (`DCACHE_WAY_INDEX),
          .NUM_WORDS_PER_BLOCK (`DCACHE_NUM_WORDS_PER_BLOCK),
          .OFFSET_SIZE_START   (`DCACHE_OFFSET_ST),
          .OFFSET_SIZE_END     (`DCACHE_OFFSET_ED),
          .TAG_SIZE_START      (`DCACHE_TAG_SIZE_START),
          .TAG_SIZE_END        (`DCACHE_TAG_SIZE_END),
          .IND_SIZE_START      (`DCACHE_IND_SIZE_START),
          .IND_SIZE_END        (`DCACHE_IND_SIZE_END),
          .ADDR_TAG_START      (`DCACHE_ADDR_TAG_START),
          .ADDR_TAG_END        (`DCACHE_ADDR_TAG_END),
          .ADDR_OFFSET_START   (`DCACHE_ADDR_OFFSET_ST),
          .ADDR_OFFSET_END     (`DCACHE_ADDR_OFFSET_ED),
          .ADDR_IND_START      (`DCACHE_IND_ST),
          .ADDR_IND_END        (`DCACHE_IND_ED),
          .MEM_ADDR_REQ_MASK   (`DCACHE_MEM_REQ_ADDR_MASK)
    )
	 dcache
	(
		.clk                (clk),
		.rst                (reset),
		.i_p_valid          (cache_driver_in_valid), 
		.i_p_addr           (cache_driver_in_address),
		.i_p_writedata      (cache_driver_in_data),
		.i_p_read_or_write  (read_or_write),
		.i_p_mem_read       (cache_driver_in_mem_read),
		.i_p_mem_write      (cache_driver_in_mem_write),
		.o_p_readdata       (cache_driver_out_data),
		.o_p_delay          (cache_delay),
		.o_m_evict_addr     (VX_dram_req_rsp.o_m_evict_addr),
		.o_m_read_addr      (VX_dram_req_rsp.o_m_read_addr),
		.o_m_valid          (VX_dram_req_rsp.o_m_valid),
		.o_m_writedata      (VX_dram_req_rsp.o_m_writedata),
		.o_m_read_or_write  (VX_dram_req_rsp.o_m_read_or_write),
		.i_m_readdata       (VX_dram_req_rsp.i_m_readdata),
		.i_m_ready          (VX_dram_req_rsp.i_m_ready)
		);


	assign VX_dcache_rsp.in_cache_driver_out_data = to_shm ? sm_driver_out_data : cache_driver_out_data;
	assign VX_dcache_rsp.delay                    = sm_delay || cache_delay;


endmodule