`include "VX_define.v"

module VX_dmem_controller (
	input wire               clk,
	input wire               reset,
	// MEM-RAM
	VX_gpu_dcache_dram_req_inter VX_gpu_dcache_dram_req,
	VX_gpu_dcache_dram_res_inter VX_gpu_dcache_dram_res,

	VX_dram_req_rsp_inter    VX_dram_req_rsp_icache,
	// MEM-Processor
	VX_icache_request_inter  VX_icache_req,
	VX_icache_response_inter VX_icache_rsp,
	VX_gpu_dcache_req_inter  VX_dcache_req,
	VX_gpu_dcache_res_inter  VX_dcache_rsp
);

	wire to_shm = VX_dcache_req.core_req_addr[0][31:24] == 8'hFF;

	wire[`NT_M1:0]       cache_driver_in_valid  = VX_dcache_req.core_req_valid & {`NT{~to_shm}};

	wire[`NT_M1:0]       sm_driver_in_valid     = VX_dcache_req.core_req_valid & {`NT{to_shm}};
	wire[2:0]            sm_driver_in_mem_read  = !(|sm_driver_in_valid) ? `NO_MEM_READ   : VX_dcache_req.core_req_mem_read;
	wire[2:0]            sm_driver_in_mem_write = !(|sm_driver_in_valid) ? `NO_MEM_WRITE  : VX_dcache_req.core_req_mem_write;

	wire[`NT_M1:0][31:0] cache_driver_out_data;
	wire[`NT_M1:0][31:0] sm_driver_out_data;
	wire[`NT_M1:0]       cache_driver_out_valid; // Not used for now
	wire                 sm_delay;

	// I_Cache Signals

 	wire[31:0] 			 icache_instruction_out;
 	wire                 icache_delay;
 	wire                 icache_driver_in_valid     = VX_icache_req.out_cache_driver_in_valid;
 	wire[31:0]           icache_driver_in_address   = VX_icache_req.pc_address;
 	wire[2:0]            icache_driver_in_mem_read  = !(|icache_driver_in_valid) ? `NO_MEM_READ  : VX_icache_req.out_cache_driver_in_mem_read;
 	wire[2:0]            icache_driver_in_mem_write = !(|icache_driver_in_valid) ? `NO_MEM_WRITE : VX_icache_req.out_cache_driver_in_mem_write;
 	wire[31:0]           icache_driver_in_data      = VX_icache_req.out_cache_driver_in_data;
	wire                 read_or_write_ic           = (VX_icache_req.out_cache_driver_in_mem_write != `NO_MEM_WRITE) && (|icache_driver_in_valid);


	VX_shared_memory #(
		.SM_SIZE                (`SHARED_MEMORY_SIZE),
	  .SM_BANKS 				      (`SHARED_MEMORY_BANKS),
	  .SM_BYTES_PER_READ      (`SHARED_MEMORY_BYTES_PER_READ),
	  .SM_WORDS_PER_READ      (`SHARED_MEMORY_WORDS_PER_READ),
	  .SM_LOG_WORDS_PER_READ  (`SHARED_MEMORY_LOG_WORDS_PER_READ),
	  .SM_BANK_OFFSET_START   (`SHARED_MEMORY_BANK_OFFSET_ST),
	  .SM_BANK_OFFSET_END     (`SHARED_MEMORY_BANK_OFFSET_ED),
	  .SM_BLOCK_OFFSET_START  (`SHARED_MEMORY_BLOCK_OFFSET_ST),
	  .SM_BLOCK_OFFSET_END    (`SHARED_MEMORY_BLOCK_OFFSET_ED),
	  .SM_INDEX_START         (`SHARED_MEMORY_INDEX_OFFSET_ST),
	  .SM_INDEX_END           (`SHARED_MEMORY_INDEX_OFFSET_ED),
	  .SM_HEIGHT 				      (`SHARED_MEMORY_HEIGHT),
	  .NUM_REQ                (`SHARED_MEMORY_NUM_REQ),
	  .BITS_PER_BANK 			    (`SHARED_MEMORY_BITS_PER_BANK) 
		)
		shared_memory
		(
		.clk       (clk),
		.reset     (reset),
		.in_valid  (sm_driver_in_valid),
		.in_address(VX_dcache_req.core_req_addr),
		.in_data   (VX_dcache_req.core_req_writedata),
		.mem_read  (sm_driver_in_mem_read),
		.mem_write (sm_driver_in_mem_write),
		.out_valid (cache_driver_out_valid),
		.out_data  (sm_driver_out_data),
		.stall     (sm_delay)
		);


	VX_cache gpu_dcache(
		.clk               (clk),
		.reset             (reset),

		// Core req
		.core_req_valid    (cache_driver_in_valid),
		.core_req_addr     (VX_dcache_req.core_req_addr),
		.core_req_writedata(VX_dcache_req.core_req_writedata),
		.core_req_mem_read (VX_dcache_req.core_req_mem_read),
		.core_req_mem_write(VX_dcache_req.core_req_mem_write),
		.core_req_rd       (VX_dcache_req.core_req_rd),
		.core_req_wb       (VX_dcache_req.core_req_wb),
		.core_req_warp_num (VX_dcache_req.core_req_warp_num),

		// Delay Core Req
		.delay_req         (VX_dcache_rsp.delay_req),

		// Core Cache Can't WB
		.core_no_wb_slot   (VX_dcache_req.core_no_wb_slot),

		// Cache CWB
		.core_wb_valid     (VX_dcache_rsp.core_wb_valid),
		.core_wb_req_rd    (VX_dcache_rsp.core_wb_req_rd),
		.core_wb_req_wb    (VX_dcache_rsp.core_wb_req_wb),
		.core_wb_warp_num  (VX_dcache_rsp.core_wb_warp_num),
		.core_wb_readdata  (VX_dcache_rsp.core_wb_readdata),

		// DRAM response
		.dram_fill_rsp     (VX_gpu_dcache_dram_res.dram_fill_rsp),
		.dram_fill_rsp_addr(VX_gpu_dcache_dram_res.dram_fill_rsp_addr),
		.dram_fill_rsp_data(VX_gpu_dcache_dram_res.dram_fill_rsp_data),

		// DRAM accept response
		.dram_fill_accept  (VX_gpu_dcache_dram_req.dram_fill_accept),

		// DRAM Req
		.dram_req          (VX_gpu_dcache_dram_req.dram_req),
		.dram_req_write    (VX_gpu_dcache_dram_req.dram_req_write),
		.dram_req_read     (VX_gpu_dcache_dram_req.dram_req_read),
		.dram_req_addr     (VX_gpu_dcache_dram_req.dram_req_addr),
		.dram_req_size     (VX_gpu_dcache_dram_req.dram_req_size),
		.dram_req_data     (VX_gpu_dcache_dram_req.dram_req_data),

		// Snoop Response
		.dram_req_because_of_wb(VX_gpu_dcache_dram_req.dram_because_of_snp),
		.dram_snp_full         (VX_gpu_dcache_dram_req.dram_snp_full),

		// Snoop Request
		.snp_req               (0),
		.snp_req_addr          (0)
		);



	// VX_d_cache#(
 //          .CACHE_SIZE          (`DCACHE_SIZE),
 //          .CACHE_WAYS          (`DCACHE_WAYS),
 //          .CACHE_BLOCK         (`DCACHE_BLOCK),
 //          .CACHE_BANKS         (`DCACHE_BANKS),
 //          .LOG_NUM_BANKS       (`DCACHE_LOG_NUM_BANKS),
 //          .NUM_REQ             (`DCACHE_NUM_REQ),
 //          .LOG_NUM_REQ         (`DCACHE_LOG_NUM_REQ),
 //          .NUM_IND			   (`DCACHE_NUM_IND),
 //          .CACHE_WAY_INDEX     (`DCACHE_WAY_INDEX),
 //          .NUM_WORDS_PER_BLOCK (`DCACHE_NUM_WORDS_PER_BLOCK),
 //          .OFFSET_SIZE_START   (`DCACHE_OFFSET_ST),
 //          .OFFSET_SIZE_END     (`DCACHE_OFFSET_ED),
 //          .TAG_SIZE_START      (`DCACHE_TAG_SIZE_START),
 //          .TAG_SIZE_END        (`DCACHE_TAG_SIZE_END),
 //          .IND_SIZE_START      (`DCACHE_IND_SIZE_START),
 //          .IND_SIZE_END        (`DCACHE_IND_SIZE_END),
 //          .ADDR_TAG_START      (`DCACHE_ADDR_TAG_START),
 //          .ADDR_TAG_END        (`DCACHE_ADDR_TAG_END),
 //          .ADDR_OFFSET_START   (`DCACHE_ADDR_OFFSET_ST),
 //          .ADDR_OFFSET_END     (`DCACHE_ADDR_OFFSET_ED),
 //          .ADDR_IND_START      (`DCACHE_IND_ST),
 //          .ADDR_IND_END        (`DCACHE_IND_ED),
 //          .MEM_ADDR_REQ_MASK   (`DCACHE_MEM_REQ_ADDR_MASK)
 //    )
	//  dcache
	// (
	// 	.clk                (clk),
	// 	.rst                (reset),
	// 	.i_p_valid          (cache_driver_in_valid),
	// 	.i_p_addr           (cache_driver_in_address),
	// 	.i_p_writedata      (cache_driver_in_data),
	// 	.i_p_read_or_write  (read_or_write),
	// 	.i_p_mem_read       (cache_driver_in_mem_read),
	// 	.i_p_mem_write      (cache_driver_in_mem_write),
	// 	.o_p_readdata       (cache_driver_out_data),
	// 	.o_p_delay          (cache_delay),
	// 	.o_m_evict_addr     (VX_dram_req_rsp.o_m_evict_addr),
	// 	.o_m_read_addr      (VX_dram_req_rsp.o_m_read_addr),
	// 	.o_m_valid          (VX_dram_req_rsp.o_m_valid),
	// 	.o_m_writedata      (VX_dram_req_rsp.o_m_writedata),
	// 	.o_m_read_or_write  (VX_dram_req_rsp.o_m_read_or_write),
	// 	.i_m_readdata       (VX_dram_req_rsp.i_m_readdata),
	// 	.i_m_ready          (VX_dram_req_rsp.i_m_ready)
	// 	);


VX_d_cache #(
		.CACHE_SIZE          (`ICACHE_SIZE),
		.CACHE_WAYS          (`ICACHE_WAYS),
		.CACHE_BLOCK         (`ICACHE_BLOCK),
		.CACHE_BANKS         (`ICACHE_BANKS),
		.LOG_NUM_BANKS       (`ICACHE_LOG_NUM_BANKS),
		.NUM_REQ             (`ICACHE_NUM_REQ),
		.LOG_NUM_REQ         (`ICACHE_LOG_NUM_REQ),
		.NUM_IND			   (`ICACHE_NUM_IND),
		.CACHE_WAY_INDEX     (`ICACHE_WAY_INDEX),
		.NUM_WORDS_PER_BLOCK (`ICACHE_NUM_WORDS_PER_BLOCK),
		.OFFSET_SIZE_START   (`ICACHE_OFFSET_ST),
		.OFFSET_SIZE_END     (`ICACHE_OFFSET_ED),
		.TAG_SIZE_START      (`ICACHE_TAG_SIZE_START),
		.TAG_SIZE_END        (`ICACHE_TAG_SIZE_END),
		.IND_SIZE_START      (`ICACHE_IND_SIZE_START),
		.IND_SIZE_END        (`ICACHE_IND_SIZE_END),
		.ADDR_TAG_START      (`ICACHE_ADDR_TAG_START),
		.ADDR_TAG_END        (`ICACHE_ADDR_TAG_END),
		.ADDR_OFFSET_START   (`ICACHE_ADDR_OFFSET_ST),
		.ADDR_OFFSET_END     (`ICACHE_ADDR_OFFSET_ED),
		.ADDR_IND_START      (`ICACHE_IND_ST),
		.ADDR_IND_END        (`ICACHE_IND_ED),
		.MEM_ADDR_REQ_MASK   (`ICACHE_MEM_REQ_ADDR_MASK)
		) icache
		(
			.clk                (clk),
			.rst                (reset),
			.i_p_valid          (icache_driver_in_valid),
			.i_p_addr           (icache_driver_in_address),
			.i_p_writedata      (icache_driver_in_data),
			.i_p_read_or_write  (read_or_write_ic),
			.i_p_mem_read       (icache_driver_in_mem_read),
			.i_p_mem_write      (icache_driver_in_mem_write),
			.o_p_readdata       (icache_instruction_out),
			.o_p_delay          (icache_delay),
			.o_m_evict_addr     (VX_dram_req_rsp_icache.o_m_evict_addr),
			.o_m_read_addr      (VX_dram_req_rsp_icache.o_m_read_addr),
			.o_m_valid          (VX_dram_req_rsp_icache.o_m_valid),
			.o_m_writedata      (VX_dram_req_rsp_icache.o_m_writedata),
			.o_m_read_or_write  (VX_dram_req_rsp_icache.o_m_read_or_write),
			.i_m_readdata       (VX_dram_req_rsp_icache.i_m_readdata),
			.i_m_ready          (VX_dram_req_rsp_icache.i_m_ready)
			);

	// assign VX_dcache_rsp.in_cache_driver_out_data = (to_shm && 0)   ? sm_driver_out_data : cache_driver_out_data;
	// assign VX_dcache_rsp.delay                    = (sm_delay && 0) || cache_delay;

	assign VX_icache_rsp.instruction              = icache_instruction_out;
	assign VX_icache_rsp.delay                    = icache_delay;


endmodule
