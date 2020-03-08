`include "VX_define.v"

module VX_dmem_controller (
	input wire               clk,
	input wire               reset,

	// Dram <-> Dcache
	VX_gpu_dcache_dram_req_inter VX_gpu_dcache_dram_req,
	VX_gpu_dcache_dram_res_inter VX_gpu_dcache_dram_res,

	// Dram <-> Icache
	VX_gpu_dcache_dram_req_inter VX_gpu_icache_dram_req,
	VX_gpu_dcache_dram_res_inter VX_gpu_icache_dram_res,

	// Core <-> Dcache
	VX_gpu_dcache_res_inter  VX_dcache_rsp,
	VX_gpu_dcache_req_inter  VX_dcache_req,

	// Core <-> Icache
	VX_gpu_dcache_res_inter  VX_icache_rsp,
	VX_gpu_dcache_req_inter  VX_icache_req
);

	wire to_shm = VX_dcache_req.core_req_addr[0][31:24] == 8'hFF;

	wire[`NT_M1:0]       cache_driver_in_valid  = VX_dcache_req.core_req_valid & {`NT{~to_shm}};

	// wire[`NT_M1:0]       sm_driver_in_valid     = VX_dcache_req.core_req_valid & {`NT{to_shm}};
	// wire[2:0]            sm_driver_in_mem_read  = !(|sm_driver_in_valid) ? `NO_MEM_READ   : VX_dcache_req.core_req_mem_read;
	// wire[2:0]            sm_driver_in_mem_write = !(|sm_driver_in_valid) ? `NO_MEM_WRITE  : VX_dcache_req.core_req_mem_write;

	// wire[`NT_M1:0][31:0] cache_driver_out_data;
	// wire[`NT_M1:0][31:0] sm_driver_out_data;
	// wire[`NT_M1:0]       cache_driver_out_valid; // Not used for now
	// wire                 sm_delay;


	// VX_shared_memory #(
	// 	.SM_SIZE                (`SHARED_MEMORY_SIZE),
	//   .SM_BANKS 				(`SHARED_MEMORY_BANKS),
	//   .SM_BYTES_PER_READ      (`SHARED_MEMORY_BYTES_PER_READ),
	//   .SM_WORDS_PER_READ      (`SHARED_MEMORY_WORDS_PER_READ),
	//   .SM_LOG_WORDS_PER_READ  (`SHARED_MEMORY_LOG_WORDS_PER_READ),
	//   .SM_BANK_OFFSET_START   (`SHARED_MEMORY_BANK_OFFSET_ST),
	//   .SM_BANK_OFFSET_END     (`SHARED_MEMORY_BANK_OFFSET_ED),
	//   .SM_BLOCK_OFFSET_START  (`SHARED_MEMORY_BLOCK_OFFSET_ST),
	//   .SM_BLOCK_OFFSET_END    (`SHARED_MEMORY_BLOCK_OFFSET_ED),
	//   .SM_INDEX_START         (`SHARED_MEMORY_INDEX_OFFSET_ST),
	//   .SM_INDEX_END           (`SHARED_MEMORY_INDEX_OFFSET_ED),
	//   .SM_HEIGHT 			  (`SHARED_MEMORY_HEIGHT),
	//   .NUM_REQ                (`SHARED_MEMORY_NUM_REQ),
	//   .BITS_PER_BANK 		  (`SHARED_MEMORY_BITS_PER_BANK) 
	// 	)
	// 	shared_memory
	// 	(
	// 	.clk       (clk),
	// 	.reset     (reset),
	// 	.in_valid  (sm_driver_in_valid),
	// 	.in_address(VX_dcache_req.core_req_addr),
	// 	.in_data   (VX_dcache_req.core_req_writedata),
	// 	.mem_read  (sm_driver_in_mem_read),
	// 	.mem_write (sm_driver_in_mem_write),
	// 	.out_valid (cache_driver_out_valid),
	// 	.out_data  (sm_driver_out_data),
	// 	.stall     (sm_delay)
	// 	);


    wire                                                    Dllvq_pop;
    wire[`DNUMBER_REQUESTS-1:0]                             Dllvq_valid;
    wire[`DNUMBER_REQUESTS-1:0][31:0]                       Dllvq_res_addr;
    wire[`DNUMBER_REQUESTS-1:0][`DBANK_LINE_SIZE_RNG][31:0] Dllvq_res_data;

    assign Dllvq_pop = 0;
	VX_cache #(
		.CACHE_SIZE_BYTES             (`DCACHE_SIZE_BYTES),
		.BANK_LINE_SIZE_BYTES         (`DBANK_LINE_SIZE_BYTES),
		.NUMBER_BANKS                 (`DNUMBER_BANKS),
		.WORD_SIZE_BYTES              (`DWORD_SIZE_BYTES),
		.NUMBER_REQUESTS              (`DNUMBER_REQUESTS),
		.STAGE_1_CYCLES               (`DSTAGE_1_CYCLES),
		.REQQ_SIZE                    (`DREQQ_SIZE),
		.MRVQ_SIZE                    (`DMRVQ_SIZE),
		.DFPQ_SIZE                    (`DDFPQ_SIZE),
		.SNRQ_SIZE                    (`DSNRQ_SIZE),
		.CWBQ_SIZE                    (`DCWBQ_SIZE),
		.DWBQ_SIZE                    (`DDWBQ_SIZE),
		.DFQQ_SIZE                    (`DDFQQ_SIZE),
		.LLVQ_SIZE                    (`DLLVQ_SIZE),
		.FILL_INVALIDAOR_SIZE         (`DFILL_INVALIDAOR_SIZE),
		.SIMULATED_DRAM_LATENCY_CYCLES(`DSIMULATED_DRAM_LATENCY_CYCLES)
		)
		gpu_dcache
		(
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
		.core_req_pc       (VX_dcache_req.core_req_pc),

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
		.core_wb_pc        (VX_dcache_rsp.core_wb_pc),

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
		.snp_req_addr          (0),

		// LLVQ stuff
		.llvq_pop              (Dllvq_pop),
		.llvq_valid            (Dllvq_valid),
		.llvq_res_addr         (Dllvq_res_addr),
		.llvq_res_data         (Dllvq_res_data)
		);



	VX_cache #(
		.CACHE_SIZE_BYTES             (`ICACHE_SIZE_BYTES),
		.BANK_LINE_SIZE_BYTES         (`IBANK_LINE_SIZE_BYTES),
		.NUMBER_BANKS                 (`INUMBER_BANKS),
		.WORD_SIZE_BYTES              (`IWORD_SIZE_BYTES),
		.NUMBER_REQUESTS              (`INUMBER_REQUESTS),
		.STAGE_1_CYCLES               (`ISTAGE_1_CYCLES),
		.REQQ_SIZE                    (`IREQQ_SIZE),
		.MRVQ_SIZE                    (`IMRVQ_SIZE),
		.DFPQ_SIZE                    (`IDFPQ_SIZE),
		.SNRQ_SIZE                    (`ISNRQ_SIZE),
		.CWBQ_SIZE                    (`ICWBQ_SIZE),
		.DWBQ_SIZE                    (`IDWBQ_SIZE),
		.DFQQ_SIZE                    (`IDFQQ_SIZE),
		.LLVQ_SIZE                    (`ILLVQ_SIZE),
		.FILL_INVALIDAOR_SIZE         (`IFILL_INVALIDAOR_SIZE),
		.SIMULATED_DRAM_LATENCY_CYCLES(`ISIMULATED_DRAM_LATENCY_CYCLES)
		)
		gpu_icache
		(
		.clk               (clk),
		.reset             (reset),

		// Core req
		.core_req_valid    (VX_icache_req.core_req_valid),
		.core_req_addr     (VX_icache_req.core_req_addr),
		.core_req_writedata(VX_icache_req.core_req_writedata),
		.core_req_mem_read (VX_icache_req.core_req_mem_read),
		.core_req_mem_write(VX_icache_req.core_req_mem_write),
		.core_req_rd       (VX_icache_req.core_req_rd),
		.core_req_wb       (VX_icache_req.core_req_wb),
		.core_req_warp_num (VX_icache_req.core_req_warp_num),
		.core_req_pc       (VX_icache_req.core_req_pc),

		// Delay Core Req
		.delay_req         (VX_icache_rsp.delay_req),

		// Core Cache Can't WB
		.core_no_wb_slot   (VX_icache_req.core_no_wb_slot),

		// Cache CWB
		.core_wb_valid     (VX_icache_rsp.core_wb_valid),
		.core_wb_req_rd    (VX_icache_rsp.core_wb_req_rd),
		.core_wb_req_wb    (VX_icache_rsp.core_wb_req_wb),
		.core_wb_warp_num  (VX_icache_rsp.core_wb_warp_num),
		.core_wb_readdata  (VX_icache_rsp.core_wb_readdata),
		.core_wb_pc        (VX_icache_rsp.core_wb_pc),

		// DRAM response
		.dram_fill_rsp     (VX_gpu_icache_dram_res.dram_fill_rsp),
		.dram_fill_rsp_addr(VX_gpu_icache_dram_res.dram_fill_rsp_addr),
		.dram_fill_rsp_data(VX_gpu_icache_dram_res.dram_fill_rsp_data),

		// DRAM accept response
		.dram_fill_accept  (VX_gpu_icache_dram_req.dram_fill_accept),

		// DRAM Req
		.dram_req          (VX_gpu_icache_dram_req.dram_req),
		.dram_req_write    (VX_gpu_icache_dram_req.dram_req_write),
		.dram_req_read     (VX_gpu_icache_dram_req.dram_req_read),
		.dram_req_addr     (VX_gpu_icache_dram_req.dram_req_addr),
		.dram_req_size     (VX_gpu_icache_dram_req.dram_req_size),
		.dram_req_data     (VX_gpu_icache_dram_req.dram_req_data),

		// Snoop Response
		.dram_req_because_of_wb(VX_gpu_icache_dram_req.dram_because_of_snp),
		.dram_snp_full         (VX_gpu_icache_dram_req.dram_snp_full),

		// Snoop Request
		.snp_req               (0),
		.snp_req_addr          (0),

		// LLVQ stuff
		.llvq_pop              (Dllvq_pop),
		.llvq_valid            (Dllvq_valid),
		.llvq_res_addr         (Dllvq_res_addr),
		.llvq_res_data         (Dllvq_res_data)
		);



endmodule
