`include "VX_define.vh"

module VX_dmem_controller (
	input wire               clk,
	input wire               reset,

	// Dram <-> Dcache
	VX_gpu_dcache_dram_req_if vx_gpu_dcache_dram_req,
	VX_gpu_dcache_dram_rsp_if vx_gpu_dcache_dram_res,
	VX_gpu_snp_req_rsp_if     vx_gpu_dcache_snp_req,

	// Dram <-> Icache
	VX_gpu_dcache_dram_req_if vx_gpu_icache_dram_req,
	VX_gpu_dcache_dram_rsp_if vx_gpu_icache_dram_res,
	VX_gpu_snp_req_rsp_if     vx_gpu_icache_snp_req,

	// Core <-> Dcache
	VX_gpu_dcache_rsp_if  vx_dcache_rsp,
	VX_gpu_dcache_req_if  vx_dcache_req,

	// Core <-> Icache
	VX_gpu_dcache_rsp_if  vx_icache_rsp,
	VX_gpu_dcache_req_if  vx_icache_req
);

	VX_gpu_dcache_rsp_if   #(.NUM_REQUESTS(`DNUM_REQUESTS))   vx_dcache_rsp_smem();
	VX_gpu_dcache_req_if   #(.NUM_REQUESTS(`DNUM_REQUESTS))   vx_dcache_req_smem();

	VX_gpu_dcache_rsp_if   #(.NUM_REQUESTS(`DNUM_REQUESTS))   vx_dcache_rsp_dcache();
	VX_gpu_dcache_req_if   #(.NUM_REQUESTS(`DNUM_REQUESTS))   vx_dcache_req_dcache();

	wire to_shm          = vx_dcache_req.core_req_addr[0][31:24] == 8'hFF;
    wire dcache_wants_wb = (|vx_dcache_rsp_dcache.core_wb_valid);

	// Dcache Request
	assign vx_dcache_req_dcache.core_req_valid       = vx_dcache_req.core_req_valid & {`NUM_THREADS{~to_shm}};
	assign vx_dcache_req_dcache.core_req_addr        = vx_dcache_req.core_req_addr;
	assign vx_dcache_req_dcache.core_req_writedata   = vx_dcache_req.core_req_writedata;
	assign vx_dcache_req_dcache.core_req_mem_read    = vx_dcache_req.core_req_mem_read;
	assign vx_dcache_req_dcache.core_req_mem_write   = vx_dcache_req.core_req_mem_write;
    assign vx_dcache_req_dcache.core_req_rd          = vx_dcache_req.core_req_rd;
    assign vx_dcache_req_dcache.core_req_wb          = vx_dcache_req.core_req_wb;
    assign vx_dcache_req_dcache.core_req_warp_num    = vx_dcache_req.core_req_warp_num;
    assign vx_dcache_req_dcache.core_req_pc          = vx_dcache_req.core_req_pc;
    assign vx_dcache_req_dcache.core_no_wb_slot      = vx_dcache_req.core_no_wb_slot;

    // Shred Memory Request
	assign vx_dcache_req_smem.core_req_valid       = vx_dcache_req.core_req_valid & {`NUM_THREADS{to_shm}};
	assign vx_dcache_req_smem.core_req_addr        = vx_dcache_req.core_req_addr;
	assign vx_dcache_req_smem.core_req_writedata   = vx_dcache_req.core_req_writedata;
	assign vx_dcache_req_smem.core_req_mem_read    = vx_dcache_req.core_req_mem_read;
	assign vx_dcache_req_smem.core_req_mem_write   = vx_dcache_req.core_req_mem_write;
    assign vx_dcache_req_smem.core_req_rd          = vx_dcache_req.core_req_rd;
    assign vx_dcache_req_smem.core_req_wb          = vx_dcache_req.core_req_wb;
    assign vx_dcache_req_smem.core_req_warp_num    = vx_dcache_req.core_req_warp_num;
    assign vx_dcache_req_smem.core_req_pc          = vx_dcache_req.core_req_pc;
    assign vx_dcache_req_smem.core_no_wb_slot      = vx_dcache_req.core_no_wb_slot || dcache_wants_wb;

    // Dcache Response
    assign vx_dcache_rsp.core_wb_valid     = dcache_wants_wb ? vx_dcache_rsp_dcache.core_wb_valid    : vx_dcache_rsp_smem.core_wb_valid;
    assign vx_dcache_rsp.core_wb_req_rd    = dcache_wants_wb ? vx_dcache_rsp_dcache.core_wb_req_rd   : vx_dcache_rsp_smem.core_wb_req_rd;
    assign vx_dcache_rsp.core_wb_req_wb    = dcache_wants_wb ? vx_dcache_rsp_dcache.core_wb_req_wb   : vx_dcache_rsp_smem.core_wb_req_wb;
    assign vx_dcache_rsp.core_wb_warp_num  = dcache_wants_wb ? vx_dcache_rsp_dcache.core_wb_warp_num : vx_dcache_rsp_smem.core_wb_warp_num;
    assign vx_dcache_rsp.core_wb_readdata  = dcache_wants_wb ? vx_dcache_rsp_dcache.core_wb_readdata : vx_dcache_rsp_smem.core_wb_readdata;
    assign vx_dcache_rsp.core_wb_pc        = dcache_wants_wb ? vx_dcache_rsp_dcache.core_wb_pc       : vx_dcache_rsp_smem.core_wb_pc;

    assign vx_dcache_rsp.delay_req         = to_shm          ? vx_dcache_rsp_smem.delay_req : vx_dcache_rsp_dcache.delay_req;

	VX_gpu_dcache_dram_req_if #(.BANK_LINE_WORDS(`DBANK_LINE_WORDS)) vx_gpu_smem_dram_req();
	VX_gpu_dcache_dram_rsp_if #(.BANK_LINE_WORDS(`DBANK_LINE_WORDS)) vx_gpu_smem_dram_res();

	VX_cache #(
		.CACHE_SIZE_BYTES             (`SCACHE_SIZE_BYTES),
		.BANK_LINE_SIZE_BYTES         (`SBANK_LINE_SIZE_BYTES),
		.NUM_BANKS                    (`SNUM_BANKS),
		.WORD_SIZE_BYTES              (`SWORD_SIZE_BYTES),
		.NUM_REQUESTS                 (`SNUM_REQUESTS),
		.STAGE_1_CYCLES               (`SSTAGE_1_CYCLES),
		.FUNC_ID                      (`SFUNC_ID),
		.REQQ_SIZE                    (`SREQQ_SIZE),
		.MRVQ_SIZE                    (`SMRVQ_SIZE),
		.DFPQ_SIZE                    (`SDFPQ_SIZE),
		.SNRQ_SIZE                    (`SSNRQ_SIZE),
		.CWBQ_SIZE                    (`SCWBQ_SIZE),
		.DWBQ_SIZE                    (`SDWBQ_SIZE),
		.DFQQ_SIZE                    (`SDFQQ_SIZE),
		.LLVQ_SIZE                    (`SLLVQ_SIZE),
		.FFSQ_SIZE                    (`SFFSQ_SIZE),
		.PRFQ_SIZE                    (`SPRFQ_SIZE),
		.PRFQ_STRIDE                  (`SPRFQ_STRIDE),
		.FILL_INVALIDAOR_SIZE         (`SFILL_INVALIDAOR_SIZE),
		.SIMULATED_DRAM_LATENCY_CYCLES(`SSIMULATED_DRAM_LATENCY_CYCLES)
	) gpu_smem (
		.clk               (clk),
		.reset             (reset),

		// Core req
		.core_req_valid    (vx_dcache_req_smem.core_req_valid),
		.core_req_mem_read (vx_dcache_req_smem.core_req_mem_read),
		.core_req_mem_write(vx_dcache_req_smem.core_req_mem_write),
		.core_req_addr     (vx_dcache_req_smem.core_req_addr),
		.core_req_writedata(vx_dcache_req_smem.core_req_writedata),		
		.core_req_rd       (vx_dcache_req_smem.core_req_rd),
		.core_req_wb       (vx_dcache_req_smem.core_req_wb),
		.core_req_warp_num (vx_dcache_req_smem.core_req_warp_num),
		.core_req_pc       (vx_dcache_req_smem.core_req_pc),

		// Delay Core Req
		.delay_req         (vx_dcache_rsp_smem.delay_req),

		// Core Cache Can't WB
		.core_no_wb_slot   (vx_dcache_req_smem.core_no_wb_slot),

		// Cache CWB
		.core_wb_valid     	(vx_dcache_rsp_smem.core_wb_valid),
		.core_wb_req_rd    	(vx_dcache_rsp_smem.core_wb_req_rd),
		.core_wb_req_wb    	(vx_dcache_rsp_smem.core_wb_req_wb),
		.core_wb_warp_num  	(vx_dcache_rsp_smem.core_wb_warp_num),
		.core_wb_readdata  	(vx_dcache_rsp_smem.core_wb_readdata),
		.core_wb_pc        	(vx_dcache_rsp_smem.core_wb_pc),
	`IGNORE_WARNINGS_BEGIN
		.core_wb_address   	(),
	`IGNORE_WARNINGS_END

		// DRAM response
		.dram_rsp_valid     (vx_gpu_smem_dram_res.dram_rsp_valid),
		.dram_rsp_addr		(vx_gpu_smem_dram_res.dram_rsp_addr),
		.dram_rsp_data		(vx_gpu_smem_dram_res.dram_rsp_data),

		// DRAM accept response
		.dram_rsp_ready  	(vx_gpu_smem_dram_req.dram_rsp_ready),

		// DRAM Req
		.dram_req_read     	(vx_gpu_smem_dram_req.dram_req_read),
		.dram_req_write    	(vx_gpu_smem_dram_req.dram_req_write),		
		.dram_req_addr     	(vx_gpu_smem_dram_req.dram_req_addr),
		.dram_req_data     	(vx_gpu_smem_dram_req.dram_req_data),
		.dram_req_full    	(1),

		// Snoop Request
		.snp_req_valid      (0),
		.snp_req_addr       (0),
	`IGNORE_WARNINGS_BEGIN
		.snp_req_full       (),
	`IGNORE_WARNINGS_END

		// Snoop Forward
	`IGNORE_WARNINGS_BEGIN
		.snp_fwd_valid      (),
		.snp_fwd_addr       (),
	`IGNORE_WARNINGS_END
		.snp_fwd_full       (0)
	);

	VX_cache #(
		.CACHE_SIZE_BYTES             (`DCACHE_SIZE_BYTES),
		.BANK_LINE_SIZE_BYTES         (`DBANK_LINE_SIZE_BYTES),
		.NUM_BANKS                    (`DNUM_BANKS),
		.WORD_SIZE_BYTES              (`DWORD_SIZE_BYTES),
		.NUM_REQUESTS                 (`DNUM_REQUESTS),
		.STAGE_1_CYCLES               (`DSTAGE_1_CYCLES),
		.FUNC_ID                      (`DFUNC_ID),
		.REQQ_SIZE                    (`DREQQ_SIZE),
		.MRVQ_SIZE                    (`DMRVQ_SIZE),
		.DFPQ_SIZE                    (`DDFPQ_SIZE),
		.SNRQ_SIZE                    (`DSNRQ_SIZE),
		.CWBQ_SIZE                    (`DCWBQ_SIZE),
		.DWBQ_SIZE                    (`DDWBQ_SIZE),
		.DFQQ_SIZE                    (`DDFQQ_SIZE),
		.LLVQ_SIZE                    (`DLLVQ_SIZE),
		.FFSQ_SIZE                    (`DFFSQ_SIZE),
		.PRFQ_SIZE                    (`DPRFQ_SIZE),
		.PRFQ_STRIDE                  (`DPRFQ_STRIDE),
		.FILL_INVALIDAOR_SIZE         (`DFILL_INVALIDAOR_SIZE),
		.SIMULATED_DRAM_LATENCY_CYCLES(`DSIMULATED_DRAM_LATENCY_CYCLES)
	) gpu_dcache (
		.clk               (clk),
		.reset             (reset),

		// Core req
		.core_req_valid    (vx_dcache_req_dcache.core_req_valid),
		.core_req_mem_read (vx_dcache_req_dcache.core_req_mem_read),
		.core_req_mem_write(vx_dcache_req_dcache.core_req_mem_write),
		.core_req_addr     (vx_dcache_req_dcache.core_req_addr),
		.core_req_writedata(vx_dcache_req_dcache.core_req_writedata),		
		.core_req_rd       (vx_dcache_req_dcache.core_req_rd),
		.core_req_wb       (vx_dcache_req_dcache.core_req_wb),
		.core_req_warp_num (vx_dcache_req_dcache.core_req_warp_num),
		.core_req_pc       (vx_dcache_req_dcache.core_req_pc),

		// Delay Core Req
		.delay_req         (vx_dcache_rsp_dcache.delay_req),

		// Core Cache Can't WB
		.core_no_wb_slot   (vx_dcache_req_dcache.core_no_wb_slot),

		// Cache CWB
		.core_wb_valid     (vx_dcache_rsp_dcache.core_wb_valid),
		.core_wb_req_rd    (vx_dcache_rsp_dcache.core_wb_req_rd),
		.core_wb_req_wb    (vx_dcache_rsp_dcache.core_wb_req_wb),
		.core_wb_warp_num  (vx_dcache_rsp_dcache.core_wb_warp_num),
		.core_wb_readdata  (vx_dcache_rsp_dcache.core_wb_readdata),
		.core_wb_pc        (vx_dcache_rsp_dcache.core_wb_pc),
	`IGNORE_WARNINGS_BEGIN
		.core_wb_address   (),
	`IGNORE_WARNINGS_END

		// DRAM response
		.dram_rsp_valid     (vx_gpu_dcache_dram_res.dram_rsp_valid),
		.dram_rsp_addr      (vx_gpu_dcache_dram_res.dram_rsp_addr),
		.dram_rsp_data      (vx_gpu_dcache_dram_res.dram_rsp_data),

		// DRAM accept response
		.dram_rsp_ready  	(vx_gpu_dcache_dram_req.dram_rsp_ready),

		// DRAM Req
		.dram_req_read      (vx_gpu_dcache_dram_req.dram_req_read),
		.dram_req_write     (vx_gpu_dcache_dram_req.dram_req_write),		
		.dram_req_addr      (vx_gpu_dcache_dram_req.dram_req_addr),
		.dram_req_data      (vx_gpu_dcache_dram_req.dram_req_data),
		.dram_req_full      (vx_gpu_dcache_dram_req.dram_req_full),

		// Snoop Request
		.snp_req_valid      (vx_gpu_dcache_snp_req.snp_req_valid),
		.snp_req_addr       (vx_gpu_dcache_snp_req.snp_req_addr),
		.snp_req_full       (vx_gpu_dcache_snp_req.snp_req_full),

		// Snoop Forward
	`IGNORE_WARNINGS_BEGIN
		.snp_fwd_valid      (),
		.snp_fwd_addr       (),
	`IGNORE_WARNINGS_END
		.snp_fwd_full       (0)
	);

	VX_cache #(
		.CACHE_SIZE_BYTES             (`ICACHE_SIZE_BYTES),
		.BANK_LINE_SIZE_BYTES         (`IBANK_LINE_SIZE_BYTES),
		.NUM_BANKS                    (`INUM_BANKS),
		.WORD_SIZE_BYTES              (`IWORD_SIZE_BYTES),
		.NUM_REQUESTS                 (`INUM_REQUESTS),
		.STAGE_1_CYCLES               (`ISTAGE_1_CYCLES),
		.FUNC_ID                      (`IFUNC_ID),
		.REQQ_SIZE                    (`IREQQ_SIZE),
		.MRVQ_SIZE                    (`IMRVQ_SIZE),
		.DFPQ_SIZE                    (`IDFPQ_SIZE),
		.SNRQ_SIZE                    (`ISNRQ_SIZE),
		.CWBQ_SIZE                    (`ICWBQ_SIZE),
		.DWBQ_SIZE                    (`IDWBQ_SIZE),
		.DFQQ_SIZE                    (`IDFQQ_SIZE),
		.LLVQ_SIZE                    (`ILLVQ_SIZE),
		.FFSQ_SIZE                    (`IFFSQ_SIZE),
		.PRFQ_SIZE                    (`IPRFQ_SIZE),
		.PRFQ_STRIDE                  (`IPRFQ_STRIDE),
		.FILL_INVALIDAOR_SIZE         (`IFILL_INVALIDAOR_SIZE),
		.SIMULATED_DRAM_LATENCY_CYCLES(`ISIMULATED_DRAM_LATENCY_CYCLES)
	) gpu_icache (
		.clk               	(clk),
		.reset             	(reset),

		// Core req
		.core_req_valid     (vx_icache_req.core_req_valid),
		.core_req_mem_read 	(vx_icache_req.core_req_mem_read),
		.core_req_mem_write	(vx_icache_req.core_req_mem_write),
		.core_req_addr     	(vx_icache_req.core_req_addr),
		.core_req_writedata	(vx_icache_req.core_req_writedata),		
		.core_req_rd       	(vx_icache_req.core_req_rd),
		.core_req_wb       	(vx_icache_req.core_req_wb),
		.core_req_warp_num 	(vx_icache_req.core_req_warp_num),
		.core_req_pc       	(vx_icache_req.core_req_pc),

		// Delay Core Req
		.delay_req         	(vx_icache_rsp.delay_req),

		// Core Cache Can't WB
		.core_no_wb_slot   	(vx_icache_req.core_no_wb_slot),

		// Cache CWB
		.core_wb_valid    	(vx_icache_rsp.core_wb_valid),
		.core_wb_req_rd   	(vx_icache_rsp.core_wb_req_rd),
		.core_wb_req_wb   	(vx_icache_rsp.core_wb_req_wb),
		.core_wb_warp_num	(vx_icache_rsp.core_wb_warp_num),
		.core_wb_readdata  	(vx_icache_rsp.core_wb_readdata),
		.core_wb_pc        	(vx_icache_rsp.core_wb_pc),
	`IGNORE_WARNINGS_BEGIN
		.core_wb_address   	(),
	`IGNORE_WARNINGS_END

		// DRAM response
		.dram_rsp_valid     (vx_gpu_icache_dram_res.dram_rsp_valid),
		.dram_rsp_addr		(vx_gpu_icache_dram_res.dram_rsp_addr),
		.dram_rsp_data		(vx_gpu_icache_dram_res.dram_rsp_data),

		// DRAM accept response
		.dram_rsp_ready   	(vx_gpu_icache_dram_req.dram_rsp_ready),

		// DRAM Req
		.dram_req_read     	(vx_gpu_icache_dram_req.dram_req_read),
		.dram_req_write    	(vx_gpu_icache_dram_req.dram_req_write),		
		.dram_req_addr     	(vx_gpu_icache_dram_req.dram_req_addr),
		.dram_req_data     	(vx_gpu_icache_dram_req.dram_req_data),
		.dram_req_full    	(vx_gpu_icache_dram_req.dram_req_full),

		// Snoop Request
		.snp_req_valid     	(vx_gpu_icache_snp_req.snp_req_valid),
		.snp_req_addr       (vx_gpu_icache_snp_req.snp_req_addr),
		.snp_req_full       (vx_gpu_icache_snp_req.snp_req_full),

		// Snoop Forward
	`IGNORE_WARNINGS_BEGIN
		.snp_fwd_valid      (),
		.snp_fwd_addr       (),
	`IGNORE_WARNINGS_END
		.snp_fwd_full       (0)
	);

endmodule
