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


	VX_gpu_dcache_res_inter      VX_dcache_rsp_smem();
	VX_gpu_dcache_req_inter      VX_dcache_req_smem();


	VX_gpu_dcache_res_inter      VX_dcache_rsp_dcache();
	VX_gpu_dcache_req_inter      VX_dcache_req_dcache();


	wire to_shm          = VX_dcache_req.core_req_addr[0][31:24] == 8'hFF;
    wire dcache_wants_wb = (|VX_dcache_rsp_dcache.core_wb_valid);

	// Dcache Request
	assign VX_dcache_req_dcache.core_req_valid       = VX_dcache_req.core_req_valid & {`NT{~to_shm}};
	assign VX_dcache_req_dcache.core_req_addr        = VX_dcache_req.core_req_addr;
	assign VX_dcache_req_dcache.core_req_writedata   = VX_dcache_req.core_req_writedata;
	assign VX_dcache_req_dcache.core_req_mem_read    = VX_dcache_req.core_req_mem_read;
	assign VX_dcache_req_dcache.core_req_mem_write   = VX_dcache_req.core_req_mem_write;
    assign VX_dcache_req_dcache.core_req_rd          = VX_dcache_req.core_req_rd;
    assign VX_dcache_req_dcache.core_req_wb          = VX_dcache_req.core_req_wb;
    assign VX_dcache_req_dcache.core_req_warp_num    = VX_dcache_req.core_req_warp_num;
    assign VX_dcache_req_dcache.core_req_pc          = VX_dcache_req.core_req_pc;
    assign VX_dcache_req_dcache.core_no_wb_slot      = VX_dcache_req.core_no_wb_slot;


    // Shred Memory Request
	assign VX_dcache_req_smem.core_req_valid       = VX_dcache_req.core_req_valid & {`NT{to_shm}};
	assign VX_dcache_req_smem.core_req_addr        = VX_dcache_req.core_req_addr;
	assign VX_dcache_req_smem.core_req_writedata   = VX_dcache_req.core_req_writedata;
	assign VX_dcache_req_smem.core_req_mem_read    = VX_dcache_req.core_req_mem_read;
	assign VX_dcache_req_smem.core_req_mem_write   = VX_dcache_req.core_req_mem_write;
    assign VX_dcache_req_smem.core_req_rd          = VX_dcache_req.core_req_rd;
    assign VX_dcache_req_smem.core_req_wb          = VX_dcache_req.core_req_wb;
    assign VX_dcache_req_smem.core_req_warp_num    = VX_dcache_req.core_req_warp_num;
    assign VX_dcache_req_smem.core_req_pc          = VX_dcache_req.core_req_pc;
    assign VX_dcache_req_smem.core_no_wb_slot      = VX_dcache_req.core_no_wb_slot || dcache_wants_wb;


    // Dcache Response
    assign VX_dcache_rsp.core_wb_valid     = dcache_wants_wb ? VX_dcache_rsp_dcache.core_wb_valid    : VX_dcache_rsp_smem.core_wb_valid;
    assign VX_dcache_rsp.core_wb_req_rd    = dcache_wants_wb ? VX_dcache_rsp_dcache.core_wb_req_rd   : VX_dcache_rsp_smem.core_wb_req_rd;
    assign VX_dcache_rsp.core_wb_req_wb    = dcache_wants_wb ? VX_dcache_rsp_dcache.core_wb_req_wb   : VX_dcache_rsp_smem.core_wb_req_wb;
    assign VX_dcache_rsp.core_wb_warp_num  = dcache_wants_wb ? VX_dcache_rsp_dcache.core_wb_warp_num : VX_dcache_rsp_smem.core_wb_warp_num;
    assign VX_dcache_rsp.core_wb_readdata  = dcache_wants_wb ? VX_dcache_rsp_dcache.core_wb_readdata : VX_dcache_rsp_smem.core_wb_readdata;
    assign VX_dcache_rsp.core_wb_pc        = dcache_wants_wb ? VX_dcache_rsp_dcache.core_wb_pc       : VX_dcache_rsp_smem.core_wb_pc;

    assign VX_dcache_rsp.delay_req         = to_shm          ? VX_dcache_rsp_smem.delay_req : VX_dcache_rsp_dcache.delay_req;





    wire                                                    Sllvq_pop;
    wire[`DNUMBER_REQUESTS-1:0]                             Sllvq_valid;
    wire[`DNUMBER_REQUESTS-1:0][31:0]                       Sllvq_res_addr;
    wire[`DNUMBER_REQUESTS-1:0][`DBANK_LINE_SIZE_RNG][31:0] Sllvq_res_data;

	VX_gpu_dcache_dram_req_inter VX_gpu_smem_dram_req();
	VX_gpu_dcache_dram_res_inter VX_gpu_smem_dram_res();



    assign Sllvq_pop = 0;
	VX_cache #(
		.CACHE_SIZE_BYTES             (`SCACHE_SIZE_BYTES),
		.BANK_LINE_SIZE_BYTES         (`SBANK_LINE_SIZE_BYTES),
		.NUMBER_BANKS                 (`SNUMBER_BANKS),
		.WORD_SIZE_BYTES              (`SWORD_SIZE_BYTES),
		.NUMBER_REQUESTS              (`SNUMBER_REQUESTS),
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
		.FILL_INVALIDAOR_SIZE         (`SFILL_INVALIDAOR_SIZE),
		.SIMULATED_DRAM_LATENCY_CYCLES(`SSIMULATED_DRAM_LATENCY_CYCLES)
		)
		gpu_smem
		(
		.clk               (clk),
		.reset             (reset),

		// Core req
		.core_req_valid    (VX_dcache_req_smem.core_req_valid),
		.core_req_addr     (VX_dcache_req_smem.core_req_addr),
		.core_req_writedata(VX_dcache_req_smem.core_req_writedata),
		.core_req_mem_read (VX_dcache_req_smem.core_req_mem_read),
		.core_req_mem_write(VX_dcache_req_smem.core_req_mem_write),
		.core_req_rd       (VX_dcache_req_smem.core_req_rd),
		.core_req_wb       (VX_dcache_req_smem.core_req_wb),
		.core_req_warp_num (VX_dcache_req_smem.core_req_warp_num),
		.core_req_pc       (VX_dcache_req_smem.core_req_pc),

		// Delay Core Req
		.delay_req         (VX_dcache_rsp_smem.delay_req),

		// Core Cache Can't WB
		.core_no_wb_slot   (VX_dcache_req_smem.core_no_wb_slot),

		// Cache CWB
		.core_wb_valid     (VX_dcache_rsp_smem.core_wb_valid),
		.core_wb_req_rd    (VX_dcache_rsp_smem.core_wb_req_rd),
		.core_wb_req_wb    (VX_dcache_rsp_smem.core_wb_req_wb),
		.core_wb_warp_num  (VX_dcache_rsp_smem.core_wb_warp_num),
		.core_wb_readdata  (VX_dcache_rsp_smem.core_wb_readdata),
		.core_wb_pc        (VX_dcache_rsp_smem.core_wb_pc),

		// DRAM response
		.dram_fill_rsp     (VX_gpu_smem_dram_res.dram_fill_rsp),
		.dram_fill_rsp_addr(VX_gpu_smem_dram_res.dram_fill_rsp_addr),
		.dram_fill_rsp_data(VX_gpu_smem_dram_res.dram_fill_rsp_data),

		// DRAM accept response
		.dram_fill_accept  (VX_gpu_smem_dram_req.dram_fill_accept),

		// DRAM Req
		.dram_req          (VX_gpu_smem_dram_req.dram_req),
		.dram_req_write    (VX_gpu_smem_dram_req.dram_req_write),
		.dram_req_read     (VX_gpu_smem_dram_req.dram_req_read),
		.dram_req_addr     (VX_gpu_smem_dram_req.dram_req_addr),
		.dram_req_size     (VX_gpu_smem_dram_req.dram_req_size),
		.dram_req_data     (VX_gpu_smem_dram_req.dram_req_data),

		// Snoop Response
		.dram_req_because_of_wb(VX_gpu_smem_dram_req.dram_because_of_snp),
		.dram_snp_full         (VX_gpu_smem_dram_req.dram_snp_full),

		// Snoop Request
		.snp_req               (0),
		.snp_req_addr          (0),

		// LLVQ stuff
		.llvq_pop              (Sllvq_pop),
		.llvq_valid            (Sllvq_valid),
		.llvq_res_addr         (Sllvq_res_addr),
		.llvq_res_data         (Sllvq_res_data)
		);


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
		.FUNC_ID                      (`DFUNC_ID),
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
		.core_req_valid    (VX_dcache_req_dcache.core_req_valid),
		.core_req_addr     (VX_dcache_req_dcache.core_req_addr),
		.core_req_writedata(VX_dcache_req_dcache.core_req_writedata),
		.core_req_mem_read (VX_dcache_req_dcache.core_req_mem_read),
		.core_req_mem_write(VX_dcache_req_dcache.core_req_mem_write),
		.core_req_rd       (VX_dcache_req_dcache.core_req_rd),
		.core_req_wb       (VX_dcache_req_dcache.core_req_wb),
		.core_req_warp_num (VX_dcache_req_dcache.core_req_warp_num),
		.core_req_pc       (VX_dcache_req_dcache.core_req_pc),

		// Delay Core Req
		.delay_req         (VX_dcache_rsp_dcache.delay_req),

		// Core Cache Can't WB
		.core_no_wb_slot   (VX_dcache_req_dcache.core_no_wb_slot),

		// Cache CWB
		.core_wb_valid     (VX_dcache_rsp_dcache.core_wb_valid),
		.core_wb_req_rd    (VX_dcache_rsp_dcache.core_wb_req_rd),
		.core_wb_req_wb    (VX_dcache_rsp_dcache.core_wb_req_wb),
		.core_wb_warp_num  (VX_dcache_rsp_dcache.core_wb_warp_num),
		.core_wb_readdata  (VX_dcache_rsp_dcache.core_wb_readdata),
		.core_wb_pc        (VX_dcache_rsp_dcache.core_wb_pc),

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



    wire                                                    Illvq_pop;
    wire[`DNUMBER_REQUESTS-1:0]                             Illvq_valid;
    wire[`DNUMBER_REQUESTS-1:0][31:0]                       Illvq_res_addr;
    wire[`DNUMBER_REQUESTS-1:0][`DBANK_LINE_SIZE_RNG][31:0] Illvq_res_data;
    assign Illvq_pop = 0;
	VX_cache #(
		.CACHE_SIZE_BYTES             (`ICACHE_SIZE_BYTES),
		.BANK_LINE_SIZE_BYTES         (`IBANK_LINE_SIZE_BYTES),
		.NUMBER_BANKS                 (`INUMBER_BANKS),
		.WORD_SIZE_BYTES              (`IWORD_SIZE_BYTES),
		.NUMBER_REQUESTS              (`INUMBER_REQUESTS),
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
		.llvq_pop              (Illvq_pop),
		.llvq_valid            (Illvq_valid),
		.llvq_res_addr         (Illvq_res_addr),
		.llvq_res_data         (Illvq_res_data)
		);



endmodule
