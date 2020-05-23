`include "VX_define.vh"

module VX_dmem_ctrl # (
    parameter CORE_ID = 0
) (
    input wire              clk,
    input wire              reset,

    // Core <-> Dcache    
    VX_cache_core_req_if    dcache_core_req_if,
    VX_cache_core_rsp_if    dcache_core_rsp_if,

    // Dram <-> Dcache
    VX_cache_dram_req_if    dcache_dram_req_if,
    VX_cache_dram_rsp_if    dcache_dram_rsp_if,
    VX_cache_snp_req_if     dcache_snp_req_if,
    VX_cache_snp_rsp_if     dcache_snp_rsp_if,

    // Core <-> Icache    
    VX_cache_core_req_if    icache_core_req_if,  
    VX_cache_core_rsp_if    icache_core_rsp_if,

    // Dram <-> Icache
    VX_cache_dram_req_if    icache_dram_req_if,
    VX_cache_dram_rsp_if    icache_dram_rsp_if
);
    VX_cache_core_req_if #(
        .NUM_REQUESTS(`DNUM_REQUESTS), 
        .WORD_SIZE(`DWORD_SIZE), 
        .CORE_TAG_WIDTH(`CORE_REQ_TAG_WIDTH),
        .CORE_TAG_ID_BITS(`CORE_TAG_ID_BITS)
    ) dcache_core_req_qual_if(), smem_core_req_if();

    VX_cache_core_rsp_if #(
        .NUM_REQUESTS(`DNUM_REQUESTS), 
        .WORD_SIZE(`DWORD_SIZE), 
        .CORE_TAG_WIDTH(`CORE_REQ_TAG_WIDTH),
        .CORE_TAG_ID_BITS(`CORE_TAG_ID_BITS)
    ) dcache_core_rsp_qual_if(), smem_core_rsp_if();

    // use "case equality" to handle uninitialized entry
    wire smem_select = ((dcache_core_req_if.core_req_addr[0][31:24] == `SHARED_MEM_TOP_ADDR) === 1'b1);

    VX_dcache_io_arb dcache_io_arb (
        .io_select          (smem_select),
        .core_req_if        (dcache_core_req_if),
        .dcache_core_req_if (dcache_core_req_qual_if),
        .io_core_req_if     (smem_core_req_if),    
        .dcache_core_rsp_if (dcache_core_rsp_qual_if),
        .io_core_rsp_if     (smem_core_rsp_if),    
        .core_rsp_if        (dcache_core_rsp_if)
    );

    VX_cache #(
        .CACHE_ID               (`SCACHE_ID),
        .CACHE_SIZE             (`SCACHE_SIZE),
        .BANK_LINE_SIZE         (`SBANK_LINE_SIZE),
        .NUM_BANKS              (`SNUM_BANKS),
        .WORD_SIZE              (`SWORD_SIZE),
        .NUM_REQUESTS           (`SNUM_REQUESTS),
        .STAGE_1_CYCLES         (`SSTAGE_1_CYCLES),
        .REQQ_SIZE              (`SREQQ_SIZE),
        .MRVQ_SIZE              (`SMRVQ_SIZE),
        .DFPQ_SIZE              (`SDFPQ_SIZE),
        .SNRQ_SIZE              (0),
        .CWBQ_SIZE              (`SCWBQ_SIZE),
        .DWBQ_SIZE              (`SDWBQ_SIZE),
        .DFQQ_SIZE              (`SDFQQ_SIZE),
        .PRFQ_SIZE              (`SPRFQ_SIZE),
        .PRFQ_STRIDE            (`SPRFQ_STRIDE),
        .FILL_INVALIDAOR_SIZE   (`SFILL_INVALIDAOR_SIZE),
        .SNOOP_FORWARDING       (0),
        .DRAM_ENABLE            (0),
        .WRITE_ENABLE           (1),
        .CORE_TAG_WIDTH         (`CORE_REQ_TAG_WIDTH),
        .CORE_TAG_ID_BITS       (`CORE_TAG_ID_BITS),
        .DRAM_TAG_WIDTH         (`SDRAM_TAG_WIDTH)
    ) gpu_smem (
        .clk                (clk),
        .reset              (reset),

        // Core request
        .core_req_valid     (smem_core_req_if.core_req_valid),
        .core_req_read      (smem_core_req_if.core_req_read),
        .core_req_write     (smem_core_req_if.core_req_write),
        .core_req_addr      (smem_core_req_if.core_req_addr),
        .core_req_data      (smem_core_req_if.core_req_data),        
        .core_req_tag       (smem_core_req_if.core_req_tag),
        .core_req_ready     (smem_core_req_if.core_req_ready),

        // Core response
        .core_rsp_valid     (smem_core_rsp_if.core_rsp_valid),
        .core_rsp_data      (smem_core_rsp_if.core_rsp_data),
        .core_rsp_tag       (smem_core_rsp_if.core_rsp_tag),
        .core_rsp_ready     (smem_core_rsp_if.core_rsp_ready),

        // DRAM request
        `UNUSED_PIN (dram_req_read),
        `UNUSED_PIN (dram_req_write),        
        `UNUSED_PIN (dram_req_addr),
        `UNUSED_PIN (dram_req_data),
        `UNUSED_PIN (dram_req_tag),
        .dram_req_ready     (1'b0),       

        // DRAM response
        .dram_rsp_valid     (1'b0),
        .dram_rsp_data      (0),
        .dram_rsp_tag       (`SDRAM_TAG_WIDTH'(0)),
        `UNUSED_PIN (dram_rsp_ready),

        // Snoop request
        .snp_req_valid      (1'b0),
        .snp_req_addr       (0),
        .snp_req_tag        (0),
        `UNUSED_PIN (snp_req_ready),

        // Snoop response
        `UNUSED_PIN (snp_rsp_valid),
        `UNUSED_PIN (snp_rsp_tag),
        .snp_rsp_ready      (1'b0),

        // Snoop forward out
        `UNUSED_PIN (snp_fwdout_valid),
        `UNUSED_PIN (snp_fwdout_addr),    
        `UNUSED_PIN (snp_fwdout_tag),    
        .snp_fwdout_ready   (0),

         // Snoop forward in
        .snp_fwdin_valid    (0),
        .snp_fwdin_tag      (0),    
        `UNUSED_PIN (snp_fwdin_ready)
    );

    VX_cache #(
        .CACHE_ID               (`DCACHE_ID),
        .CACHE_SIZE             (`DCACHE_SIZE),
        .BANK_LINE_SIZE         (`DBANK_LINE_SIZE),
        .NUM_BANKS              (`DNUM_BANKS),
        .WORD_SIZE              (`DWORD_SIZE),
        .NUM_REQUESTS           (`DNUM_REQUESTS),
        .STAGE_1_CYCLES         (`DSTAGE_1_CYCLES),
        .REQQ_SIZE              (`DREQQ_SIZE),
        .MRVQ_SIZE              (`DMRVQ_SIZE),
        .DFPQ_SIZE              (`DDFPQ_SIZE),
        .SNRQ_SIZE              (`DSNRQ_SIZE),
        .CWBQ_SIZE              (`DCWBQ_SIZE),
        .DWBQ_SIZE              (`DDWBQ_SIZE),
        .DFQQ_SIZE              (`DDFQQ_SIZE),
        .PRFQ_SIZE              (`DPRFQ_SIZE),
        .PRFQ_STRIDE            (`DPRFQ_STRIDE),
        .FILL_INVALIDAOR_SIZE   (`DFILL_INVALIDAOR_SIZE),
        .SNOOP_FORWARDING       (0),
        .DRAM_ENABLE            (1),
        .WRITE_ENABLE           (1),
        .CORE_TAG_WIDTH         (`CORE_REQ_TAG_WIDTH),
        .CORE_TAG_ID_BITS       (`CORE_TAG_ID_BITS),
        .DRAM_TAG_WIDTH         (`DDRAM_TAG_WIDTH),
        .SNP_REQ_TAG_WIDTH      (`DSNP_TAG_WIDTH)
    ) gpu_dcache (
        .clk                (clk),
        .reset              (reset),

        // Core req
        .core_req_valid     (dcache_core_req_qual_if.core_req_valid),
        .core_req_read      (dcache_core_req_qual_if.core_req_read),
        .core_req_write     (dcache_core_req_qual_if.core_req_write),
        .core_req_addr      (dcache_core_req_qual_if.core_req_addr),
        .core_req_data      (dcache_core_req_qual_if.core_req_data),        
        .core_req_tag       (dcache_core_req_qual_if.core_req_tag),
        .core_req_ready     (dcache_core_req_qual_if.core_req_ready),

        // Core response
        .core_rsp_valid     (dcache_core_rsp_qual_if.core_rsp_valid),
        .core_rsp_data      (dcache_core_rsp_qual_if.core_rsp_data),
        .core_rsp_tag       (dcache_core_rsp_qual_if.core_rsp_tag),
        .core_rsp_ready     (dcache_core_rsp_qual_if.core_rsp_ready),

        // DRAM request
        .dram_req_read      (dcache_dram_req_if.dram_req_read),
        .dram_req_write     (dcache_dram_req_if.dram_req_write),        
        .dram_req_addr      (dcache_dram_req_if.dram_req_addr),
        .dram_req_data      (dcache_dram_req_if.dram_req_data),
        .dram_req_tag       (dcache_dram_req_if.dram_req_tag),
        .dram_req_ready     (dcache_dram_req_if.dram_req_ready),

        // DRAM response
        .dram_rsp_valid     (dcache_dram_rsp_if.dram_rsp_valid),        
        .dram_rsp_data      (dcache_dram_rsp_if.dram_rsp_data),
        .dram_rsp_tag       (dcache_dram_rsp_if.dram_rsp_tag),
        .dram_rsp_ready     (dcache_dram_rsp_if.dram_rsp_ready),

        // Snoop request
        .snp_req_valid      (dcache_snp_req_if.snp_req_valid),
        .snp_req_addr       (dcache_snp_req_if.snp_req_addr),
        .snp_req_tag        (dcache_snp_req_if.snp_req_tag),
        .snp_req_ready      (dcache_snp_req_if.snp_req_ready),

        // Snoop response
        .snp_rsp_valid      (dcache_snp_rsp_if.snp_rsp_valid),
        .snp_rsp_tag        (dcache_snp_rsp_if.snp_rsp_tag),
        .snp_rsp_ready      (dcache_snp_rsp_if.snp_rsp_ready),
        
        // Snoop forward out
        `UNUSED_PIN (snp_fwdout_valid),
        `UNUSED_PIN (snp_fwdout_addr),    
        `UNUSED_PIN (snp_fwdout_tag),    
        .snp_fwdout_ready   (0),

         // Snoop forward in
        .snp_fwdin_valid    (0),
        .snp_fwdin_tag      (0),    
        `UNUSED_PIN (snp_fwdin_ready)
    );

    VX_cache #(
        .CACHE_ID               (`ICACHE_ID),
        .CACHE_SIZE             (`ICACHE_SIZE),
        .BANK_LINE_SIZE         (`IBANK_LINE_SIZE),
        .NUM_BANKS              (`INUM_BANKS),
        .WORD_SIZE              (`IWORD_SIZE),
        .NUM_REQUESTS           (`INUM_REQUESTS),
        .STAGE_1_CYCLES         (`ISTAGE_1_CYCLES),
        .REQQ_SIZE              (`IREQQ_SIZE),
        .MRVQ_SIZE              (`IMRVQ_SIZE),
        .DFPQ_SIZE              (`IDFPQ_SIZE),
        .SNRQ_SIZE              (0),
        .CWBQ_SIZE              (`ICWBQ_SIZE),
        .DWBQ_SIZE              (`IDWBQ_SIZE),
        .DFQQ_SIZE              (`IDFQQ_SIZE),
        .PRFQ_SIZE              (`IPRFQ_SIZE),
        .PRFQ_STRIDE            (`IPRFQ_STRIDE),
        .FILL_INVALIDAOR_SIZE   (`IFILL_INVALIDAOR_SIZE),
        .SNOOP_FORWARDING       (0),
        .DRAM_ENABLE            (1),
        .WRITE_ENABLE           (0),
        .CORE_TAG_WIDTH         (`CORE_REQ_TAG_WIDTH),
        .CORE_TAG_ID_BITS       (`CORE_TAG_ID_BITS),
        .DRAM_TAG_WIDTH         (`IDRAM_TAG_WIDTH)
    ) gpu_icache (
        .clk                   (clk),
        .reset                 (reset),

        // Core request
        .core_req_valid        (icache_core_req_if.core_req_valid),
        .core_req_read         (icache_core_req_if.core_req_read),
        .core_req_write        (icache_core_req_if.core_req_write),
        .core_req_addr         (icache_core_req_if.core_req_addr),
        .core_req_data         (icache_core_req_if.core_req_data),        
        .core_req_tag          (icache_core_req_if.core_req_tag),
        .core_req_ready        (icache_core_req_if.core_req_ready),

        // Core response
        .core_rsp_valid        (icache_core_rsp_if.core_rsp_valid),
        .core_rsp_data         (icache_core_rsp_if.core_rsp_data),
        .core_rsp_tag          (icache_core_rsp_if.core_rsp_tag),
        .core_rsp_ready        (icache_core_rsp_if.core_rsp_ready),

        // DRAM Req
        .dram_req_read         (icache_dram_req_if.dram_req_read),
        .dram_req_write        (icache_dram_req_if.dram_req_write),        
        .dram_req_addr         (icache_dram_req_if.dram_req_addr),
        .dram_req_data         (icache_dram_req_if.dram_req_data),
        .dram_req_tag          (icache_dram_req_if.dram_req_tag),
        .dram_req_ready        (icache_dram_req_if.dram_req_ready),        

        // DRAM response
        .dram_rsp_valid        (icache_dram_rsp_if.dram_rsp_valid),        
        .dram_rsp_data         (icache_dram_rsp_if.dram_rsp_data),
        .dram_rsp_tag          (icache_dram_rsp_if.dram_rsp_tag),
        .dram_rsp_ready        (icache_dram_rsp_if.dram_rsp_ready),

        // Snoop request
        .snp_req_valid         (1'b0),
        .snp_req_addr          (0),
        .snp_req_tag           (0),
        `UNUSED_PIN (snp_req_ready),

        // Snoop response
        `UNUSED_PIN (snp_rsp_valid),
        `UNUSED_PIN (snp_rsp_tag),
        .snp_rsp_ready         (1'b0),

        // Snoop forward out
        `UNUSED_PIN (snp_fwdout_valid),
        `UNUSED_PIN (snp_fwdout_addr),    
        `UNUSED_PIN (snp_fwdout_tag),    
        .snp_fwdout_ready      (0),

         // Snoop forward in
        .snp_fwdin_valid       (0),
        .snp_fwdin_tag         (0),    
        `UNUSED_PIN (snp_fwdin_ready)
    );

endmodule
