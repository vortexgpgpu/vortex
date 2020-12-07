`include "VX_define.vh"

module VX_mem_unit # (
    parameter CORE_ID = 0
) (
    `SCOPE_IO_VX_mem_unit

    input wire              clk,
    input wire              reset,

    // Core <-> Dcache    
    VX_cache_core_req_if    core_dcache_req_if,
    VX_cache_core_rsp_if    core_dcache_rsp_if,
    
    // Core <-> Icache    
    VX_cache_core_req_if    core_icache_req_if,  
    VX_cache_core_rsp_if    core_icache_rsp_if,

    // Dcache Snoop
    VX_cache_snp_req_if     dcache_snp_req_if,
    VX_cache_snp_rsp_if     dcache_snp_rsp_if,

    // DRAM
    VX_cache_dram_req_if    dram_req_if,
    VX_cache_dram_rsp_if    dram_rsp_if,    

    // I/O
    VX_cache_core_req_if    io_req_if,
    VX_cache_core_rsp_if    io_rsp_if
);
    VX_cache_dram_req_if #(
        .DRAM_LINE_WIDTH (`DDRAM_LINE_WIDTH),
        .DRAM_ADDR_WIDTH (`DDRAM_ADDR_WIDTH),
        .DRAM_TAG_WIDTH  (`DDRAM_TAG_WIDTH)
    ) dcache_dram_req_if(), icache_dram_req_if();

    VX_cache_dram_rsp_if #(
        .DRAM_LINE_WIDTH (`DDRAM_LINE_WIDTH),
        .DRAM_TAG_WIDTH  (`DDRAM_TAG_WIDTH)
    ) dcache_dram_rsp_if(), icache_dram_rsp_if();

    VX_cache_core_req_if #(
        .NUM_REQS         (`DNUM_REQUESTS), 
        .WORD_SIZE        (`DWORD_SIZE), 
        .CORE_TAG_WIDTH   (`DCORE_TAG_WIDTH),
        .CORE_TAG_ID_BITS (`DCORE_TAG_ID_BITS)
    ) dcache_req_if();

    VX_cache_core_rsp_if #(
        .NUM_REQS         (`DNUM_REQUESTS), 
        .WORD_SIZE        (`DWORD_SIZE), 
        .CORE_TAG_WIDTH   (`DCORE_TAG_WIDTH),
        .CORE_TAG_ID_BITS (`DCORE_TAG_ID_BITS)
    ) dcache_rsp_if();

    VX_cache_core_req_if #(
        .NUM_REQS         (`DNUM_REQUESTS), 
        .WORD_SIZE        (`DWORD_SIZE), 
        .CORE_TAG_WIDTH   (`DCORE_TAG_WIDTH),
        .CORE_TAG_ID_BITS (`DCORE_TAG_ID_BITS)
    ) smem_req_if();

    VX_cache_core_rsp_if #(
        .NUM_REQS         (`DNUM_REQUESTS), 
        .WORD_SIZE        (`DWORD_SIZE), 
        .CORE_TAG_WIDTH   (`DCORE_TAG_WIDTH),
        .CORE_TAG_ID_BITS (`DCORE_TAG_ID_BITS)
    ) smem_rsp_if();

    VX_dcache_arb dcache_arb (   
        .clk            (clk),
        .reset          (reset),

        .core_req_if    (core_dcache_req_if),
        .cache_req_if   (dcache_req_if),
        .smem_req_if    (smem_req_if),  
        .io_req_if      (io_req_if),  

        .cache_rsp_if   (dcache_rsp_if),
        .smem_rsp_if    (smem_rsp_if),  
        .io_rsp_if      (io_rsp_if),          
        .core_rsp_if    (core_dcache_rsp_if)
    );

    VX_cache #(
        .CACHE_ID           (`DCACHE_ID),
        .CACHE_SIZE         (`DCACHE_SIZE),
        .BANK_LINE_SIZE     (`DBANK_LINE_SIZE),
        .NUM_BANKS          (`DNUM_BANKS),
        .WORD_SIZE          (`DWORD_SIZE),
        .NUM_REQS           (`DNUM_REQUESTS),
        .CREQ_SIZE          (`DCREQ_SIZE),
        .MSHR_SIZE          (`DMSHR_SIZE),
        .DRPQ_SIZE          (`DDRPQ_SIZE),
        .SNRQ_SIZE          (`DSNRQ_SIZE),
        .CWBQ_SIZE          (`DCWBQ_SIZE),
        .DREQ_SIZE          (`DDREQ_SIZE),        
        .SNPQ_SIZE          (`DSNPQ_SIZE),
        .DRAM_ENABLE        (1),
        .FLUSH_ENABLE       (1),
        .WRITE_ENABLE       (1),
        .CORE_TAG_WIDTH     (`DCORE_TAG_WIDTH),
        .CORE_TAG_ID_BITS   (`DCORE_TAG_ID_BITS),
        .DRAM_TAG_WIDTH     (`DDRAM_TAG_WIDTH),
        .SNP_TAG_WIDTH      (`DSNP_TAG_WIDTH)
    ) dcache (
        `SCOPE_BIND_VX_mem_unit_dcache
        
        .clk                (clk),
        .reset              (reset),

        // Core req
        .core_req_valid     (dcache_req_if.valid),
        .core_req_rw        (dcache_req_if.rw),
        .core_req_byteen    (dcache_req_if.byteen),
        .core_req_addr      (dcache_req_if.addr),
        .core_req_data      (dcache_req_if.data),        
        .core_req_tag       (dcache_req_if.tag),
        .core_req_ready     (dcache_req_if.ready),

        // Core response
        .core_rsp_valid     (dcache_rsp_if.valid),
        .core_rsp_data      (dcache_rsp_if.data),
        .core_rsp_tag       (dcache_rsp_if.tag),
        .core_rsp_ready     (dcache_rsp_if.ready),

        // DRAM request
        .dram_req_valid     (dcache_dram_req_if.valid),
        .dram_req_rw        (dcache_dram_req_if.rw),        
        .dram_req_byteen    (dcache_dram_req_if.byteen),        
        .dram_req_addr      (dcache_dram_req_if.addr),
        .dram_req_data      (dcache_dram_req_if.data),
        .dram_req_tag       (dcache_dram_req_if.tag),
        .dram_req_ready     (dcache_dram_req_if.ready),

        // DRAM response
        .dram_rsp_valid     (dcache_dram_rsp_if.valid),        
        .dram_rsp_data      (dcache_dram_rsp_if.data),
        .dram_rsp_tag       (dcache_dram_rsp_if.tag),
        .dram_rsp_ready     (dcache_dram_rsp_if.ready),

        // Snoop request
        .snp_req_valid      (dcache_snp_req_if.valid),
        .snp_req_addr       (dcache_snp_req_if.addr),
        .snp_req_inv        (dcache_snp_req_if.invalidate),
        .snp_req_tag        (dcache_snp_req_if.tag),
        .snp_req_ready      (dcache_snp_req_if.ready),

        // Snoop response
        .snp_rsp_valid      (dcache_snp_rsp_if.valid),
        .snp_rsp_tag        (dcache_snp_rsp_if.tag),
        .snp_rsp_ready      (dcache_snp_rsp_if.ready),

        // Miss status
        `UNUSED_PIN (miss_vec)
    );

    VX_cache #(
        .CACHE_ID           (`ICACHE_ID),
        .CACHE_SIZE         (`ICACHE_SIZE),
        .BANK_LINE_SIZE     (`IBANK_LINE_SIZE),
        .NUM_BANKS          (`INUM_BANKS),
        .WORD_SIZE          (`IWORD_SIZE),
        .NUM_REQS           (`INUM_REQUESTS),
        .CREQ_SIZE          (`ICREQ_SIZE),
        .MSHR_SIZE          (`IMSHR_SIZE),
        .DRPQ_SIZE          (`IDRPQ_SIZE),
        .SNRQ_SIZE          (1),
        .CWBQ_SIZE          (`ICWBQ_SIZE),
        .DREQ_SIZE          (`IDREQ_SIZE),
        .SNPQ_SIZE          (1),
        .DRAM_ENABLE        (1),
        .FLUSH_ENABLE       (0),
        .WRITE_ENABLE       (0),
        .CORE_TAG_WIDTH     (`ICORE_TAG_WIDTH),
        .CORE_TAG_ID_BITS   (`ICORE_TAG_ID_BITS),
        .DRAM_TAG_WIDTH     (`IDRAM_TAG_WIDTH)
    ) icache (
        `SCOPE_BIND_VX_mem_unit_icache

        .clk                (clk),
        .reset              (reset),

        // Core request
        .core_req_valid     (core_icache_req_if.valid),
        .core_req_rw        (core_icache_req_if.rw),
        .core_req_byteen    (core_icache_req_if.byteen),
        .core_req_addr      (core_icache_req_if.addr),
        .core_req_data      (core_icache_req_if.data),        
        .core_req_tag       (core_icache_req_if.tag),
        .core_req_ready     (core_icache_req_if.ready),

        // Core response
        .core_rsp_valid     (core_icache_rsp_if.valid),
        .core_rsp_data      (core_icache_rsp_if.data),
        .core_rsp_tag       (core_icache_rsp_if.tag),
        .core_rsp_ready     (core_icache_rsp_if.ready),

        // DRAM Req
        .dram_req_valid     (icache_dram_req_if.valid),
        .dram_req_rw        (icache_dram_req_if.rw),        
        .dram_req_byteen    (icache_dram_req_if.byteen),        
        .dram_req_addr      (icache_dram_req_if.addr),
        .dram_req_data      (icache_dram_req_if.data),
        .dram_req_tag       (icache_dram_req_if.tag),
        .dram_req_ready     (icache_dram_req_if.ready),        

        // DRAM response
        .dram_rsp_valid     (icache_dram_rsp_if.valid),        
        .dram_rsp_data      (icache_dram_rsp_if.data),
        .dram_rsp_tag       (icache_dram_rsp_if.tag),
        .dram_rsp_ready     (icache_dram_rsp_if.ready),

        // Snoop request
        .snp_req_valid      (1'b0),
        .snp_req_addr       (0),
        .snp_req_inv        (1'b0),
        .snp_req_tag        (0),
        `UNUSED_PIN (snp_req_ready),

        // Snoop response
        `UNUSED_PIN (snp_rsp_valid),
        `UNUSED_PIN (snp_rsp_tag),
        .snp_rsp_ready         (1'b0),

        // Miss status
        `UNUSED_PIN (miss_vec)
    );    

    VX_cache #(
        .CACHE_ID           (`SCACHE_ID),
        .CACHE_SIZE         (`SMEM_SIZE),
        .BANK_LINE_SIZE     (`SBANK_LINE_SIZE),
        .NUM_BANKS          (`SNUM_BANKS),
        .WORD_SIZE          (`SWORD_SIZE),
        .NUM_REQS           (`SNUM_REQUESTS),
        .CREQ_SIZE          (`SCREQ_SIZE),
        .MSHR_SIZE          (8),
        .DRPQ_SIZE          (1),
        .SNRQ_SIZE          (1),
        .CWBQ_SIZE          (`SCWBQ_SIZE),
        .DREQ_SIZE          (1),
        .SNPQ_SIZE          (1),        
        .DRAM_ENABLE        (0),
        .FLUSH_ENABLE       (0),
        .WRITE_ENABLE       (1),
        .CORE_TAG_WIDTH     (`DCORE_TAG_WIDTH),
        .CORE_TAG_ID_BITS   (`DCORE_TAG_ID_BITS),
        .DRAM_TAG_WIDTH     (`SDRAM_TAG_WIDTH)
    ) smem (
        `SCOPE_BIND_VX_mem_unit_smem
        
        .clk                (clk),
        .reset              (reset),

        // Core request
        .core_req_valid     (smem_req_if.valid),
        .core_req_rw        (smem_req_if.rw),
        .core_req_byteen    (smem_req_if.byteen),
        .core_req_addr      (smem_req_if.addr),
        .core_req_data      (smem_req_if.data),        
        .core_req_tag       (smem_req_if.tag),
        .core_req_ready     (smem_req_if.ready),

        // Core response
        .core_rsp_valid     (smem_rsp_if.valid),
        .core_rsp_data      (smem_rsp_if.data),
        .core_rsp_tag       (smem_rsp_if.tag),
        .core_rsp_ready     (smem_rsp_if.ready),

        // DRAM request
        `UNUSED_PIN (dram_req_valid),
        `UNUSED_PIN (dram_req_rw),        
        `UNUSED_PIN (dram_req_byteen),        
        `UNUSED_PIN (dram_req_addr),
        `UNUSED_PIN (dram_req_data),
        `UNUSED_PIN (dram_req_tag),
        .dram_req_ready     (1'b0),       

        // DRAM response
        .dram_rsp_valid     (0),
        .dram_rsp_data      (0),
        .dram_rsp_tag       (0),
        `UNUSED_PIN (dram_rsp_ready),

        // Snoop request
        .snp_req_valid      (1'b0),
        .snp_req_addr       (0),
        .snp_req_inv        (0),
        .snp_req_tag        (0),
        `UNUSED_PIN (snp_req_ready),

        // Snoop response
        `UNUSED_PIN (snp_rsp_valid),
        `UNUSED_PIN (snp_rsp_tag),
        .snp_rsp_ready      (1'b0),

        // Miss status
        `UNUSED_PIN (miss_vec)
    );

    VX_mem_arb #(
        .NUM_REQS      (2),
        .DATA_WIDTH    (`DDRAM_LINE_WIDTH),
        .ADDR_WIDTH    (`DDRAM_ADDR_WIDTH),
        .TAG_IN_WIDTH  (`DDRAM_TAG_WIDTH),
        .TAG_OUT_WIDTH (`XDRAM_TAG_WIDTH)
    ) dram_arb (
        .clk            (clk),
        .reset          (reset),

        // Source request
        .req_valid_in   ({dcache_dram_req_if.valid,   icache_dram_req_if.valid}),
        .req_rw_in      ({dcache_dram_req_if.rw,      icache_dram_req_if.rw}),
        .req_byteen_in  ({dcache_dram_req_if.byteen,  icache_dram_req_if.byteen}),
        .req_addr_in    ({dcache_dram_req_if.addr,    icache_dram_req_if.addr}),
        .req_data_in    ({dcache_dram_req_if.data,    icache_dram_req_if.data}),  
        .req_tag_in     ({dcache_dram_req_if.tag,     icache_dram_req_if.tag}),  
        .req_ready_in   ({dcache_dram_req_if.ready,   icache_dram_req_if.ready}),

        // DRAM request
        .req_valid_out  (dram_req_if.valid),
        .req_rw_out     (dram_req_if.rw),        
        .req_byteen_out (dram_req_if.byteen),        
        .req_addr_out   (dram_req_if.addr),
        .req_data_out   (dram_req_if.data),
        .req_tag_out    (dram_req_if.tag),
        .req_ready_out  (dram_req_if.ready),

        // Source response
        .rsp_valid_out  ({dcache_dram_rsp_if.valid,   icache_dram_rsp_if.valid}),
        .rsp_data_out   ({dcache_dram_rsp_if.data,    icache_dram_rsp_if.data}),
        .rsp_tag_out    ({dcache_dram_rsp_if.tag,     icache_dram_rsp_if.tag}),
        .rsp_ready_out  ({dcache_dram_rsp_if.ready,   icache_dram_rsp_if.ready}),
        
        // DRAM response
        .rsp_valid_in   (dram_rsp_if.valid),
        .rsp_tag_in     (dram_rsp_if.tag),
        .rsp_data_in    (dram_rsp_if.data),
        .rsp_ready_in   (dram_rsp_if.ready)
    );

endmodule
