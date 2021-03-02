`include "VX_define.vh"

module VX_core #( 
    parameter CORE_ID = 0
) (        
    `SCOPE_IO_VX_core
    
    // Clock
    input  wire                             clk,
    input  wire                             reset,

    // DRAM request
    output wire                             dram_req_valid,
    output wire                             dram_req_rw,    
    output wire [`DDRAM_BYTEEN_WIDTH-1:0]   dram_req_byteen,
    output wire [`DDRAM_ADDR_WIDTH-1:0]     dram_req_addr,
    output wire [`DDRAM_LINE_WIDTH-1:0]     dram_req_data,
    output wire [`XDRAM_TAG_WIDTH-1:0]      dram_req_tag,
    input  wire                             dram_req_ready,

    // DRAM reponse    
    input  wire                             dram_rsp_valid,
    input  wire [`DDRAM_LINE_WIDTH-1:0]     dram_rsp_data,
    input  wire [`XDRAM_TAG_WIDTH-1:0]      dram_rsp_tag,
    output wire                             dram_rsp_ready,

    // CSR request
    input  wire                             csr_req_valid,
    input  wire [11:0]                      csr_req_addr,
    input  wire                             csr_req_rw,
    input  wire [31:0]                      csr_req_data,
    output wire                             csr_req_ready,

    // CSR response
    output wire                             csr_rsp_valid,
    output wire [31:0]                      csr_rsp_data,
    input wire                              csr_rsp_ready,

    // Status
    output wire                             busy, 
    output wire                             ebreak
);
`ifdef PERF_ENABLE
    VX_perf_memsys_if perf_memsys_if();
`endif

    VX_cache_dram_req_if #(
        .DRAM_LINE_WIDTH(`DDRAM_LINE_WIDTH),
        .DRAM_ADDR_WIDTH(`DDRAM_ADDR_WIDTH),
        .DRAM_TAG_WIDTH(`XDRAM_TAG_WIDTH)
    ) dram_req_if();

    VX_cache_dram_rsp_if #(
        .DRAM_LINE_WIDTH(`DDRAM_LINE_WIDTH),
        .DRAM_TAG_WIDTH(`XDRAM_TAG_WIDTH)
    ) dram_rsp_if();

    assign dram_req_valid = dram_req_if.valid;
    assign dram_req_rw    = dram_req_if.rw;
    assign dram_req_byteen= dram_req_if.byteen;
    assign dram_req_addr  = dram_req_if.addr;
    assign dram_req_data  = dram_req_if.data;
    assign dram_req_tag   = dram_req_if.tag;
    assign dram_req_if.ready = dram_req_ready;

    assign dram_rsp_if.valid = dram_rsp_valid;
    assign dram_rsp_if.data  = dram_rsp_data;
    assign dram_rsp_if.tag   = dram_rsp_tag;
    assign dram_rsp_ready = dram_rsp_if.ready;

    //--

    VX_dcache_core_req_if #(
        .NUM_REQS(`DNUM_REQUESTS), 
        .WORD_SIZE(`DWORD_SIZE), 
        .CORE_TAG_WIDTH(`DCORE_TAG_WIDTH)
    ) dcache_core_req_if();

    VX_dcache_core_rsp_if #(
        .NUM_REQS(`DNUM_REQUESTS), 
        .WORD_SIZE(`DWORD_SIZE), 
        .CORE_TAG_WIDTH(`DCORE_TAG_WIDTH)
    ) dcache_core_rsp_if();
    
    VX_icache_core_req_if #(
        .WORD_SIZE(`IWORD_SIZE), 
        .CORE_TAG_WIDTH(`ICORE_TAG_WIDTH)
    ) icache_core_req_if();

    VX_icache_core_rsp_if #(
        .WORD_SIZE(`IWORD_SIZE), 
        .CORE_TAG_WIDTH(`ICORE_TAG_WIDTH)
    ) icache_core_rsp_if();
    
    VX_pipeline #(
        .CORE_ID(CORE_ID)
    ) pipeline (
        `SCOPE_BIND_VX_core_pipeline
    `ifdef PERF_ENABLE
        .perf_memsys_if (perf_memsys_if),
    `endif

        .clk(clk),
        .reset(reset),

        // Dcache core request
        .dcache_req_valid   (dcache_core_req_if.valid),
        .dcache_req_rw      (dcache_core_req_if.rw),
        .dcache_req_byteen  (dcache_core_req_if.byteen),
        .dcache_req_addr    (dcache_core_req_if.addr),
        .dcache_req_data    (dcache_core_req_if.data),
        .dcache_req_tag     (dcache_core_req_if.tag),
        .dcache_req_ready   (dcache_core_req_if.ready),

        // Dcache core reponse    
        .dcache_rsp_valid   (dcache_core_rsp_if.valid),
        .dcache_rsp_data    (dcache_core_rsp_if.data),
        .dcache_rsp_tag     (dcache_core_rsp_if.tag),
        .dcache_rsp_ready   (dcache_core_rsp_if.ready),

        // Icache core request
        .icache_req_valid   (icache_core_req_if.valid),
        .icache_req_addr    (icache_core_req_if.addr),
        .icache_req_tag     (icache_core_req_if.tag),
        .icache_req_ready   (icache_core_req_if.ready),

        // Icache core reponse    
        .icache_rsp_valid   (icache_core_rsp_if.valid),
        .icache_rsp_data    (icache_core_rsp_if.data),
        .icache_rsp_tag     (icache_core_rsp_if.tag),
        .icache_rsp_ready   (icache_core_rsp_if.ready),     

        // CSR request
        .csr_req_valid      (csr_req_valid),
        .csr_req_rw         (csr_req_rw),
        .csr_req_addr       (csr_req_addr),
        .csr_req_data       (csr_req_data),
        .csr_req_ready      (csr_req_ready),

        // CSR response
        .csr_rsp_valid      (csr_rsp_valid),            
        .csr_rsp_data       (csr_rsp_data),
        .csr_rsp_ready      (csr_rsp_ready),

        // Status
        .busy(busy), 
        .ebreak(ebreak)
    );  

    //--

    VX_mem_unit #(
        .CORE_ID(CORE_ID)
    ) mem_unit (
        `SCOPE_BIND_VX_core_mem_unit
    `ifdef PERF_ENABLE
        .perf_memsys_if (perf_memsys_if),
    `endif

        .clk                (clk),
        .reset              (reset),

        // Core <-> Dcache
        .dcache_core_req_if (dcache_core_req_if),
        .dcache_core_rsp_if (dcache_core_rsp_if),
        
        // Core <-> Icache
        .icache_core_req_if (icache_core_req_if),
        .icache_core_rsp_if (icache_core_rsp_if),

        // DRAM
        .dram_req_if        (dram_req_if),
        .dram_rsp_if        (dram_rsp_if)
    );
    
endmodule





