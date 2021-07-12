`include "VX_define.vh"

module VX_core #( 
    parameter CORE_ID = 0
) (        
    `SCOPE_IO_VX_core
    
    // Clock
    input  wire                             clk,
    input  wire                             reset,

    // Memory request
    output wire                             mem_req_valid,
    output wire                             mem_req_rw,    
    output wire [`DMEM_BYTEEN_WIDTH-1:0]    mem_req_byteen,
    output wire [`DMEM_ADDR_WIDTH-1:0]      mem_req_addr,
    output wire [`DMEM_LINE_WIDTH-1:0]      mem_req_data,
    output wire [`XMEM_TAG_WIDTH-1:0]       mem_req_tag,
    input  wire                             mem_req_ready,

    // Memory reponse    
    input  wire                             mem_rsp_valid,
    input  wire [`DMEM_LINE_WIDTH-1:0]      mem_rsp_data,
    input  wire [`XMEM_TAG_WIDTH-1:0]       mem_rsp_tag,
    output wire                             mem_rsp_ready,

    // Status
    output wire                             busy
);
`ifdef PERF_ENABLE
    VX_perf_memsys_if perf_memsys_if();
`endif

    VX_cache_mem_req_if #(
        .MEM_LINE_WIDTH(`DMEM_LINE_WIDTH),
        .MEM_ADDR_WIDTH(`DMEM_ADDR_WIDTH),
        .MEM_TAG_WIDTH(`XMEM_TAG_WIDTH)
    ) mem_req_if();

    VX_cache_mem_rsp_if #(
        .MEM_LINE_WIDTH(`DMEM_LINE_WIDTH),
        .MEM_TAG_WIDTH(`XMEM_TAG_WIDTH)
    ) mem_rsp_if();

    assign mem_req_valid = mem_req_if.valid;
    assign mem_req_rw    = mem_req_if.rw;
    assign mem_req_byteen= mem_req_if.byteen;
    assign mem_req_addr  = mem_req_if.addr;
    assign mem_req_data  = mem_req_if.data;
    assign mem_req_tag   = mem_req_if.tag;
    assign mem_req_if.ready = mem_req_ready;

    assign mem_rsp_if.valid = mem_rsp_valid;
    assign mem_rsp_if.data  = mem_rsp_data;
    assign mem_rsp_if.tag   = mem_rsp_tag;
    assign mem_rsp_ready = mem_rsp_if.ready;

    //--

    VX_dcache_core_req_if #(
        .NUM_REQS(`DNUM_REQS), 
        .WORD_SIZE(`DWORD_SIZE), 
        .CORE_TAG_WIDTH(`DCORE_TAG_WIDTH)
    ) dcache_core_req_if();

    VX_dcache_core_rsp_if #(
        .NUM_REQS(`DNUM_REQS), 
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
        .dcache_rsp_tmask   (dcache_core_rsp_if.tmask),
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

        // Status
        .busy(busy)
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

        // Memory
        .mem_req_if         (mem_req_if),
        .mem_rsp_if         (mem_rsp_if)
    );
    
endmodule





