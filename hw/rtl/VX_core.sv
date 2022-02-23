`include "VX_define.vh"

module VX_core #( 
    parameter CORE_ID = 0
) (        
    `SCOPE_IO_VX_core
    
    // Clock
    input  wire                             clk,
    input  wire                             reset,

`ifdef EXT_TEX_ENABLE
    VX_tex_csr_if.slave                     tex_csr_if,
`endif

    // Memory request
    output wire                             mem_req_valid,
    output wire                             mem_req_rw,    
    output wire [`DCACHE_MEM_BYTEEN_WIDTH-1:0] mem_req_byteen,
    output wire [`DCACHE_MEM_ADDR_WIDTH-1:0] mem_req_addr,
    output wire [`DCACHE_MEM_DATA_WIDTH-1:0] mem_req_data,
    output wire [`L1_MEM_TAG_WIDTH-1:0]     mem_req_tag,
    input  wire                             mem_req_ready,

    // Memory reponse    
    input  wire                             mem_rsp_valid,
    input  wire [`DCACHE_MEM_DATA_WIDTH-1:0] mem_rsp_data,
    input  wire [`L1_MEM_TAG_WIDTH-1:0]     mem_rsp_tag,
    output wire                             mem_rsp_ready,

    // Status
    output wire                             busy
);
`ifdef PERF_ENABLE
    VX_perf_memsys_if perf_memsys_if();
`endif

    VX_mem_req_if #(
        .DATA_WIDTH (`DCACHE_MEM_DATA_WIDTH),
        .ADDR_WIDTH (`DCACHE_MEM_ADDR_WIDTH),
        .TAG_WIDTH  (`L1_MEM_TAG_WIDTH)
    ) mem_req_if();

    VX_mem_rsp_if #(
        .DATA_WIDTH (`DCACHE_MEM_DATA_WIDTH),
        .TAG_WIDTH  (`L1_MEM_TAG_WIDTH)
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

    VX_dcache_req_if #(
        .NUM_REQS  (`DCACHE_NUM_REQS), 
        .WORD_SIZE (`DCACHE_WORD_SIZE), 
        .TAG_WIDTH (`DCACHE_CORE_TAG_WIDTH)
    ) dcache_req_if();

    VX_dcache_rsp_if #(
        .NUM_REQS  (`DCACHE_NUM_REQS), 
        .WORD_SIZE (`DCACHE_WORD_SIZE), 
        .TAG_WIDTH (`DCACHE_CORE_TAG_WIDTH)
    ) dcache_rsp_if();
    
    VX_icache_req_if #(
        .WORD_SIZE (`ICACHE_WORD_SIZE), 
        .TAG_WIDTH (`ICACHE_CORE_TAG_WIDTH)
    ) icache_req_if();

    VX_icache_rsp_if #(
        .WORD_SIZE (`ICACHE_WORD_SIZE), 
        .TAG_WIDTH (`ICACHE_CORE_TAG_WIDTH)
    ) icache_rsp_if();

`ifdef EXT_TEX_ENABLE
    VX_dcache_req_if #(
        .NUM_REQS  (`NUM_THREADS), 
        .WORD_SIZE (4), 
        .TAG_WIDTH (`DCACHE_TAG_WIDTH)
    ) tcache_req_if();

    VX_dcache_rsp_if #(
        .NUM_REQS  (`NUM_THREADS), 
        .WORD_SIZE (4), 
        .TAG_WIDTH (`DCACHE_TAG_WIDTH)
    ) tcache_rsp_if();
`endif
    
    VX_pipeline #(
        .CORE_ID(CORE_ID)
    ) pipeline (
        `SCOPE_BIND_VX_core_pipeline
    `ifdef PERF_ENABLE
        .perf_memsys_if (perf_memsys_if),
    `endif

        .clk            (clk),
        .reset          (reset),

        // dcache interface
        .dcache_req_if  (dcache_req_if),
        .dcache_rsp_if  (dcache_rsp_if),

        // icache interface
        .icache_req_if  (icache_req_if),
        .icache_rsp_if  (icache_rsp_if),

    `ifdef EXT_TEX_ENABLE
        .tex_csr_if     (tex_csr_if),
        .tcache_req_if  (tcache_req_if),
        .tcache_rsp_if  (tcache_rsp_if),
    `endif

        // Status
        .busy           (busy)
    );  

    //--

    VX_mem_unit #(
        .CORE_ID(CORE_ID)
    ) mem_unit (
        `SCOPE_BIND_VX_core_mem_unit
    `ifdef PERF_ENABLE
        .perf_memsys_if (perf_memsys_if),
    `endif

        .clk            (clk),
        .reset          (reset),

        // dcache interface
        .dcache_req_if  (dcache_req_if),
        .dcache_rsp_if  (dcache_rsp_if),
        
        // icache interface
        .icache_req_if  (icache_req_if),
        .icache_rsp_if  (icache_rsp_if),

    `ifdef EXT_TEX_ENABLE
        // tcache interface
        .tcache_req_if (tcache_req_if),
        .tcache_rsp_if (tcache_rsp_if),
    `endif

        // Memory
        .mem_req_if     (mem_req_if),
        .mem_rsp_if     (mem_rsp_if)
    );
    
endmodule





