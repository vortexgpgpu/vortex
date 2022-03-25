`include "VX_define.vh"

module VX_core #( 
    parameter CORE_ID = 0
) (        
    `SCOPE_IO_VX_core
    
    // Clock
    input  wire             clk,
    input  wire             reset,

`ifdef EXT_TEX_ENABLE
    VX_tex_dcr_if.slave     tex_dcr_if,
`endif
`ifdef EXT_RASTER_ENABLE        
    VX_raster_req_if        raster_req_if,
`endif
`ifdef EXT_ROP_ENABLE        
    VX_rop_req_if           rop_req_if,
`endif

    // Memory
    VX_mem_req_if.master    mem_req_if,
    VX_mem_rsp_if.slave     mem_rsp_if,

    // simulation helper signals
    output wire             sim_ebreak,
    output wire [`NUM_REGS-1:0][31:0] sim_last_wb_value,

    // Status
    output wire             busy
);
`ifdef PERF_ENABLE
    VX_perf_memsys_if perf_memsys_if();
`endif

    VX_cache_req_if #(
        .NUM_REQS  (`DCACHE_NUM_REQS), 
        .WORD_SIZE (`DCACHE_WORD_SIZE), 
        .TAG_WIDTH (`DCACHE_TAG_WIDTH)
    ) dcache_req_if();

    VX_cache_rsp_if #(
        .NUM_REQS  (`DCACHE_NUM_REQS), 
        .WORD_SIZE (`DCACHE_WORD_SIZE), 
        .TAG_WIDTH (`DCACHE_TAG_WIDTH)
    ) dcache_rsp_if();
    
    VX_cache_req_if #(
        .NUM_REQS  (`ICACHE_NUM_REQS), 
        .WORD_SIZE (`ICACHE_WORD_SIZE), 
        .TAG_WIDTH (`ICACHE_TAG_WIDTH)
    ) icache_req_if();

    VX_cache_rsp_if #(
        .NUM_REQS  (`ICACHE_NUM_REQS), 
        .WORD_SIZE (`ICACHE_WORD_SIZE), 
        .TAG_WIDTH (`ICACHE_TAG_WIDTH)
    ) icache_rsp_if();

`ifdef EXT_TEX_ENABLE
    VX_cache_req_if #(
        .NUM_REQS  (`TCACHE_NUM_REQS), 
        .WORD_SIZE (`TCACHE_WORD_SIZE), 
        .TAG_WIDTH (`TCACHE_TAG_WIDTH)
    ) tcache_req_if();

    VX_cache_rsp_if #(
        .NUM_REQS  (`TCACHE_NUM_REQS), 
        .WORD_SIZE (`TCACHE_WORD_SIZE), 
        .TAG_WIDTH (`TCACHE_TAG_WIDTH)
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
        .tex_dcr_if     (tex_dcr_if),
        .tcache_req_if  (tcache_req_if),
        .tcache_rsp_if  (tcache_rsp_if),
    `endif
    `ifdef EXT_RASTER_ENABLE        
        .raster_req_if  (raster_req_if),
    `endif
    `ifdef EXT_ROP_ENABLE        
        .rop_req_if     (rop_req_if),
    `endif

        .sim_ebreak     (sim_ebreak),
        .sim_last_wb_value (sim_last_wb_value),

        // Status
        .busy           (busy)
    );

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
        .tcache_req_if  (tcache_req_if),
        .tcache_rsp_if  (tcache_rsp_if),
    `endif

        // Memory
        .mem_req_if     (mem_req_if),
        .mem_rsp_if     (mem_rsp_if)
    );
    
endmodule





