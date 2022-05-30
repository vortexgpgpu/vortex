`include "VX_define.vh"
`include "VX_gpu_types.vh"

`IGNORE_WARNINGS_BEGIN
import VX_gpu_types::*;
`IGNORE_WARNINGS_END

module VX_core #( 
    parameter CORE_ID = 0
) (        
    `SCOPE_IO_VX_core
    
    // Clock
    input  wire             clk,
    input  wire             reset,

    VX_dcr_base_if.slave    dcr_base_if,

`ifdef EXT_TEX_ENABLE
    VX_tex_dcr_if.slave     tex_dcr_if,
`endif
`ifdef EXT_RASTER_ENABLE        
    VX_raster_req_if        raster_req_if,
`ifdef PERF_ENABLE
    VX_raster_perf_if.slave raster_perf_if,
    VX_perf_cache_if.slave  perf_rcache_if,
`endif
`endif
`ifdef EXT_ROP_ENABLE        
    VX_rop_req_if           rop_req_if,
`ifdef PERF_ENABLE
    VX_rop_perf_if.slave    rop_perf_if,
    VX_perf_cache_if.slave  perf_ocache_if,
`endif
`endif

    // Memory
    VX_mem_req_if.master    mem_req_if,
    VX_mem_rsp_if.slave     mem_rsp_if,

    // simulation helper signals
    output wire             sim_ebreak,
    output wire [`NUM_REGS-1:0][31:0] sim_wb_value,

    // Status
    output wire             busy
);

    base_dcrs_t base_dcrs;
    always @(posedge clk) begin
        base_dcrs <= dcr_base_if.data;
    end

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
`ifdef PERF_ENABLE
    VX_perf_cache_if perf_tcache_if();
`endif
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

        .base_dcrs      (base_dcrs),

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
    `ifdef PERF_ENABLE
        .perf_tcache_if (perf_tcache_if),
    `endif
    `endif
    `ifdef EXT_RASTER_ENABLE        
        .raster_req_if  (raster_req_if),
    `ifdef PERF_ENABLE
        .raster_perf_if (raster_perf_if),
        .perf_rcache_if (perf_rcache_if),
    `endif
    `endif
    `ifdef EXT_ROP_ENABLE        
        .rop_req_if     (rop_req_if),
    `ifdef PERF_ENABLE
        .rop_perf_if    (rop_perf_if),
        .perf_ocache_if (perf_ocache_if),
    `endif
    `endif

        .sim_ebreak     (sim_ebreak),
        .sim_wb_value   (sim_wb_value),

        // Status
        .busy           (busy)
    );

    VX_mem_unit #(
        .CORE_ID(CORE_ID)
    ) mem_unit (
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
    `ifdef PERF_ENABLE
        .perf_tcache_if  (perf_tcache_if),
    `endif
    `endif

        // Memory
        .mem_req_if     (mem_req_if),
        .mem_rsp_if     (mem_rsp_if)
    );
    
endmodule





