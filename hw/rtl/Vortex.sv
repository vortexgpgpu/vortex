`include "VX_define.vh"
`include "VX_cache_types.vh"

`ifdef EXT_TEX_ENABLE
`include "VX_tex_define.vh"
`endif

`ifdef EXT_RASTER_ENABLE
`include "VX_raster_define.vh"
`endif

`ifdef EXT_ROP_ENABLE
`include "VX_rop_define.vh"
`endif

`IGNORE_WARNINGS_BEGIN
import VX_cache_types::*;
`IGNORE_WARNINGS_END

module Vortex (
    `SCOPE_IO_Vortex

    // Clock
    input  wire                             clk,
    input  wire                             reset,

    // Memory request
    output wire                             mem_req_valid,
    output wire                             mem_req_rw,    
    output wire [`VX_MEM_BYTEEN_WIDTH-1:0]  mem_req_byteen,    
    output wire [`VX_MEM_ADDR_WIDTH-1:0]    mem_req_addr,
    output wire [`VX_MEM_DATA_WIDTH-1:0]    mem_req_data,
    output wire [`VX_MEM_TAG_WIDTH-1:0]     mem_req_tag,
    input  wire                             mem_req_ready,

    // Memory response    
    input wire                              mem_rsp_valid,        
    input wire [`VX_MEM_DATA_WIDTH-1:0]     mem_rsp_data,
    input wire [`VX_MEM_TAG_WIDTH-1:0]      mem_rsp_tag,
    output wire                             mem_rsp_ready,

    // DCR write request
    input  wire                             dcr_wr_valid,
    input  wire [`VX_DCR_ADDR_WIDTH-1:0]    dcr_wr_addr,
    input  wire [`VX_DCR_DATA_WIDTH-1:0]    dcr_wr_data,
    output wire                             dcr_wr_ready,

    // Control / status
    input wire                              start,
    output wire                             busy
);

`ifdef PERF_ENABLE
    VX_perf_memsys_if perf_memsys_if[`NUM_CLUSTERS]();
    VX_perf_memsys_if perf_memsys_total_if();
    VX_perf_cache_if  perf_l3cache_if();
    `PERF_MEMSYS_ADD (perf_memsys_total_if, perf_memsys_if, `NUM_CLUSTERS);
`endif

    VX_mem_req_if #(
        .DATA_WIDTH (L3_MEM_DATA_WIDTH),
        .TAG_WIDTH  (L3_MEM_TAG_WIDTH)
    ) mem_req_if();

    VX_mem_rsp_if #(
        .DATA_WIDTH (L3_MEM_DATA_WIDTH),
        .TAG_WIDTH  (L3_MEM_TAG_WIDTH)
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

    VX_dcr_base_if #(
    ) dcr_base_if();

`ifdef EXT_TEX_ENABLE
`ifdef PERF_ENABLE    
    VX_tex_perf_if      perf_tex_if[`NUM_CLUSTERS]();
    VX_perf_cache_if    perf_tcache_if[`NUM_CLUSTERS]();
    VX_tex_perf_if      perf_tex_total_if();
    VX_perf_cache_if    perf_tcache_total_if();
    `PERF_TEX_ADD (perf_tex_total_if, perf_tex_if, `NUM_CLUSTERS);
    `PERF_CACHE_ADD (perf_tcache_total_if, perf_tcache_if, `NUM_CLUSTERS);
`endif
    VX_tex_dcr_if #(
        .NUM_STAGES (`TEX_STAGE_COUNT)
    ) tex_dcr_if();
`endif

`ifdef EXT_RASTER_ENABLE
`ifdef PERF_ENABLE    
    VX_raster_perf_if   perf_raster_if[`NUM_CLUSTERS]();
    VX_perf_cache_if    perf_rcache_if[`NUM_CLUSTERS]();
    VX_raster_perf_if   perf_raster_total_if();
    VX_perf_cache_if    perf_rcache_total_if();
    `PERF_RASTER_ADD (perf_raster_total_if, perf_raster_if, `NUM_CLUSTERS);
    `PERF_CACHE_ADD (perf_rcache_total_if, perf_rcache_if, `NUM_CLUSTERS);
`endif
    VX_raster_dcr_if    raster_dcr_if();
`endif

`ifdef EXT_ROP_ENABLE
`ifdef PERF_ENABLE    
    VX_rop_perf_if      perf_rop_if[`NUM_CLUSTERS]();
    VX_perf_cache_if    perf_ocache_if[`NUM_CLUSTERS]();
    VX_rop_perf_if      perf_rop_total_if();
    VX_perf_cache_if    perf_ocache_total_if();
    `PERF_ROP_ADD (perf_rop_total_if, perf_rop_if, `NUM_CLUSTERS);
    `PERF_CACHE_ADD (perf_ocache_total_if, perf_ocache_if, `NUM_CLUSTERS);
`endif
    VX_rop_dcr_if       rop_dcr_if();
`endif

    `RESET_RELAY (dcr_reset, reset);
    
    VX_dcr_data dcr_data(
        .clk          (clk),
        .reset        (dcr_reset),
        .dcr_base_if  (dcr_base_if),
    `ifdef EXT_TEX_ENABLE
        .tex_dcr_if    (tex_dcr_if),
    `endif
    `ifdef EXT_RASTER_ENABLE
        .raster_dcr_if (raster_dcr_if),  
    `endif
    `ifdef EXT_ROP_ENABLE
        .rop_dcr_if    (rop_dcr_if),
    `endif
        .dcr_wr_valid (dcr_wr_valid),
        .dcr_wr_addr  (dcr_wr_addr),
        .dcr_wr_data  (dcr_wr_data),
        .dcr_wr_ready (dcr_wr_ready)
    );

    wire sim_ebreak /* verilator public */;
    wire [`NUM_REGS-1:0][31:0] sim_wb_value /* verilator public */;    
    wire [`NUM_CLUSTERS-1:0] per_cluster_sim_ebreak;
    wire [`NUM_CLUSTERS-1:0][`NUM_REGS-1:0][31:0] per_cluster_sim_wb_value;
    assign sim_ebreak = per_cluster_sim_ebreak[0];
    assign sim_wb_value = per_cluster_sim_wb_value[0];
    `UNUSED_VAR (per_cluster_sim_ebreak)
    `UNUSED_VAR (per_cluster_sim_wb_value)

    // also reset device on start
    wire reset_or_start = reset || start;

    VX_mem_req_if #(
        .DATA_WIDTH (L2_MEM_DATA_WIDTH),
        .TAG_WIDTH  (L2_MEM_TAG_WIDTH)
    ) per_cluster_mem_req_if[`NUM_CLUSTERS]();        

    VX_mem_rsp_if #(
        .DATA_WIDTH (L2_MEM_DATA_WIDTH),
        .TAG_WIDTH  (L2_MEM_TAG_WIDTH)
    ) per_cluster_mem_rsp_if[`NUM_CLUSTERS]();

    wire [`NUM_CLUSTERS-1:0] per_cluster_busy;

    // Generate all clusters
    for (genvar i = 0; i < `NUM_CLUSTERS; ++i) begin

        `RESET_RELAY (cluster_reset, reset_or_start);

        VX_cluster #(
            .CLUSTER_ID (i)
        ) cluster (
            `SCOPE_BIND_Vortex_cluster(i)

            .clk                (clk),
            .reset              (cluster_reset),

        `ifdef PERF_ENABLE
            .perf_memsys_if     (perf_memsys_if[i]),
            .perf_memsys_total_if (perf_memsys_total_if),
        `endif
            
            .dcr_base_if        (dcr_base_if),

        `ifdef EXT_TEX_ENABLE
        `ifdef PERF_ENABLE
            .perf_tex_if        (perf_tex_if[i]),
            .perf_tcache_if     (perf_tcache_if[i]), 
            .perf_tex_total_if  (perf_tex_total_if),                       
            .perf_tcache_total_if (perf_tcache_total_if),
        `endif
            .tex_dcr_if         (tex_dcr_if),
        `endif

        `ifdef EXT_RASTER_ENABLE
        `ifdef PERF_ENABLE
            .perf_raster_if     (perf_raster_if[i]),
            .perf_rcache_if     (perf_rcache_if[i]), 
            .perf_raster_total_if (perf_raster_total_if),                       
            .perf_rcache_total_if (perf_rcache_total_if),
        `endif
            .raster_dcr_if      (raster_dcr_if),  
        `endif
        
        `ifdef EXT_ROP_ENABLE
        `ifdef PERF_ENABLE
            .perf_rop_if        (perf_rop_if[i]),
            .perf_ocache_if     (perf_ocache_if[i]),    
            .perf_rop_total_if  (perf_rop_total_if),                    
            .perf_ocache_total_if (perf_ocache_total_if),
        `endif
            .rop_dcr_if         (rop_dcr_if),
        `endif

            .mem_req_if         (per_cluster_mem_req_if[i]),
            .mem_rsp_if         (per_cluster_mem_rsp_if[i]),

            .sim_ebreak         (per_cluster_sim_ebreak[i]),
            .sim_wb_value       (per_cluster_sim_wb_value[i]),

            .busy               (per_cluster_busy[i])
        );
    end

    assign busy = (| per_cluster_busy);

    `RESET_RELAY (l3_reset, reset_or_start);

    VX_cache_wrap #(
        .INSTANCE_ID    ("l3cache"),
        .CACHE_SIZE     (`L3_CACHE_SIZE),
        .LINE_SIZE      (L3_LINE_SIZE),
        .NUM_BANKS      (`L3_NUM_BANKS),
        .NUM_WAYS       (`L3_NUM_WAYS),
        .NUM_PORTS      (`L3_NUM_PORTS),
        .WORD_SIZE      (L3_WORD_SIZE),
        .NUM_REQS       (L3_NUM_REQS),
        .CREQ_SIZE      (`L3_CREQ_SIZE),
        .CRSQ_SIZE      (`L3_CRSQ_SIZE),
        .MSHR_SIZE      (`L3_MSHR_SIZE),
        .MRSQ_SIZE      (`L3_MRSQ_SIZE),
        .MREQ_SIZE      (`L3_MREQ_SIZE),
        .TAG_WIDTH      (L2_MEM_TAG_WIDTH),
        .WRITE_ENABLE   (1),
        .UUID_WIDTH     (`UUID_BITS),  
        .CORE_OUT_REG   (3),
        .MEM_OUT_REG    (3),
        .NC_ENABLE      (1),
        .PASSTHRU       (!`L3_ENABLED)
    ) l3cache_wrap (
        .clk            (clk),
        .reset          (l3_reset),

    `ifdef PERF_ENABLE
        .perf_cache_if  (perf_l3cache_if),
    `endif

        .core_req_if    (per_cluster_mem_req_if),
        .core_rsp_if    (per_cluster_mem_rsp_if),
        .mem_req_if     (mem_req_if),
        .mem_rsp_if     (mem_rsp_if)
    );

    `SCOPE_ASSIGN (reset, reset);
    `SCOPE_ASSIGN (mem_req_fire, mem_req_valid && mem_req_ready);
    `SCOPE_ASSIGN (mem_req_addr, `TO_FULL_ADDR(mem_req_addr));
    `SCOPE_ASSIGN (mem_req_rw,   mem_req_rw);
    `SCOPE_ASSIGN (mem_rsp_fire, mem_rsp_valid && mem_rsp_ready);
    `SCOPE_ASSIGN (busy, busy);

`ifdef DBG_TRACE_CORE_MEM
    always @(posedge clk) begin
        if (mem_req_valid && mem_req_ready) begin
            if (mem_req_rw)
                `TRACE(1, ("%d: MEM Wr Req: addr=0x%0h, tag=0x%0h, byteen=0x%0h data=0x%0h\n", $time, `TO_FULL_ADDR(mem_req_addr), mem_req_tag, mem_req_byteen, mem_req_data));
            else
                `TRACE(1, ("%d: MEM Rd Req: addr=0x%0h, tag=0x%0h, byteen=0x%0h\n", $time, `TO_FULL_ADDR(mem_req_addr), mem_req_tag, mem_req_byteen));
        end
        if (mem_rsp_valid && mem_rsp_ready) begin
            `TRACE(1, ("%d: MEM Rsp: tag=0x%0h, data=0x%0h\n", $time, mem_rsp_tag, mem_rsp_data));
        end
    end
`endif

`ifdef SIMULATION
    always @(posedge clk) begin
        $fflush(); // flush stdout buffer
    end
`endif

endmodule
