`include "VX_define.vh"
`include "VX_gpu_types.vh"

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
import VX_gpu_types::*;
`IGNORE_WARNINGS_END

module Vortex (
    `SCOPE_IO_DECL

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

    // Status
    output wire                             busy
);

`ifdef PERF_ENABLE
    VX_mem_perf_if mem_perf_if[`NUM_CLUSTERS]();
    VX_mem_perf_if perf_memsys_total_if();    
    VX_cache_perf_if perf_l3cache_if();    
`endif

    VX_mem_bus_if #(
        .DATA_WIDTH (L3_MEM_DATA_WIDTH),
        .TAG_WIDTH  (L3_MEM_TAG_WIDTH)
    ) mem_bus_if();

    assign mem_req_valid = mem_bus_if.req_valid;
    assign mem_req_rw    = mem_bus_if.req_rw;
    assign mem_req_byteen= mem_bus_if.req_byteen;
    assign mem_req_addr  = mem_bus_if.req_addr;
    assign mem_req_data  = mem_bus_if.req_data;
    assign mem_req_tag   = mem_bus_if.req_tag;
    assign mem_bus_if.req_ready = mem_req_ready;

    assign mem_bus_if.rsp_valid = mem_rsp_valid;
    assign mem_bus_if.rsp_data  = mem_rsp_data;
    assign mem_bus_if.rsp_tag   = mem_rsp_tag;
    assign mem_rsp_ready = mem_bus_if.rsp_ready;

    wire mem_req_fire = mem_req_valid && mem_req_ready;
    wire mem_rsp_fire = mem_rsp_valid && mem_rsp_ready;
    `UNUSED_VAR (mem_req_fire)
    `UNUSED_VAR (mem_rsp_fire)

`ifdef EXT_TEX_ENABLE
`ifdef PERF_ENABLE    
    VX_tex_perf_if      perf_tex_if[`NUM_CLUSTERS]();
    VX_cache_perf_if    perf_tcache_if[`NUM_CLUSTERS]();
    VX_tex_perf_if      perf_tex_total_if();
    VX_cache_perf_if    perf_tcache_total_if();
    `PERF_TEX_ADD (perf_tex_total_if, perf_tex_if, `NUM_CLUSTERS);
    `PERF_CACHE_ADD (perf_tcache_total_if, perf_tcache_if, `NUM_CLUSTERS);
`endif
`endif

`ifdef EXT_RASTER_ENABLE
`ifdef PERF_ENABLE    
    VX_raster_perf_if   perf_raster_if[`NUM_CLUSTERS]();
    VX_cache_perf_if    perf_rcache_if[`NUM_CLUSTERS]();
    VX_raster_perf_if   perf_raster_total_if();
    VX_cache_perf_if    perf_rcache_total_if();
    `PERF_RASTER_ADD (perf_raster_total_if, perf_raster_if, `NUM_CLUSTERS);
    `PERF_CACHE_ADD (perf_rcache_total_if, perf_rcache_if, `NUM_CLUSTERS);
`endif
`endif

`ifdef EXT_ROP_ENABLE
`ifdef PERF_ENABLE    
    VX_rop_perf_if      perf_rop_if[`NUM_CLUSTERS]();
    VX_cache_perf_if    perf_ocache_if[`NUM_CLUSTERS]();
    VX_rop_perf_if      perf_rop_total_if();
    VX_cache_perf_if    perf_ocache_total_if();
    `PERF_ROP_ADD (perf_rop_total_if, perf_rop_if, `NUM_CLUSTERS);
    `PERF_CACHE_ADD (perf_ocache_total_if, perf_ocache_if, `NUM_CLUSTERS);
`endif
`endif

    wire sim_ebreak /* verilator public */;
    wire [`NUM_REGS-1:0][`XLEN-1:0] sim_wb_value /* verilator public */;    
    wire [`NUM_CLUSTERS-1:0] per_cluster_sim_ebreak;
    wire [`NUM_CLUSTERS-1:0][`NUM_REGS-1:0][`XLEN-1:0] per_cluster_sim_wb_value;
    assign sim_ebreak = per_cluster_sim_ebreak[0];
    assign sim_wb_value = per_cluster_sim_wb_value[0];
    `UNUSED_VAR (per_cluster_sim_ebreak)
    `UNUSED_VAR (per_cluster_sim_wb_value)

    VX_mem_bus_if #(
        .DATA_WIDTH (L2_MEM_DATA_WIDTH),
        .TAG_WIDTH  (L2_MEM_TAG_WIDTH)
    ) per_cluster_mem_bus_if[`NUM_CLUSTERS]();        

    VX_dcr_write_if dcr_write_if();
    assign dcr_write_if.valid = dcr_wr_valid;
    assign dcr_write_if.addr  = dcr_wr_addr;
    assign dcr_write_if.data  = dcr_wr_data;

    wire [`NUM_CLUSTERS-1:0] per_cluster_busy;

    `SCOPE_IO_SWITCH (`NUM_CLUSTERS)

    // Generate all clusters
    for (genvar i = 0; i < `NUM_CLUSTERS; ++i) begin

        `RESET_RELAY (cluster_reset, reset);

        `BUFFER_DCR_WRITE_IF (cluster_dcr_write_if, dcr_write_if, (`NUM_CLUSTERS > 1));

        VX_cluster #(
            .CLUSTER_ID (i)
        ) cluster (
            `SCOPE_IO_BIND      (i)

            .clk                (clk),
            .reset              (cluster_reset),

        `ifdef PERF_ENABLE
            .mem_perf_if        (mem_perf_if[i]),
            .perf_memsys_total_if (perf_memsys_total_if),
        `endif
            
            .dcr_write_if       (cluster_dcr_write_if),

        `ifdef EXT_TEX_ENABLE
        `ifdef PERF_ENABLE
            .perf_tex_if        (perf_tex_if[i]),
            .perf_tcache_if     (perf_tcache_if[i]), 
            .perf_tex_total_if  (perf_tex_total_if),                       
            .perf_tcache_total_if (perf_tcache_total_if),
        `endif
        `endif

        `ifdef EXT_RASTER_ENABLE
        `ifdef PERF_ENABLE
            .perf_raster_if     (perf_raster_if[i]),
            .perf_rcache_if     (perf_rcache_if[i]), 
            .perf_raster_total_if (perf_raster_total_if),                       
            .perf_rcache_total_if (perf_rcache_total_if),
        `endif
        `endif
        
        `ifdef EXT_ROP_ENABLE
        `ifdef PERF_ENABLE
            .perf_rop_if        (perf_rop_if[i]),
            .perf_ocache_if     (perf_ocache_if[i]),    
            .perf_rop_total_if  (perf_rop_total_if),                    
            .perf_ocache_total_if (perf_ocache_total_if),
        `endif
        `endif

            .mem_bus_if         (per_cluster_mem_bus_if[i]),

            .sim_ebreak         (per_cluster_sim_ebreak[i]),
            .sim_wb_value       (per_cluster_sim_wb_value[i]),

            .busy               (per_cluster_busy[i])
        );
    end

    `BUFFER_BUSY ((| per_cluster_busy), (`NUM_CLUSTERS > 1));

    `RESET_RELAY (l3_reset, reset);

    VX_cache_wrap #(
        .INSTANCE_ID    ("l3cache"),
        .CACHE_SIZE     (`L3_CACHE_SIZE),
        .LINE_SIZE      (`L3_LINE_SIZE),
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
        .cache_perf_if  (perf_l3cache_if),
    `endif

        .core_bus_if    (per_cluster_mem_bus_if),
        .mem_bus_if     (mem_bus_if)
    );

`ifdef PERF_ENABLE

    `REDUCE_ADD (perf_memsys_total_if, mem_perf_if, icache_reads, `PERF_CTR_BITS, `NUM_CLUSTERS);
    `REDUCE_ADD (perf_memsys_total_if, mem_perf_if, icache_read_misses, `PERF_CTR_BITS, `NUM_CLUSTERS);
    `REDUCE_ADD (perf_memsys_total_if, mem_perf_if, dcache_reads, `PERF_CTR_BITS, `NUM_CLUSTERS);
    `REDUCE_ADD (perf_memsys_total_if, mem_perf_if, dcache_writes, `PERF_CTR_BITS, `NUM_CLUSTERS);
    `REDUCE_ADD (perf_memsys_total_if, mem_perf_if, dcache_read_misses, `PERF_CTR_BITS, `NUM_CLUSTERS);
    `REDUCE_ADD (perf_memsys_total_if, mem_perf_if, dcache_write_misses, `PERF_CTR_BITS, `NUM_CLUSTERS);
    `REDUCE_ADD (perf_memsys_total_if, mem_perf_if, dcache_bank_stalls, `PERF_CTR_BITS, `NUM_CLUSTERS);
    `REDUCE_ADD (perf_memsys_total_if, mem_perf_if, dcache_mshr_stalls, `PERF_CTR_BITS, `NUM_CLUSTERS);
    `REDUCE_ADD (perf_memsys_total_if, mem_perf_if, smem_reads, `PERF_CTR_BITS, `NUM_CLUSTERS);
    `REDUCE_ADD (perf_memsys_total_if, mem_perf_if, smem_writes, `PERF_CTR_BITS, `NUM_CLUSTERS);
    `REDUCE_ADD (perf_memsys_total_if, mem_perf_if, smem_bank_stalls, `PERF_CTR_BITS, `NUM_CLUSTERS);
    `REDUCE_ADD (perf_memsys_total_if, mem_perf_if, l2cache_reads, `PERF_CTR_BITS, `NUM_CLUSTERS);
    `REDUCE_ADD (perf_memsys_total_if, mem_perf_if, l2cache_writes, `PERF_CTR_BITS, `NUM_CLUSTERS);
    `REDUCE_ADD (perf_memsys_total_if, mem_perf_if, l2cache_read_misses, `PERF_CTR_BITS, `NUM_CLUSTERS);
    `REDUCE_ADD (perf_memsys_total_if, mem_perf_if, l2cache_write_misses, `PERF_CTR_BITS, `NUM_CLUSTERS);
    `REDUCE_ADD (perf_memsys_total_if, mem_perf_if, l2cache_bank_stalls, `PERF_CTR_BITS, `NUM_CLUSTERS);
    `REDUCE_ADD (perf_memsys_total_if, mem_perf_if, l2cache_mshr_stalls, `PERF_CTR_BITS, `NUM_CLUSTERS);

`ifdef L3_ENABLE
    assign perf_memsys_total_if.l3cache_reads       = perf_l3cache_if.reads;
    assign perf_memsys_total_if.l3cache_writes      = perf_l3cache_if.writes;
    assign perf_memsys_total_if.l3cache_read_misses = perf_l3cache_if.read_misses;
    assign perf_memsys_total_if.l3cache_write_misses= perf_l3cache_if.write_misses;
    assign perf_memsys_total_if.l3cache_bank_stalls = perf_l3cache_if.bank_stalls;
    assign perf_memsys_total_if.l3cache_mshr_stalls = perf_l3cache_if.mshr_stalls;
`else
    assign perf_memsys_total_if.l3cache_reads       = '0;
    assign perf_memsys_total_if.l3cache_writes      = '0;
    assign perf_memsys_total_if.l3cache_read_misses = '0;
    assign perf_memsys_total_if.l3cache_write_misses= '0;
    assign perf_memsys_total_if.l3cache_bank_stalls = '0;
    assign perf_memsys_total_if.l3cache_mshr_stalls = '0;
`endif

    reg [`PERF_CTR_BITS-1:0] perf_mem_pending_reads;

    always @(posedge clk) begin
        if (reset) begin
            perf_mem_pending_reads <= '0;
        end else begin
            perf_mem_pending_reads <= $signed(perf_mem_pending_reads) + 
                `PERF_CTR_BITS'($signed(2'(mem_req_fire && ~mem_bus_if.req_rw) - 2'(mem_rsp_fire)));
        end
    end
    
    reg [`PERF_CTR_BITS-1:0] perf_mem_reads;
    reg [`PERF_CTR_BITS-1:0] perf_mem_writes;
    reg [`PERF_CTR_BITS-1:0] perf_mem_lat;

    always @(posedge clk) begin
        if (reset) begin       
            perf_mem_reads  <= '0;
            perf_mem_writes <= '0;
            perf_mem_lat    <= '0;
        end else begin  
            if (mem_req_fire && ~mem_bus_if.req_rw) begin
                perf_mem_reads <= perf_mem_reads + `PERF_CTR_BITS'(1);
            end
            if (mem_req_fire && mem_bus_if.req_rw) begin
                perf_mem_writes <= perf_mem_writes + `PERF_CTR_BITS'(1);
            end      
            perf_mem_lat <= perf_mem_lat + perf_mem_pending_reads;
        end
    end

    assign perf_memsys_total_if.mem_reads   = perf_mem_reads;       
    assign perf_memsys_total_if.mem_writes  = perf_mem_writes;
    assign perf_memsys_total_if.mem_latency = perf_mem_lat;
    
`endif

`ifdef DBG_TRACE_CORE_MEM
    always @(posedge clk) begin
        if (mem_req_fire) begin
            if (mem_req_rw)
                `TRACE(1, ("%d: MEM Wr Req: addr=0x%0h, tag=0x%0h, byteen=0x%0h data=0x%0h\n", $time, `TO_FULL_ADDR(mem_req_addr), mem_req_tag, mem_req_byteen, mem_req_data));
            else
                `TRACE(1, ("%d: MEM Rd Req: addr=0x%0h, tag=0x%0h, byteen=0x%0h\n", $time, `TO_FULL_ADDR(mem_req_addr), mem_req_tag, mem_req_byteen));
        end
        if (mem_rsp_fire) begin
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
